"""
Training Script
===============
Fine-tunes Mask R-CNN on the football player segmentation dataset.

Run with:
    python -m src.train                          # uses configs/config.yaml
    python -m src.train --config my_config.yaml  # custom config

What happens during one training epoch
---------------------------------------
  1. A batch of 2 images is loaded with ground-truth boxes, masks, and labels.
  2. The model forward pass returns a dict of four losses:
       loss_classifier  — cross-entropy on the class prediction per ROI
       loss_box_reg     — smooth-L1 loss on the bounding-box regression
       loss_mask        — binary cross-entropy on the predicted 28×28 masks
       loss_objectness  — binary cross-entropy on the RPN proposals
  3. The four losses are summed and back-propagated.
  4. The optimizer updates all weights.
  5. The learning-rate scheduler steps at the end of each epoch.
"""

import argparse
import os

import torch
import torch.utils.data
import yaml
from tqdm import tqdm

from src.dataset import FootballDataset, collate_fn, get_transforms
from src.evaluate import evaluate_model
from src.model import count_parameters, get_model
from src.utils import AverageMeter, load_checkpoint, save_checkpoint, set_seed

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


# ---------------------------------------------------------------------------
# Single epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, optimizer, data_loader, device, epoch, writer=None):
    """
    Train for one complete pass over the training set.

    Returns:
        avg_loss : mean total loss across all batches in this epoch
    """
    model.train()

    total_loss   = AverageMeter()
    loss_cls     = AverageMeter()
    loss_box     = AverageMeter()
    loss_mask    = AverageMeter()
    loss_obj     = AverageMeter()

    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}", leave=False, unit="batch")

    for images, targets in pbar:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # In training mode, Mask R-CNN returns a loss dict, not predictions.
        loss_dict = model(images, targets)

        losses = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        n = len(images)
        total_loss.update(losses.item(), n)
        loss_cls .update(loss_dict.get("loss_classifier", torch.tensor(0.0)).item(), n)
        loss_box .update(loss_dict.get("loss_box_reg",    torch.tensor(0.0)).item(), n)
        loss_mask.update(loss_dict.get("loss_mask",       torch.tensor(0.0)).item(), n)
        loss_obj .update(loss_dict.get("loss_objectness", torch.tensor(0.0)).item(), n)

        pbar.set_postfix(
            loss=f"{total_loss.avg:.4f}",
            cls=f"{loss_cls.avg:.3f}",
            box=f"{loss_box.avg:.3f}",
            mask=f"{loss_mask.avg:.3f}",
            rpn=f"{loss_obj.avg:.3f}",
        )

    if writer:
        writer.add_scalar("Loss/total",       total_loss.avg, epoch)
        writer.add_scalar("Loss/classifier",  loss_cls.avg,   epoch)
        writer.add_scalar("Loss/box_reg",     loss_box.avg,   epoch)
        writer.add_scalar("Loss/mask",        loss_mask.avg,  epoch)
        writer.add_scalar("Loss/objectness",  loss_obj.avg,   epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

    return total_loss.avg


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["data"]["seed"])

    # Use GPU if available; CUDA availability overrides the config value
    device = torch.device(
        cfg["training"]["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create output directories
    for key in ("checkpoint_dir", "log_dir", "visualization_dir"):
        os.makedirs(cfg["output"][key], exist_ok=True)

    writer = (
        __import__("torch.utils.tensorboard", fromlist=["SummaryWriter"])
        .SummaryWriter(cfg["output"]["log_dir"])
        if TENSORBOARD_AVAILABLE
        else None
    )

    # --- Datasets ---
    splits_dir  = cfg["data"]["splits_dir"]
    images_dir  = cfg["data"]["images_dir"]

    train_dataset = FootballDataset(
        images_dir=images_dir,
        annotation_file=os.path.join(splits_dir, "train.json"),
        transforms=get_transforms(train=True, cfg=cfg),
    )
    val_dataset = FootballDataset(
        images_dir=images_dir,
        annotation_file=os.path.join(splits_dir, "val.json"),
        transforms=get_transforms(train=False),
    )

    print(f"\nTraining images  : {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # --- Model ---
    model = get_model(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
    )
    model.to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"\nTotal parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")

    # --- Optimiser ---
    # SGD with momentum is the standard recipe for Mask R-CNN (from the
    # original Facebook paper).  AdamW can also work but requires a lower LR.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg["training"]["learning_rate"],
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # Drop LR by 10× at epoch step_size (e.g. epoch 10 of 20).
    # This lets the model converge quickly early on, then fine-tune precisely.
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["training"]["lr_step_size"],
        gamma=cfg["training"]["lr_gamma"],
    )

    best_ap  = 0.0
    start_ep = 0

    # Resume from an existing checkpoint if one exists
    resume_path = os.path.join(cfg["output"]["checkpoint_dir"], "latest.pth")
    if os.path.exists(resume_path):
        start_ep, best_ap = load_checkpoint(model, optimizer, resume_path, device)
        print(f"\nResumed from checkpoint: epoch {start_ep}, best AP {best_ap:.4f}")

    # --- Training loop ---
    epochs = cfg["training"]["epochs"]
    print(f"\n{'='*65}")
    print(f"Starting training: {epochs} epochs, batch size {cfg['training']['batch_size']}")
    print(f"{'='*65}\n")

    for epoch in range(start_ep, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}  (LR={optimizer.param_groups[0]['lr']:.6f})")
        print("-" * 50)

        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch, writer
        )
        lr_scheduler.step()

        print(f"  → Average training loss: {train_loss:.4f}")

        # Validate every 2 epochs and on the final epoch
        if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
            print("\n  Running COCO evaluation on validation set…")
            val_metrics = evaluate_model(model, val_loader, device)

            segm_ap   = val_metrics["segm_AP"]
            segm_ap50 = val_metrics["segm_AP50"]
            segm_ap75 = val_metrics["segm_AP75"]
            bbox_ap   = val_metrics["bbox_AP"]

            print(f"  Segmentation  AP     : {segm_ap:.4f}")
            print(f"  Segmentation  AP50   : {segm_ap50:.4f}")
            print(f"  Segmentation  AP75   : {segm_ap75:.4f}")
            print(f"  Detection     AP     : {bbox_ap:.4f}")

            if writer:
                writer.add_scalar("Val/segm_AP",   segm_ap,   epoch)
                writer.add_scalar("Val/segm_AP50",  segm_ap50, epoch)
                writer.add_scalar("Val/segm_AP75",  segm_ap75, epoch)
                writer.add_scalar("Val/bbox_AP",    bbox_ap,   epoch)

            if segm_ap > best_ap:
                best_ap = segm_ap
                save_checkpoint(
                    model, optimizer, epoch, best_ap,
                    os.path.join(cfg["output"]["checkpoint_dir"], "best.pth"),
                )
                print(f"  ★ New best model saved  (segm AP = {best_ap:.4f})")

        # Periodic checkpoint so we can resume after interruptions
        if (epoch + 1) % cfg["output"]["save_every"] == 0:
            save_checkpoint(
                model, optimizer, epoch, best_ap,
                os.path.join(cfg["output"]["checkpoint_dir"], "latest.pth"),
            )

    if writer:
        writer.close()

    print(f"\n{'='*65}")
    print(f"Training complete.  Best segmentation AP: {best_ap:.4f}")
    print(f"Best checkpoint: {cfg['output']['checkpoint_dir']}/best.pth")
    print(f"{'='*65}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Mask R-CNN for football player segmentation")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
