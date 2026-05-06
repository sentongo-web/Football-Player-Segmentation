"""
Test-Set Evaluation Script
===========================
Loads the best trained checkpoint and evaluates it on the held-out test split.

Run after training is complete:
    python scripts/test.py
    python scripts/test.py --checkpoint outputs/checkpoints/best.pth

Results are printed to the console and saved to outputs/test_results.json.
"""

import argparse
import json
import os
import sys

import torch
import yaml

# Allow running from the project root without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import FootballDataset, collate_fn, get_transforms
from src.evaluate import evaluate_model
from src.model import get_model
from src.utils import set_seed


def main(checkpoint_path: str, config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["data"]["seed"])

    device = torch.device(
        cfg["training"]["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # --- Dataset ---
    test_json = os.path.join(cfg["data"]["splits_dir"], "test.json")
    if not os.path.exists(test_json):
        raise FileNotFoundError(
            f"Test split not found: {test_json}\n"
            "Run  python scripts/prepare_data.py  first."
        )

    test_dataset = FootballDataset(
        images_dir=cfg["data"]["images_dir"],
        annotation_file=test_json,
        transforms=get_transforms(train=False),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_fn,
    )
    print(f"Test images : {len(test_dataset)}")

    # --- Model ---
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train the model first with  python -m src.train"
        )

    model = get_model(num_classes=cfg["model"]["num_classes"], pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    trained_epoch = checkpoint.get("epoch", "?")
    best_val_ap   = checkpoint.get("best_ap", float("nan"))
    print(f"Checkpoint  : {checkpoint_path}")
    print(f"Trained epoch  : {trained_epoch}")
    print(f"Best val segm AP : {best_val_ap:.4f}\n")

    # --- Evaluate ---
    print("Running COCO evaluation on test set…")
    metrics = evaluate_model(model, test_loader, device)

    # --- Report ---
    print("\n" + "=" * 50)
    print("Test-Set Results")
    print("=" * 50)
    print(f"  Segmentation  AP     : {metrics['segm_AP']:.4f}")
    print(f"  Segmentation  AP50   : {metrics['segm_AP50']:.4f}")
    print(f"  Segmentation  AP75   : {metrics['segm_AP75']:.4f}")
    print(f"  Detection     AP     : {metrics['bbox_AP']:.4f}")
    print(f"  Detection     AP50   : {metrics['bbox_AP50']:.4f}")
    print(f"  Detection     AP75   : {metrics['bbox_AP75']:.4f}")
    print("=" * 50)

    # --- Save ---
    results = {
        "checkpoint": checkpoint_path,
        "trained_epoch": trained_epoch,
        "best_val_segm_ap": best_val_ap,
        "test_metrics": metrics,
    }
    out_dir = cfg["output"]["checkpoint_dir"].replace("checkpoints", "").rstrip("/\\")
    if not out_dir:
        out_dir = "outputs"
    out_path = os.path.join(out_dir, "test_results.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained model on the test split")
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/best.pth",
        help="Path to the .pth checkpoint to evaluate",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML config used during training",
    )
    args = parser.parse_args()
    main(args.checkpoint, args.config)
