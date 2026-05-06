"""
Inference and Visualisation
============================
Load a trained model, run it on new images, and produce annotated output
images where each detected player gets a unique coloured mask.

Typical usage
-------------
Single image:
    python -m src.inference --checkpoint outputs/checkpoints/best.pth \\
                            --image data/raw/images/42.jpg

Batch (all test images):
    python -m src.inference --checkpoint outputs/checkpoints/best.pth \\
                            --image_dir data/raw/images \\
                            --output_dir outputs/visualizations
"""

import argparse
import os
import random

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
import yaml
from PIL import Image

from src.model import get_model


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_trained_model(
    checkpoint_path: str,
    config_path: str = "configs/config.yaml",
    device: torch.device = None,
):
    """
    Reconstruct the model architecture and load trained weights.

    Args:
        checkpoint_path : path to a .pth file saved by src/train.py
        config_path     : YAML config used during training
        device          : if None, selects CUDA when available

    Returns:
        (model, cfg, device)
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(num_classes=cfg["model"]["num_classes"], pretrained=False)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, cfg, device


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
    score_threshold: float = 0.50,
):
    """
    Run inference on a single image file.

    Returns:
        image  : numpy array [H, W, 3] RGB, uint8
        boxes  : [N, 4] float  xyxy bounding boxes
        masks  : [N, H, W] uint8  binary segmentation masks
        scores : [N] float  confidence scores
        labels : [N] int    class indices (all 1 = person)
    """
    image_pil = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image_pil).unsqueeze(0).to(device)

    output = model(image_tensor)[0]

    keep = output["scores"] >= score_threshold

    boxes  = output["boxes"][keep].cpu().numpy()
    masks  = output["masks"][keep].cpu().numpy()[:, 0]   # drop channel dim
    scores = output["scores"][keep].cpu().numpy()
    labels = output["labels"][keep].cpu().numpy()

    # Binarise the soft float masks at 0.5
    masks = (masks > 0.5).astype(np.uint8)

    return np.array(image_pil), boxes, masks, scores, labels


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_predictions(
    image: np.ndarray,
    boxes: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    save_path: str = None,
    show: bool = True,
    title: str = None,
):
    """
    Draw coloured segmentation masks and bounding boxes on the image.

    Each player receives a randomly assigned colour so that neighbouring
    players are visually distinct.  The mask is drawn semi-transparently
    so the player's kit and number remain readable underneath.

    Args:
        image     : [H, W, 3] uint8 RGB image
        boxes     : [N, 4] bounding boxes
        masks     : [N, H, W] binary masks
        scores    : [N] confidence scores
        save_path : if provided, save the figure to this path
        show      : if True, display interactively with plt.show()
        title     : optional figure title
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.imshow(image)

    # Generate one distinct colour per player
    rng = np.random.default_rng(seed=0)
    colors = [rng.uniform(0.2, 1.0, 3).tolist() for _ in range(len(masks))]

    for box, mask, score, color in zip(boxes, masks, scores, colors):
        # Semi-transparent colour fill inside the player's silhouette
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[mask == 1] = [*color, 0.45]
        ax.imshow(overlay, interpolation="nearest")

        # Bounding box outline
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        # Confidence label above the box
        ax.text(
            x1, y1 - 5, f"{score:.2f}",
            color="white", fontsize=7, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7, linewidth=0),
        )

    n = len(masks)
    default_title = f"Football Player Segmentation — {n} player{'s' if n != 1 else ''} detected"
    ax.set_title(title or default_title, fontsize=13, pad=10)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def batch_inference(
    model: torch.nn.Module,
    image_dir: str,
    output_dir: str,
    device: torch.device,
    score_threshold: float = 0.50,
    limit: int = None,
):
    """
    Run inference on every .jpg in image_dir and save annotated PNG files.

    Args:
        limit : if set, process only this many images (useful for quick checks)
    """
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted(
        f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")
    )
    if limit:
        image_files = image_files[:limit]

    print(f"Running batch inference on {len(image_files)} images…")
    for i, fname in enumerate(image_files, 1):
        img_path  = os.path.join(image_dir, fname)
        save_path = os.path.join(output_dir, fname.replace(".jpg", "_pred.png"))

        image, boxes, masks, scores, labels = predict(
            model, img_path, device, score_threshold
        )
        visualize_predictions(image, boxes, masks, scores, save_path=save_path, show=False)

        if i % 20 == 0:
            print(f"  {i}/{len(image_files)} done")

    print(f"\nSaved {len(image_files)} predictions to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football player segmentation inference")
    parser.add_argument("--checkpoint",  default="outputs/checkpoints/best.pth")
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--image",       help="Path to a single image")
    parser.add_argument("--image_dir",   help="Directory of images for batch inference")
    parser.add_argument("--output_dir",  default="outputs/visualizations")
    parser.add_argument("--threshold",   type=float, default=0.50)
    parser.add_argument("--limit",       type=int,   default=None,
                        help="Max images for batch mode (omit for all)")
    args = parser.parse_args()

    model, cfg, device = load_trained_model(args.checkpoint, args.config)

    if args.image:
        image, boxes, masks, scores, labels = predict(
            model, args.image, device, args.threshold
        )
        visualize_predictions(image, boxes, masks, scores, show=True)
    elif args.image_dir:
        batch_inference(
            model, args.image_dir, args.output_dir,
            device, args.threshold, args.limit,
        )
    else:
        print("Provide --image or --image_dir.  Run with --help for usage.")
