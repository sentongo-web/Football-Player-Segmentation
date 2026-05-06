"""
Shared utility helpers: seeding, checkpointing, and metric tracking.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Fix every source of randomness so experiments are reproducible.

    Without this, two runs with the same config can produce different
    train/val splits, weight initialisations, and augmentation orders.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Makes GPU ops deterministic at a small speed cost
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Tracks a running mean of any scalar (loss, metric, time).

    Usage:
        meter = AverageMeter()
        for batch in loader:
            loss = compute_loss(batch)
            meter.update(loss.item(), n=len(batch))
        print(meter.avg)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_ap: float,
    path: str,
) -> None:
    """
    Persist the full training state to disk.

    Saving the optimizer state lets you resume training exactly where you
    left off, including the adaptive learning-rate state (e.g. SGD momentum
    buffers).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_ap": best_ap,
        },
        path,
    )


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    device: torch.device,
) -> tuple[int, float]:
    """
    Restore model and optimizer state from a saved checkpoint.

    Returns:
        (next_epoch, best_ap) so the training loop can resume seamlessly.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"] + 1, checkpoint["best_ap"]
