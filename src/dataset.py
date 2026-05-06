"""
Football Player Dataset
=======================
Loads COCO-format polygon segmentation annotations and feeds them to Mask R-CNN.

The COCO annotation format stores each player as:
  - A bounding box [x, y, width, height]
  - A polygon list of (x, y) coordinates tracing the player's outline
  - A category id (always 1 = "person" in this dataset)

This module converts those polygons to binary pixel masks, which is what
Mask R-CNN needs during training.
"""

import os
import random
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class FootballDataset(Dataset):
    """
    PyTorch Dataset wrapper around COCO-format football player annotations.

    Each sample returns:
        image  : FloatTensor[3, H, W]  — pixel values normalised to [0, 1]
        target : dict with keys
                   boxes    FloatTensor[N, 4]  xyxy bounding boxes
                   labels   Int64Tensor[N]     all 1s (person class)
                   masks    UInt8Tensor[N,H,W] binary segmentation masks
                   image_id Int64Tensor[1]
                   area     FloatTensor[N]     mask area in pixels
                   iscrowd  Int64Tensor[N]     all zeros (no crowd annotations)
    """

    def __init__(
        self,
        images_dir: str,
        annotation_file: str,
        transforms=None,
    ):
        self.images_dir = images_dir
        self.coco = COCO(annotation_file)
        self.transforms = transforms

        # Keep only images that actually have at least one player annotated.
        # A handful of frames may be pure empty-pitch shots.
        all_ids = list(sorted(self.coco.imgs.keys()))
        self.ids = [
            img_id
            for img_id in all_ids
            if len(self.coco.getAnnIds(imgIds=img_id)) > 0
        ]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        # --- Load image ---
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        # --- Load annotations ---
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        annotations = self.coco.loadAnns(ann_ids)

        boxes, masks, labels, areas = [], [], [], []

        for ann in annotations:
            x, y, w, h = ann["bbox"]
            # Skip degenerate boxes that can appear after tight polygon fits
            if w < 1 or h < 1:
                continue

            # COCO bbox is [x, y, w, h] — Mask R-CNN expects [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])

            # annToMask converts the polygon vertices to a dense binary array
            # of the same height × width as the image, with 1 where the player
            # is and 0 everywhere else.
            mask = self.coco.annToMask(ann)
            masks.append(mask)

            labels.append(1)   # class index 1 = person (0 is reserved for background)
            areas.append(ann["area"])

        # Handle the rare edge case of an image with only degenerate annotations
        if len(boxes) == 0:
            h_img, w_img = img_info["height"], img_info["width"]
            target = {
                "boxes":    torch.zeros((0, 4), dtype=torch.float32),
                "labels":   torch.zeros(0, dtype=torch.int64),
                "masks":    torch.zeros((0, h_img, w_img), dtype=torch.uint8),
                "image_id": torch.tensor([img_id]),
                "area":     torch.zeros(0, dtype=torch.float32),
                "iscrowd":  torch.zeros(0, dtype=torch.int64),
            }
        else:
            target = {
                "boxes":    torch.tensor(boxes, dtype=torch.float32),
                "labels":   torch.tensor(labels, dtype=torch.int64),
                "masks":    torch.tensor(np.stack(masks), dtype=torch.uint8),
                "image_id": torch.tensor([img_id]),
                "area":     torch.tensor(areas, dtype=torch.float32),
                "iscrowd":  torch.zeros(len(labels), dtype=torch.int64),
            }

        # PIL → FloatTensor [C, H, W] with values in [0, 1]
        image = F.to_tensor(image)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


# ---------------------------------------------------------------------------
# Transform pipeline
# ---------------------------------------------------------------------------

class Compose:
    """Chain multiple (image, target) transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    """
    Flip the image and all annotations horizontally with probability `prob`.

    Why horizontal flip?  Football footage is filmed sideways and the pitch
    is (roughly) symmetric left-to-right, so a flipped frame is just as valid
    a training example.  This effectively doubles the dataset size for free
    and helps the model generalise to both sides of the pitch.
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            _, _, width = image.shape   # C × H × W

            image = F.hflip(image)

            if target["boxes"].shape[0] > 0:
                boxes = target["boxes"].clone()
                # Mirror x-coordinates around the image centre
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target["boxes"] = boxes

            if target["masks"].shape[0] > 0:
                # .flip(-1) flips along the last (width) axis
                target["masks"] = target["masks"].flip(-1)

        return image, target


class ColorJitter:
    """
    Randomly vary brightness, contrast, saturation, and hue.

    Football broadcasts differ widely in lighting — afternoon sun, floodlights,
    overcast skies.  Colour jitter teaches the model to find players by their
    shape, not by a specific lighting condition.
    """

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.05,
    ):
        self._jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, image, target):
        # ColorJitter expects a PIL image; we momentarily convert, then convert back.
        image = self._jitter(F.to_pil_image(image))
        image = F.to_tensor(image)
        return image, target


def get_transforms(train: bool = True, cfg: Optional[dict] = None):
    """
    Build the transform pipeline for training or validation.

    During validation we never augment — we want to measure how the model
    performs on clean, unmodified frames.
    """
    aug = (cfg or {}).get("augmentation", {})
    flip_prob = aug.get("horizontal_flip_prob", 0.5)
    jitter_cfg = aug.get("color_jitter", {})

    if train:
        return Compose(
            [
                RandomHorizontalFlip(prob=flip_prob),
                ColorJitter(
                    brightness=jitter_cfg.get("brightness", 0.2),
                    contrast=jitter_cfg.get("contrast", 0.2),
                    saturation=jitter_cfg.get("saturation", 0.2),
                    hue=jitter_cfg.get("hue", 0.05),
                ),
            ]
        )
    else:
        return Compose([])   # identity — no augmentation at eval time


def collate_fn(batch):
    """
    Custom collate function required by Mask R-CNN.

    The default PyTorch collate_fn tries to stack tensors into a single
    batch tensor.  That fails here because each image has a different number
    of players (and therefore different-sized masks/boxes tensors).  Instead
    we return a tuple of lists — one list of images and one list of targets.
    """
    return tuple(zip(*batch))
