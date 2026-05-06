"""
Mask R-CNN Model
================
Configures a Mask R-CNN with a ResNet-50-FPN backbone, pretrained on the
COCO dataset, and fine-tuned to detect and segment football players.

Why Mask R-CNN?
---------------
Mask R-CNN is the gold-standard two-stage instance segmentation architecture.
For a football analytics use case it offers:

  1. High accuracy  — separates overlapping players that single-stage models
                      sometimes merge into one detection.
  2. Per-player masks — lets you compute exact player positions, areas, and
                        movement heatmaps rather than just bounding boxes.
  3. Well-understood training — the torchvision implementation is battle-tested
                                and straightforward to fine-tune.
  4. Rich pretrained features — the ResNet-50-FPN backbone already knows how
                                 to detect people from COCO training; we only
                                 need to refine the final prediction heads.

Architecture Summary
--------------------
  Input image
      │
  ResNet-50 backbone  ──→  extracts feature maps at 4 scales
      │
  Feature Pyramid Network (FPN)  ──→  combines multi-scale features
      │
  Region Proposal Network (RPN)  ──→  proposes ~2000 candidate boxes
      │
  ROI Align  ──→  crops and resizes features for each proposal
      │
  ┌───┴──────────────────────┐
  │                          │
  Box Head                 Mask Head
  (classifies + refines     (outputs a 28×28 binary mask
   bounding box)             per detected instance)
"""

import torch
import torch.nn as nn
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Build and return the Mask R-CNN model ready for training or inference.

    Transfer Learning Strategy
    --------------------------
    The COCO-pretrained weights give us a backbone that already understands
    how humans look in images.  We replace only the two task-specific heads:

      - Box predictor  : was 91 classes (COCO) → now 2 (background / person)
      - Mask predictor : same class-count change

    The backbone (ResNet-50 + FPN) keeps its COCO weights and will continue
    to update through gradient descent, but it needs far fewer epochs to
    converge because the feature representations are already meaningful.

    Args:
        num_classes : total class count including background (2 for this task)
        pretrained  : if True, load COCO pretrained backbone weights

    Returns:
        model : fully configured Mask R-CNN ready for `.to(device)`
    """
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn(weights=weights)

    # --- Replace box prediction head ---
    # FastRCNNPredictor is a two-FC-layer head: it takes the per-ROI feature
    # vector and outputs class logits + bbox deltas.
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # --- Replace mask prediction head ---
    # MaskRCNNPredictor is a small FCN that upsamples the ROI feature map and
    # outputs a soft binary mask for each class.
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_channels=in_features_mask,
        dim_reduced=256,
        num_classes=num_classes,
    )

    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """
    Return (total_params, trainable_params) for a quick sanity check.

    All parameters are trainable by default; if you later decide to freeze
    the backbone for the first few epochs, trainable_params will reflect that.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
