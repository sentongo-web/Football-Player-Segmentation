"""
COCO Evaluation Module
======================
Computes industry-standard detection and segmentation metrics using
the official COCO evaluation toolkit.

Metrics explained
-----------------
AP (Average Precision)
    The area under the precision-recall curve, averaged over IoU thresholds
    from 0.50 to 0.95 in steps of 0.05.  This is the headline number used to
    compare models in the research literature.

AP50
    AP computed at a single IoU threshold of 0.50.  A prediction counts as
    correct if it overlaps the ground-truth by at least 50%.  This is lenient
    and useful for checking whether the model finds players at all.

AP75
    AP at IoU 0.75.  Much stricter — the predicted boundary must closely hug
    the player.  Gaps here mean the segmentation masks are imprecise.

For each metric, pycocotools reports two flavours:
  - bbox  : evaluated on bounding boxes only (ignores masks)
  - segm  : evaluated on segmentation masks (the primary objective here)

A typical well-trained model on a clean sports dataset achieves:
  segm AP ≈ 0.55–0.70  (depends on player density and occlusion)
  segm AP50 ≈ 0.80–0.90

IoU (Intersection over Union)
    The ratio of the overlap area to the union area between predicted and
    ground-truth regions.  IoU = 1.0 means a perfect prediction.
"""

import numpy as np
import torch
from pycocotools import mask as coco_mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    score_threshold: float = 0.05,
) -> dict:
    """
    Run the full COCO evaluation pipeline on a given data split.

    We use a low score_threshold (0.05) here because pycocotools re-ranks
    predictions by score internally — keeping more predictions at eval time
    gives a more accurate recall curve without inflating precision.

    Args:
        model            : trained Mask R-CNN
        data_loader      : DataLoader for the val or test split
        device           : torch device
        score_threshold  : discard predictions below this confidence at eval

    Returns:
        dict with keys: segm_AP, segm_AP50, segm_AP75, bbox_AP, bbox_AP50, bbox_AP75
    """
    model.eval()

    # We accumulate predictions in COCO result format and pass them to
    # COCOeval for the official metrics calculation.
    coco_gt: COCO = data_loader.dataset.coco
    dt_bbox = []   # bounding-box predictions
    dt_segm = []   # segmentation mask predictions

    for images, targets in data_loader:
        images  = [img.to(device) for img in images]
        outputs = model(images)   # returns predictions, not losses, in eval mode

        for output, target in zip(outputs, targets):
            image_id = int(target["image_id"].item())

            boxes  = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            masks  = output["masks"].cpu().numpy()   # shape: [N, 1, H, W]

            for box, score, label, mask in zip(boxes, scores, labels, masks):
                if score < score_threshold:
                    continue

                # ---- bounding box ----
                x1, y1, x2, y2 = box.tolist()
                dt_bbox.append({
                    "image_id":   image_id,
                    "category_id": int(label),
                    "bbox":       [x1, y1, x2 - x1, y2 - y1],  # COCO: [x, y, w, h]
                    "score":      float(score),
                })

                # ---- segmentation mask ----
                # The model outputs a soft float mask; we binarise at 0.5.
                binary_mask = (mask[0] > 0.5).astype(np.uint8)
                # RLE (Run-Length Encoding) is the compact COCO mask format.
                rle = coco_mask_utils.encode(np.asfortranarray(binary_mask))
                rle["counts"] = rle["counts"].decode("utf-8")
                dt_segm.append({
                    "image_id":    image_id,
                    "category_id": int(label),
                    "segmentation": rle,
                    "score":        float(score),
                })

    metrics = {}

    for iou_type, dt_list in [("bbox", dt_bbox), ("segm", dt_segm)]:
        if not dt_list:
            metrics[f"{iou_type}_AP"]   = 0.0
            metrics[f"{iou_type}_AP50"] = 0.0
            metrics[f"{iou_type}_AP75"] = 0.0
            continue

        coco_dt   = coco_gt.loadRes(dt_list)
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # stats indices:  [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]
        metrics[f"{iou_type}_AP"]   = float(coco_eval.stats[0])
        metrics[f"{iou_type}_AP50"] = float(coco_eval.stats[1])
        metrics[f"{iou_type}_AP75"] = float(coco_eval.stats[2])

    return metrics
