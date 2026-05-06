"""
Data Preparation Script
=======================
Splits the master COCO annotation file into train / val / test subsets and
saves them as three separate JSON files under data/splits/.

Run this ONCE before training:
    python scripts/prepare_data.py

Why split by image, not by annotation?
---------------------------------------
Every annotation (player polygon) belongs to one image.  If we split randomly
at the annotation level, the same image could appear in both the training set
and the test set — the model would then be tested on a scene it literally saw
during training, which artificially inflates accuracy.  Splitting by image_id
guarantees that no frame appears in more than one split.

Output
------
data/splits/train.json   — 80% of images and their annotations
data/splits/val.json     — 10% (used to select the best checkpoint)
data/splits/test.json    — 10% (held out; evaluated only at the very end)
"""

import argparse
import json
import os
import random

import yaml


def split_coco_annotations(
    annotation_path: str,
    output_dir: str,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> None:
    print(f"Loading annotations from: {annotation_path}")
    with open(annotation_path) as f:
        coco = json.load(f)

    # Collect image IDs and shuffle deterministically
    image_ids = [img["id"] for img in coco["images"]]
    random.seed(seed)
    random.shuffle(image_ids)

    n       = len(image_ids)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    splits = {
        "train": set(image_ids[:n_train]),
        "val":   set(image_ids[n_train : n_train + n_val]),
        "test":  set(image_ids[n_train + n_val :]),
    }

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'Split':<8} {'Images':>7} {'Annotations':>13}  File")
    print("-" * 55)

    for split_name, split_ids in splits.items():
        split_images = [img for img in coco["images"] if img["id"] in split_ids]
        split_anns   = [ann for ann in coco["annotations"] if ann["image_id"] in split_ids]

        split_coco = {
            "licenses":    coco.get("licenses", []),
            "info":        coco.get("info", {}),
            "categories":  coco["categories"],
            "images":      split_images,
            "annotations": split_anns,
        }

        out_path = os.path.join(output_dir, f"{split_name}.json")
        with open(out_path, "w") as f:
            json.dump(split_coco, f)

        print(
            f"{split_name:<8} {len(split_images):>7} {len(split_anns):>13}  {out_path}"
        )

    # Quick sanity checks
    all_split_ids = splits["train"] | splits["val"] | splits["test"]
    assert len(all_split_ids) == n, "Some images were lost in the split!"
    assert len(splits["train"] & splits["val"]) == 0, "Train/val overlap!"
    assert len(splits["train"] & splits["test"]) == 0, "Train/test overlap!"
    assert len(splits["val"]   & splits["test"]) == 0, "Val/test overlap!"

    print("\nAll sanity checks passed.")
    print(f"\nData splits saved to: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split COCO annotations into train/val/test")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    split_coco_annotations(
        annotation_path=cfg["data"]["annotation_file"],
        output_dir=cfg["data"]["splits_dir"],
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        seed=cfg["data"]["seed"],
    )
