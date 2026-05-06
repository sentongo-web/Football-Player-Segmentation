# Football Player Segmentation

Instance segmentation of football players in broadcast footage using **Mask R-CNN** (ResNet-50-FPN backbone) fine-tuned on a COCO-format dataset of ~520 labelled frames.

---

## Architecture

```text
Input image
    │
ResNet-50 backbone  ──→  multi-scale feature maps
    │
Feature Pyramid Network (FPN)  ──→  combines 4 scale levels
    │
Region Proposal Network (RPN)  ──→  ~2000 candidate boxes
    │
ROI Align  ──→  per-proposal feature crops
    │
┌───┴──────────────────────┐
│                          │
Box Head                 Mask Head
(class + bbox regression)  (28×28 binary mask per instance)
```

Transfer learning strategy: the COCO-pretrained backbone and FPN are kept and fine-tuned; only the box predictor (91 → 2 classes) and mask predictor heads are replaced from scratch.

---

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended (≥ 8 GB VRAM for batch size 2)
- See [requirements.txt](requirements.txt) for the full dependency list

---

## Installation

```bash
git clone https://github.com/<your-username>/Football-Player-Segmentation.git
cd Football-Player-Segmentation
pip install -r requirements.txt
```

On Windows with a CUDA GPU, install the matching PyTorch build first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Dataset

The project expects raw images and a single COCO-format annotation file:

```text
data/
└── raw/
    ├── images/                       # 0.jpg … 527.jpg  (520 frames)
    └── annotations/
        └── instances_default.json   # polygon masks, COCO format
```

Each image is annotated with polygon segmentations for every visible player. The single category is `person` (id = 1); background is id = 0.

---

## Quick Start

### 1 — Prepare data splits

Split the master annotation file into train / val / test subsets (80 / 10 / 10):

```bash
python scripts/prepare_data.py
```

Output:

```text
data/splits/
├── train.json   # ~410 images
├── val.json     # ~51  images
└── test.json    # ~51  images
```

### 2 — Train

```bash
python -m src.train
# or with a custom config:
python -m src.train --config configs/config.yaml
```

Progress is printed every 20 batches. COCO metrics are evaluated on the validation set every 2 epochs. The best checkpoint (highest segmentation AP) is saved to `outputs/checkpoints/best.pth`.

Monitor training with TensorBoard:

```bash
tensorboard --logdir outputs/logs
```

### 3 — Evaluate on the test set

After training, run the held-out test evaluation:

```bash
python scripts/test.py
# or with a specific checkpoint:
python scripts/test.py --checkpoint outputs/checkpoints/best.pth
```

Results are printed to the console and saved to `outputs/test_results.json`.

### 4 — Run inference

Single image:

```bash
python -m src.inference \
    --checkpoint outputs/checkpoints/best.pth \
    --image data/raw/images/42.jpg
```

Batch (all images in a directory):

```bash
python -m src.inference \
    --checkpoint outputs/checkpoints/best.pth \
    --image_dir data/raw/images \
    --output_dir outputs/visualizations \
    --limit 50
```

Annotated PNG files are written to `--output_dir`. Each player is drawn with a unique semi-transparent colour overlay and a confidence score label.

---

## Project Structure

```text
Football-Player-Segmentation/
├── configs/
│   └── config.yaml            # all hyperparameters
├── data/
│   ├── raw/
│   │   ├── images/            # source frames
│   │   └── annotations/
│   │       └── instances_default.json
│   └── splits/                # created by prepare_data.py
│       ├── train.json
│       ├── val.json
│       └── test.json
├── outputs/                   # created by training
│   ├── checkpoints/
│   │   ├── best.pth
│   │   └── latest.pth
│   ├── logs/                  # TensorBoard event files
│   └── visualizations/        # annotated inference images
├── scripts/
│   ├── prepare_data.py        # splits annotations into train/val/test
│   └── test.py                # final test-set evaluation
├── src/
│   ├── __init__.py
│   ├── dataset.py             # FootballDataset + transforms
│   ├── evaluate.py            # COCO metric computation
│   ├── inference.py           # predict + visualize
│   ├── model.py               # Mask R-CNN construction
│   ├── train.py               # training loop
│   └── utils.py               # checkpointing, seeding, metrics
└── requirements.txt
```

---

## Configuration

All hyperparameters live in [configs/config.yaml](configs/config.yaml). Key settings:

| Section | Key | Default | Notes |
| --- | --- | --- | --- |
| `data` | `train_ratio` | `0.80` | Fraction of images for training |
| `data` | `seed` | `42` | Reproducibility seed |
| `model` | `num_classes` | `2` | Background + person |
| `model` | `pretrained` | `true` | COCO pretrained backbone |
| `training` | `epochs` | `20` | Total training epochs |
| `training` | `batch_size` | `2` | Reduce to 1 if GPU OOM |
| `training` | `learning_rate` | `0.005` | Initial SGD LR |
| `training` | `lr_step_size` | `10` | Drop LR every N epochs |
| `training` | `lr_gamma` | `0.1` | LR decay factor |
| `augmentation` | `horizontal_flip_prob` | `0.5` | Left-right symmetry of pitch |
| `inference` | `score_threshold` | `0.50` | Minimum detection confidence |

---

## Expected Results

On this ~520-image dataset with the default config, a fully trained model typically achieves:

| Metric | Expected Range |
| --- | --- |
| Segmentation AP | 0.55 – 0.70 |
| Segmentation AP50 | 0.80 – 0.90 |
| Segmentation AP75 | 0.55 – 0.75 |
| Detection AP | 0.60 – 0.75 |

Results vary with available compute, GPU, and batch size.

---

## Resuming Training

If training is interrupted, it resumes automatically from `outputs/checkpoints/latest.pth`:

```bash
python -m src.train   # picks up where it left off
```

To start fresh, delete `outputs/checkpoints/latest.pth` first.

---

## Acknowledgements

- [torchvision Mask R-CNN](https://pytorch.org/vision/stable/models/mask_rcnn.html) — pretrained model and reference implementation
- [pycocotools](https://github.com/cocodataset/cocoapi) — official COCO evaluation toolkit
- Dataset annotated in COCO format using [CVAT](https://github.com/cvat-ai/cvat)
