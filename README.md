# VRDL Homework 1 - ResNet Family Classification Pipeline

## Introduction

This repository contains a reproducible PyTorch pipeline for the Visual Recognition using Deep Learning HW1
100-class image classification task.

Project constraints and design goals:

- backbone family stays within the ResNet family
- parameter count stays under `100M`
- no external training data is used
- pretrained weights are allowed and supported
- final submission artifact is `prediction.csv`

Implemented capabilities:

- folder or CSV dataset loading
- stratified train/validation handling
- single-run and batch training workflows
- validation with `acc1`, `acc5`, macro recall, `NLL`, `ECE`, and analysis exports
- inference with optional TTA and probability export
- staged training templates such as short FixRes refresh and classifier rebalance
- diversity-first evaluation and greedy ensemble search utilities

Supported residual-family backbones include:

- `resnet50`
- `resnet101`
- `resnet152`
- `wide_resnet50_2`
- `resnext101_64x4d`
- `resnext101_32x8d`
- `resnet101d_bse_sd`
- `resnext101_32x8d_d_bse_sd`
- `resnetv2_101x1_bit.goog_in21k_ft_in1k`

## Environment Setup

Python `3.10` is the target version.

Create the environment manually:

```bash
conda create -n vr_hw1 python=3.10
conda activate vr_hw1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Or create it from the checked-in environment file:

```bash
conda env create -f environment.yml
conda activate vr_hw1
```

Verify the GPU environment:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Lint the Python code before committing:

```bash
ruff check src scripts
```

The repository includes a `pyproject.toml` with the active Ruff configuration. The lint pass is intended to keep
the code aligned with PEP 8 and practical Google-style Python conventions.

## Usage

### Dataset layout

Folder layout:

```text
data/
  train/
    class_0/*.jpg
    class_1/*.jpg
    ...
  val/            # optional
    class_0/*.jpg
    class_1/*.jpg
    ...
  test/*.jpg
```

CSV layout:

```text
data/
  train.csv
  test.csv
  train_images/
  test_images/
```

### Inspect the dataset

```bash
python scripts/inspect_dataset.py --config configs/config.yaml
```

### Train a standard experiment

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --train-config configs/experiments/resnet101_strong_v1.yaml
```

### Train the current strongest BiT experiment

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --train-config configs/experiments/target099/resnetv2_101x1_bit_goog_in21k_ft_in1k_strong.yaml \
  --output-dir outputs/bit_resnetv2_101x1
```

### Validate a checkpoint

```bash
python scripts/validate.py \
  --config configs/config.yaml \
  --train-config configs/experiments/target099/resnetv2_101x1_bit_goog_in21k_ft_in1k_strong.yaml \
  --ckpt outputs/<run_dir>/checkpoints/best_ema.ckpt \
  --use-ema
```

Validation exports:

- `validate_metrics.json`
- `per_class_report.csv`
- `hardest_classes.csv`
- `val_predictions.csv`
- `val_misclassified.csv`
- `confusion_matrix.npy`
- `confusion_matrix.csv`

### Generate test predictions

```bash
python scripts/infer.py \
  --config configs/config.yaml \
  --train-config configs/experiments/target099/resnetv2_101x1_bit_goog_in21k_ft_in1k_strong.yaml \
  --ckpt outputs/<run_dir>/checkpoints/best_ema.ckpt \
  --use-ema \
  --tta
```

This writes:

- `prediction.csv`
- `inference_summary.json`
- optional probability artifacts when `--save-probs` is enabled

### Run batch jobs

```bash
python scripts/train_batch.py --batch-config configs/batch/train_sequence_example.yaml
python scripts/validate_batch.py --batch-config configs/batch/validate_resnet101_v2_fast_3seeds.yaml
python scripts/infer_batch.py --batch-config configs/batch/infer_resnet101_v2_fast_3seeds.yaml
```

### Diversity and ensemble utilities

```bash
python scripts/diversity_report.py \
  --manifest configs/manifests/target099_resnext50_32x3d_diversity_probe_vs_anchor_template.yaml \
  --output-dir outputs/diversity_target099_resnext50_32x3d_probe

python scripts/ensemble_search.py \
  --manifest configs/manifests/target099_resnext50_32x3d_diversity_probe_vs_anchor_template.yaml \
  --anchor target099_fixres_anchor_ema \
  --diversity-summary outputs/diversity_target099_resnext50_32x3d_probe/diversity_summary.csv \
  --output-dir outputs/ensemble_target099_resnext50_32x3d_probe
```

## Performance Snapshot

Recent validated `target099` checkpoints on the `300`-image local validation split:

| Model | Recipe / Branch | Val Acc@1 | Val NLL | Val ECE | Notes |
|---|---|---:|---:|---:|---|
| BiT ResNetV2-101x1 | `best_ema.ckpt` | 92.33 | 0.3806 | 0.0791 | current best local single-model result |
| ResNet101 FixRes320 from `s0_r1` | selected checkpoint | 91.00 | 0.3986 | 0.1004 | previous primary anchor |
| ResNet101 CRT FixRes | selected checkpoint | 90.33 | 0.3587 | 0.0261 | strongest calibration-oriented run |
| ResNeXt50-32x3d probe | `best_raw.ckpt` | 89.00 | 0.4936 | 0.0713 | keep only as possible ensemble candidate |
| ResNet101 LDAM-DRW | selected checkpoint | 89.00 | 3.3001 | 0.8478 | implemented correctly, killed experimentally |


![Leaderboard Performance](src\img\Leaderboard.png)
