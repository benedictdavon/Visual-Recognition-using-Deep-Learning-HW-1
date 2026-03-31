# Visual Recognition HW1 Progress Summary

## 1) Goal and Constraints

- Task: 100-class image classification.
- Dataset size: train 20,724, val 300, test 2,344 (folder-based split available locally).
- Mandatory constraints followed:
  - ResNet-only backbone
  - Parameter count under 100M
  - No external data
  - `prediction.csv` submission output

## 2) Baseline Setup (First Working End-to-End)

- Model: ResNet50 (torchvision pretrained)
- Input size: 224/256 baseline pipeline
- Training: AdamW + cosine schedule
- Features: no EMA, no Mixup/CutMix, basic augmentation
- Output: train/val loop, checkpoints, inference, `prediction.csv`

Baseline run:
- Run dir: `outputs/20260311_235832_baseline_resnet50`
- Params: **23.71M**
- Best val acc@1: **86.33%** (epoch 20)

## 3) What We Changed From Baseline

### Data/Submission correctness fixes

- Diagnosed and fixed class-label mapping issue for numeric class folders (`0..99`):
  - Internal class index vs real label ID mismatch could produce very low leaderboard score.
- Submission output standardized to:
  - `image_name,pred_label`
- Inference now uses label-name restoration safely (`use_label_name: true`) and numeric-label handling.

### Training/inference improvements added

- Added support/configuration for:
  - ResNet101 / ResNet152
  - Strong augmentation recipe
  - Mixup/CutMix
  - Label smoothing
  - EMA
  - TTA (flip averaging)
- Improved runtime behavior:
  - Correct AMP API usage
  - Better CUDA logging
  - CUDA-aware `pin_memory`

### Evaluation/analysis improvements

- Expanded validation diagnostics:
  - top-1/top-5
  - per-class report
  - hardest classes
  - confusion matrix export
  - misclassification export
- Ran dataset analysis:
  - strong class imbalance (max/min ~= 22.5)
  - val set is tiny and balanced (3 per class)
  - small number of cross-split duplicates detected

## 4) Experiment Results So Far

| Run | Params | Best val acc@1 | Best EMA val acc@1 | Public LB (reported) | Notes |
|---|---:|---:|---:|---:|---|
| `baseline_resnet50` | 23.71M | 86.33% | - | - | First stable pipeline |
| `resnet101_strong_v1` | 42.71M | 86.33% | 86.33% | **0.91** | Strong recipe + EMA |
| `resnet101_strong_v2_fast` | 42.71M | 89.67% | 89.67% | **0.94** | Best public score so far |
| `resnet152_strong_v1` | 58.35M | 87.67% | 89.67% | **0.93** | Bigger model, not better than 101_v2_fast |

Notes:
- Public LB values are your reported competition scores.
- ResNet152 run summary:
  - Run dir: `outputs/20260312_153549_resnet152_strong_v1`
  - Best acc in summary: 89.67% (EMA-selected checkpoint)
  - Early stop at epoch 31/45

## 5) Key Findings

- The biggest practical gain came from fixing submission label mapping correctness.
- ResNet101 strong v2 fast currently gives the best leaderboard score (0.94).
- Increasing depth to ResNet152 did not automatically improve public score.
- EMA is consistently helpful for this setup.
- Validation set is very small, so leaderboard can move differently from val accuracy.

## 6) Current Best Practical Recipe

- Train with `resnet101_strong_v2_fast`.
- Infer with best checkpoint using EMA + TTA.
- Keep `use_label_name: true` and submission schema `image_name,pred_label`.

## 7) Status

- End-to-end project is complete and reproducible.
- Baseline and stronger legal variants are both working.
- Current best reported public leaderboard score: **0.94**.
