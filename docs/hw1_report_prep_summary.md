# HW1 Consolidated Report Prep Summary

## 1. Project Snapshot

- Course: Visual Recognition using Deep Learning (HW1)
- Task: 100-class image classification
- Data layout: `data/train/<class_id>/*.jpg`, `data/val/<class_id>/*.jpg`, `data/test/*.jpg`
- Sizes: train 20,724, val 300, test 2,344
- Hard constraints followed:
  - ResNet-only backbone family in official pipeline
  - Parameter limit under 100M
  - No external data
  - Submission file: `prediction.csv`

## 2. What We Built and Fixed

- Complete training/validation/inference pipeline with checkpointing and submission generation.
- Fixed label mapping bug for numeric class folders (`0..99`) so submission labels align with competition IDs.
- Standardized submission schema to `image_name,pred_label`.
- Added robust evaluation artifacts: per-class report, confusion matrix, hardest classes, misclassified samples.
- Added stronger inference-selection tooling:
  - deterministic raw vs EMA checkpoint selection
  - probability-saving inference artifacts for soft-vote ensembling
  - hard/soft/weighted-soft ensemble support

## 3. Experiment Progress (Evidence Summary)

| Run | Params | Best val acc@1 | Public LB (user reported) | Key takeaway |
|---|---:|---:|---:|---|
| `20260311_235832_baseline_resnet50` | 23.71M | 86.33 | - | Stable baseline |
| `20260312_002956_resnet101_strong_v1` | 42.71M | 86.33 | 0.91 | Strong recipe alone not enough |
| `20260312_134125_resnet101_strong_v2_fast` | 42.71M | 89.67 | 0.94 | Best base training recipe |
| `20260312_153549_resnet152_strong_v1` | 58.35M | 89.67 | 0.93 | Bigger depth did not beat best ResNet101 path |
| `20260313_225258_resnet101_fixres_finetune_320` | 42.71M | 91.00 | 0.95 | FixRes-style short finetune improved leaderboard |
| `20260315_232045_resnext101_64x4d_v1` | 81.61M | 85.67 | - | Not competitive under current recipe |
| `20260316_190430_resnet101_fixres_finetune_320_seed2026` | 42.71M | 92.00 | pending | Best local val among seeded FixRes320 runs |

## 4. Core Findings

1. Correct label restoration in inference/submission was critical; correctness fixes produced the largest real gain.
2. Current performance bottleneck is recipe/selection/inference, not raw parameter count.
3. ResNet101 remains stronger than ResNet152 in this project context.
4. FixRes-style short high-resolution finetuning is the most effective improvement so far.
5. Tiny validation set (300 images, balanced 3/class) causes selection noise; multi-seed and ensembling are practical stabilizers.

## 5. Artifact Map for Report Evidence

- Training curves and epoch metrics:
  - `outputs/*/history.csv`
  - `outputs/*/train.log`
- Run-level summary metrics and checkpoint identity:
  - `outputs/*/summary.json`
- Validation analysis artifacts:
  - `outputs/validate_batch/*/validate_metrics.json`
  - `outputs/validate_batch/*/confusion_matrix.csv`
  - `outputs/validate_batch/*/per_class_report.csv`
  - `outputs/validate_batch/*/hardest_classes.csv`
  - `outputs/validate_batch/*/val_misclassified.csv`
- Submission outputs:
  - `outputs/**/prediction.csv`
- Ensemble evidence:
  - `outputs/ensemble*/prediction.csv`

## 6. Grading Policy - Report (15%)

- Format: PDF, written in English. (-5pts if not followed)
- Sections that you should include
  - Introduction to the task and core idea of your method
  - Method: Describe your data preprocessing procedure, model architecture, hyperparameter settings, and any modifications or adjustments you made.
  - Results: your findings / model performance (e.g., training curve, confusion matrix, etc.)
  - References: Your method references (paper / Github sources, must include if you use any). You may refer to the ECCV 2026 template for the citation and reference formatting guidelines.
- We encourage you to stand on the shoulders of giants - only clone and run it is not enough.
- Additional experiments to explore better performance
  - Simply tuning the hyper-parameters doesn’t count (e.g., batch-size, LR, different optimizers).
  - Hint: Try to add/remove some layers, use different loss functions, etc.
- You should
  1. include your hypothesis (why you do this),
  2. explain how this may (or may not) work, and
  3. report experiment results and their implications.

## 7. Report Drafting Outline (Use This for Final PDF)

### Introduction

- Problem setup and constraints.
- Why the selected path is ResNet101-centered despite trying deeper/wider variants.
- Core idea: correctness-first + stable recipe + targeted FixRes finetune + robust inference selection.

### Method

- Data preprocessing and augmentation (baseline vs strong recipe).
- Model family and legal constraints.
- Training settings (optimizer/scheduler/loss/EMA).
- Key modifications:
  - label mapping/submission correctness fix
  - FixRes finetune stage
  - ensemble selection at inference.

### Results

- Table of run progression (Section 3 above).
- Curves and confusion matrix from `validate_batch` artifacts.
- Error analysis highlights from hardest/misclassified outputs.
- Explain why some larger models underperformed in this dataset setup.

### References

- List all paper/code references used (ResNet, FixRes, augmentation/loss ideas, etc.)
- Follow ECCV 2026 citation formatting guidance.

### Additional Experiments (Required Analysis Framing)

1. **Label mapping correctness fix**
   - Hypothesis: incorrect class ID restoration causes severe leaderboard collapse.
   - Why it may work: restores true category IDs expected by competition.
   - Result/implication: immediate large score recovery; submission correctness is foundational.

2. **ResNet152 vs ResNet101**
   - Hypothesis: deeper model should outperform.
   - Why it may not work: optimization/regularization mismatch and limited dataset size.
   - Result/implication: ResNet152 did not beat best ResNet101 path; depth alone is insufficient.

3. **FixRes320 finetune on best ResNet101**
   - Hypothesis: reducing train-test resolution discrepancy improves generalization.
   - Why it may work: better alignment between learned features and inference resolution.
   - Result/implication: improved local val and reached 0.95 public LB; high-ROI direction.

## 8. Current Status and Next Action

- Current strongest public score in this project history: **0.95**.
- Current recommended submission workflow:
  1. evaluate raw/EMA branches deterministically,
  2. infer with TTA + saved probabilities,
  3. use soft/weighted-soft ensembles across top checkpoints/seeds,
  4. submit best `prediction.csv` generated from ensemble outputs.
