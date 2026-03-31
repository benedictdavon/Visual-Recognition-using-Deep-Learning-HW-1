# Evaluation and Validation Spec

## Overview

This spec defines the current validation contract, including the distinction between the official validation set and any internal development split.
It is derived from [scripts/validate.py](../../../scripts/validate.py), [src/engine/evaluator.py](../../../src/engine/evaluator.py), [src/engine/trainer.py](../../../src/engine/trainer.py), and [src/data/dataset.py](../../../src/data/dataset.py).

This spec must be read together with:
- [../homework-requirements/spec.md](../homework-requirements/spec.md)
- [../data-and-split/spec.md](../data-and-split/spec.md)
- [../experimentation-and-reproducibility/spec.md](../experimentation-and-reproducibility/spec.md)

## Requirements

### Observed current behavior

- Validation occurs in two places:
  - during training, once per epoch
  - through `scripts/validate.py` on a saved checkpoint
- The repo currently has an explicit validation folder under `data/val`. Under the default config, this acts as the official validation set.
- If an explicit validation dataframe is absent, the repo falls back to an internal development split or K-fold split built from training data.
- The evaluator currently computes:
  - `loss`
  - `nll`
  - `acc1`
  - `acc5`
  - `ece`
  - confusion matrix
  - per-class accuracy
  - `macro_recall`
- During training, checkpoint selection is controlled by `train.model_selection`:
  - metric: one of `val_acc`, `val_macro_recall`, `val_nll`, `val_ece`, `val_loss`
  - source: `raw`, `ema`, or `auto`
  - mode: `max`, `min`, or `auto`
- `best.ckpt` stores the checkpoint state for the selected epoch, not a promise about which weight branch a later consumer will load.
- `best_raw.ckpt` is ranked by raw `val_acc`.
- `best_ema.ckpt` is ranked by EMA `val_acc`.
- `best.ckpt` can contain both `state_dict` and `ema_state_dict`.
- `scripts/validate.py` and `scripts/infer.py` load the standard `state_dict` by default unless:
  - `--use-ema` is passed, or
  - the checkpoint filename is `best_ema.ckpt` and EMA weights are present, in which case they auto-load EMA weights
- `scripts/validate.py` writes:
  - `validate_metrics.json`
  - `confusion_matrix.npy`
  - `confusion_matrix.csv`
  - and, unless `--no-analysis` is set:
    - `per_class_report.csv`
    - `hardest_classes.csv`
    - `val_predictions.csv`
    - `val_misclassified.csv`
    - `val_probs_with_ids.npz`

### Required preserved behavior

- The "official validation set" in this repo means the explicit validation dataset loaded from `dataset.val_dir` when that setting resolves to a real validation folder.
- The "internal development split" means the split created from training data when no explicit validation set is used.
- The official validation set must take precedence unless it is explicitly disabled in config.
- Test data must remain unlabeled and excluded from model selection.
- The selected checkpoint epoch and the selected weight branch must remain distinguishable:
  - `best.ckpt` identifies the selected epoch
  - raw vs EMA loading is a separate consumer choice unless the checkpoint filename itself is `best_ema.ckpt`
- Validation metrics used for adoption decisions must remain visible in logs and machine-readable artifacts.
- Leaderboard-facing decisions must continue to require local validation evidence. Public leaderboard movement alone is not enough to bless a recipe in this spec set.

### Must not break

- Validation must not silently merge the official validation set into training.
- `summary.json` must continue to expose:
  - the selected metric
  - the selected source
  - the selected mode
  - raw and EMA alternatives
- `validate.py` must continue to support both raw and EMA evaluation for checkpoints that contain EMA state.
- `val_probs_with_ids.npz` must continue to align `sample_ids`, `targets`, `preds`, and `probs`.
- Future automation must not assume that `best.ckpt` implies the selected weight branch will be loaded automatically.

## Scenarios

### Scenario: Official validation set takes precedence

- Given `dataset.val_dir` resolves to an existing validation folder
- When `prepare_dataframes` is called
- Then the official validation set must be loaded from that folder
- And no internal development split may replace it

### Scenario: Explicitly disabling the official validation set changes behavior

- Given `dataset.val_dir` is explicitly set to `null` or an empty string
- When `prepare_dataframes` is called
- Then the repo must not auto-pick `data/val`
- And validation must come from an internal development split or K-fold path instead

### Scenario: `best.ckpt` selected on EMA metric but validated without `--use-ema`

- Given `best.ckpt` was selected using `source: ema`
- When `scripts/validate.py --ckpt <best.ckpt>` is run without `--use-ema`
- Then the script evaluates the raw `state_dict` from the selected epoch, not the EMA branch
- And this is a high-risk failure mode for automation if the branch choice is not made explicit

### Scenario: `best_ema.ckpt` auto-loads EMA weights

- Given a checkpoint path ending in `best_ema.ckpt`
- And the checkpoint contains `ema_state_dict`
- When `scripts/validate.py` runs without `--use-ema`
- Then the script auto-loads EMA weights and logs a warning

### Scenario: `--no-analysis` produces a reduced validation artifact set

- Given `scripts/validate.py --no-analysis`
- When validation completes
- Then `validate_metrics.json` and confusion-matrix outputs must still be written
- And the detailed per-sample and per-class analysis files may be omitted

### Scenario: Test data may not be used as a labeled selector

- Given a proposal to use test predictions as if they were labeled validation results
- When the proposal is evaluated against this spec
- Then the proposal is invalid, regardless of whether it improves leaderboard score

### Scenario: Recipe adoption requires local evidence

- Given a candidate recipe or checkpoint for homework submission
- When it is adopted as better than the current baseline
- Then the evidence should include at least:
  - the training `summary.json`
  - the selected-branch validation result
  - the validation artifact directory when detailed analysis is relevant

## Known ambiguities or gaps

- The official validation set is tiny and balanced, while the training set is heavily imbalanced. This makes local ranking noisy even though the validation pipeline itself is working as intended.
- `val_loss` is the configured criterion loss, while `val_nll` is always plain cross-entropy on logits. They are intentionally different and should not be treated as synonyms.
- The repo contains Stage 2 documentation that mentions train+val refit, but that workflow is not implemented in the core training path.
- Internal development splits depend on the configured seed when the official validation set is absent or disabled.

## Non-goals

- This spec does not require cross-validation, repeated holdout, or calibration fitting.
- This spec does not define leaderboard strategy beyond protecting local evidence and leakage boundaries.
- This spec does not add new validation metrics beyond the current evaluator outputs.
