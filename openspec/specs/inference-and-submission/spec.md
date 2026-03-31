# Inference and Submission Spec

## Overview

This spec defines the current inference and submission contract, including checkpoint-loading semantics, output schemas, and the assumptions made by downstream ensemble utilities.
It is derived from [scripts/infer.py](../../../scripts/infer.py), [src/engine/inference.py](../../../src/engine/inference.py), [src/submission/make_submission.py](../../../src/submission/make_submission.py), [scripts/ensemble.py](../../../scripts/ensemble.py), and the validation/checkpoint behavior described by [scripts/validate.py](../../../scripts/validate.py).

This spec must be read together with:
- [../homework-requirements/spec.md](../homework-requirements/spec.md)
- [../evaluation-and-validation/spec.md](../evaluation-and-validation/spec.md)
- [../ensemble-and-search/spec.md](../ensemble-and-search/spec.md)

## Requirements

### Observed current behavior

- `scripts/infer.py` is the primary single-model inference entrypoint.
- It rebuilds the dataset and model from config, loads a checkpoint, runs eval transforms, predicts probabilities, and writes `prediction.csv`.
- Checkpoint loading rules are:
  - default: load `state_dict`
  - `--use-ema`: load `ema_state_dict`
  - if the checkpoint filename is `best_ema.ckpt` and EMA state exists, EMA weights auto-load even without `--use-ema`
- `best.ckpt` does not auto-select the EMA branch. It behaves like any other checkpoint path unless `--use-ema` is passed.
- Test-time augmentation currently means horizontal-flip averaging only.
- Prediction aggregation averages logits across TTA views and applies softmax afterward.
- Output IDs come from `dataset.test_id_from`, which must be either `sample_id` or `filename`.
- The default submission schema from `configs/config.yaml` is:
  - id column: `image_name`
  - target column: `pred_label`
- `build_prediction_dataframe` maps predicted class indices back through `idx_to_label` when `use_label_name` is enabled.
- `scripts/infer.py` can write:
  - `prediction.csv`
  - `inference_summary.json`
  - optional `test_probs.npy`
  - optional `test_logits.npy`
  - optional `test_probs_with_ids.npz`
  - optional `idx_to_label.json`
- Current probability artifact schemas are:
  - validation analysis NPZ:
    - `sample_ids`
    - `targets`
    - `preds`
    - `probs`
  - inference NPZ:
    - `sample_ids`
    - `probs`
    - optional `logits`

### Required preserved behavior

- The primary submission artifact must continue to be named exactly `prediction.csv`.
- The default submission schema must remain compatible with the homework format unless an explicit, justified schema change is required by the homework platform.
- Inference must continue to use the same evaluation transform branch as validation.
- Probability artifacts must remain aligned to `sample_ids` and row order.
- Consumers must distinguish between:
  - selected checkpoint epoch
  - selected weight branch
- If a selected epoch was chosen using EMA metrics, automation must still explicitly request EMA loading unless the checkpoint path itself is `best_ema.ckpt`.
- Ensemble utilities may assume that candidate artifacts share:
  - the same class ordering
  - the same label space
  - compatible `sample_ids`
  - the same submission schema intent

### Must not break

- `prediction.csv` must remain the exact output filename for homework submission generation.
- The row count of `prediction.csv` must match the number of test samples seen by the inference dataset.
- Numeric-label datasets must continue to round-trip back to original label IDs when `use_label_name` is enabled.
- The repo must not silently switch from soft-vote artifact handling to hard-vote semantics when probability artifacts are missing or malformed.
- Future automation must not treat `best.ckpt` as equivalent to "best selected weights" without specifying raw vs EMA.

## Scenarios

### Scenario: Standard single-model inference succeeds

- Given a valid config and checkpoint
- When `scripts/infer.py` runs
- Then the output directory must contain `prediction.csv`
- And the CSV row count must match the test set size

### Scenario: `best_ema.ckpt` auto-loads the EMA branch

- Given a checkpoint path ending in `best_ema.ckpt`
- And the checkpoint contains `ema_state_dict`
- When `scripts/infer.py` runs without `--use-ema`
- Then the script must load the EMA branch automatically

### Scenario: `best.ckpt` selected on EMA but inferred without `--use-ema`

- Given `best.ckpt` came from an epoch selected by EMA metrics
- When `scripts/infer.py --ckpt <best.ckpt>` runs without `--use-ema`
- Then the script loads the raw `state_dict` from that epoch, not the EMA branch
- And the resulting submission is not guaranteed to match the model-selection branch that won during training

### Scenario: Numeric labels are restored for submission

- Given `idx_to_label` maps internal indices to numeric label IDs
- And `inference.output.use_label_name: true`
- When `prediction.csv` is written
- Then `pred_label` values must contain the original homework label IDs rather than internal class indices

### Scenario: Saving probability artifacts preserves alignment

- Given `scripts/infer.py --save-probs`
- When inference completes
- Then `test_probs_with_ids.npz` must contain `sample_ids` aligned with the saved probability rows
- And any optional logits must share the same row order

### Scenario: Soft-vote ensemble with missing label mapping fails

- Given `scripts/ensemble.py --mode=soft --use-label-name`
- And no `idx_to_label.json` can be resolved
- When the command runs
- Then it must fail rather than emit a submission with ambiguous label semantics

### Scenario: Mismatched sample IDs fail fast

- Given two probability artifacts with different sample-ID order or coverage
- When `scripts/ensemble.py --mode=soft` or `scripts/ensemble_search.py` uses them
- Then the command must fail rather than silently realign or truncate data

### Scenario: Mismatched label spaces are invalid even if shape matches

- Given two probability artifacts with the same tensor shape but different class ordering or label mapping
- When they are considered for soft-vote or search
- Then the candidate pool is invalid
- And current scripts are allowed to assume class-order consistency even though they do not fully validate it

## Known ambiguities or gaps

- The repo writes `prediction.csv` but does not automate the final zip packaging step.
- The competition schema is represented by config defaults and docs rather than a dedicated schema validator.
- `use_label_name: false` on numeric-label datasets only raises a warning; it does not block an obviously risky submission.
- The scripts validate sample-ID alignment more aggressively than class-order alignment.

## Non-goals

- This spec does not require new TTA modes beyond the current horizontal-flip path.
- This spec does not require a packaging CLI for CodaBench or E3.
- This spec does not define ensemble policy by itself; detailed ensemble behavior lives in the ensemble/search spec.
