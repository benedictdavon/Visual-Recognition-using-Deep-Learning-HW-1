# Data And Split Spec

## Overview

This spec defines the repository's current data-layout, split, and sample-ID contract.
It is derived from [src/data/dataset.py](../../../src/data/dataset.py), [src/data/split.py](../../../src/data/split.py), [scripts/inspect_dataset.py](../../../scripts/inspect_dataset.py), [configs/config.yaml](../../../configs/config.yaml), and the current local dataset diagnostics under `outputs/`.

This spec must be read together with:
- [../homework-requirements/spec.md](../homework-requirements/spec.md)
- [../evaluation-and-validation/spec.md](../evaluation-and-validation/spec.md)
- [../inference-and-submission/spec.md](../inference-and-submission/spec.md)

## Requirements

### Observed current behavior

- The repo supports `dataset.dataset_type` values:
  - `auto`
  - `folder`
  - `csv`
- The current local repo uses a folder dataset with:
  - `data/train/<class_id>/*.jpg`
  - `data/val/<class_id>/*.jpg`
  - `data/test/*.jpg`
- In folder mode:
  - `label_name` is the folder name
  - `label_to_idx` is built by lexicographically sorting class-folder names
  - `sample_id` is the image filename stem
- In CSV mode:
  - train `sample_id` is the filename stem
  - test `sample_id` uses `id_col` if present, otherwise filename stem
- `dataset.test_id_from` only affects the ID column used in the final submission. It does not change how internal dataset `sample_id` is built.
- Explicit validation-folder behavior in `prepare_dataframes` is:
  - if `dataset.val_dir` is explicitly present in config and resolves to a real folder, use it
  - if `dataset.val_dir` is explicitly `null` or blank, do not auto-pick `data/val`
  - if `dataset.val_dir` is absent from config and `data/val` exists, auto-pick `data/val`
- If no validation dataframe is available:
  - use K-fold when `dataset.kfold.enabled: true`
  - otherwise create a random train/val split from the training dataframe
- The repo contains duplicate-analysis artifacts in `outputs/data_analysis_summary.json`, but duplicate detection is not enforced in the runtime data loader.
- There is no core train+val refit path in the normal training scripts.

### Required preserved behavior

- The "official validation set" in this repo means the explicit validation folder used by `prepare_dataframes` when `dataset.val_dir` is active.
- The "internal development split" means the split created from training data when the official validation set is absent or explicitly disabled.
- The official validation set must take precedence by default in the current local dataset.
- A config that explicitly sets `dataset.val_dir: null` or blank is intentionally changing split ownership and must not be mistaken for the default path.
- Sample IDs must remain stable across validation and inference artifacts. Downstream utilities rely on them for alignment.
- Numeric folder labels must continue to round-trip safely through `idx_to_label` and the submission path.
- Test data must remain unlabeled from the perspective of training, validation, and recipe selection.
- If a future train+val refit path is added, it must be explicit and separate from the default training path rather than silently overloading `scripts/train.py`.

### Must not break

- The current local folder layout must remain supported.
- Explicit validation ownership must not be silently ignored.
- Internal split creation must not activate when an official validation set is already active.
- `sample_id` meaning must remain consistent enough for validation, inference, and ensemble artifacts to align.
- Test data must not be repurposed as labeled validation data.

## Scenarios

### Scenario: Default local folder dataset uses the official validation folder

- Given the current base config with `dataset.val_dir: val`
- And `data/val` exists
- When `prepare_dataframes` runs
- Then the official validation set must come from `data/val`
- And the training dataframe must remain sourced from `data/train`

### Scenario: Explicitly disabling `val_dir` forces internal splitting

- Given `dataset.val_dir` is explicitly set to `null` or an empty string
- When `prepare_dataframes` runs in folder mode
- Then the code must not auto-pick `data/val`
- And validation must come from K-fold or a train/val split instead

### Scenario: Numeric label folders require mapping instead of index identity

- Given class folders named `0..99`
- When folder labels are sorted lexicographically
- Then internal class indices are not guaranteed to equal numeric class IDs
- And downstream consumers must use `idx_to_label` for label restoration

### Scenario: CSV test IDs prefer `id_col`

- Given a CSV test file that includes both filename and ID columns
- When `prepare_dataframes` builds the test dataframe
- Then `sample_id` must use the configured `id_col`
- And later submission behavior may still choose between `sample_id` and `filename` through `dataset.test_id_from`

### Scenario: `auto` detection is not fully consistent across code paths

- Given both folder-style and CSV-style dataset artifacts exist
- When `inspect_dataset_layout` is called
- Then it prefers reporting folder mode first
- But when `prepare_dataframes` runs in `dataset_type: auto`
- Then it prefers CSV mode if `train.csv` exists
- And this inconsistency is a current repo gap that future changes must handle explicitly

### Scenario: Duplicate sample IDs are invalid inputs even if not fully checked

- Given a split contains duplicate `sample_id` values
- When downstream validation, inference, or ensemble utilities consume artifacts from that split
- Then the input should be treated as invalid
- Because current scripts assume `sample_id` uniqueness but do not robustly enforce it

### Scenario: Test data may not be used as a labeled selector

- Given predictions on `data/test`
- When a user considers using them as if they were labeled validation results
- Then that workflow is invalid under this spec

## Known ambiguities or gaps

- `inspect_dataset_layout` and `prepare_dataframes` use different `auto`-detection precedence when both folder and CSV hints exist.
- Runtime code does not enforce duplicate or leakage checks even though out-of-band diagnostics exist.
- `sample_id` uniqueness is assumed more strongly than it is validated.
- The core training path does not provide an explicit train+val refit mode.

## Non-goals

- This spec does not require duplicate-removal logic.
- This spec does not define architecture legality or checkpoint policy.
- This spec does not require K-fold to be active by default.
