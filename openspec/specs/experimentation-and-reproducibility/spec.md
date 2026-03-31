# Experimentation and Reproducibility Spec

## Overview

This spec defines the minimum reproducibility and experiment-adoption contract for the current repository.
It is derived from [configs/config.yaml](../../../configs/config.yaml), the batch scripts in `scripts/`, [src/utils/misc.py](../../../src/utils/misc.py), [src/engine/trainer.py](../../../src/engine/trainer.py), [src/utils/run_metadata.py](../../../src/utils/run_metadata.py), and the existing run-artifact patterns under `outputs/`.

This spec must be read together with:
- [../homework-requirements/spec.md](../homework-requirements/spec.md)
- [../evaluation-and-validation/spec.md](../evaluation-and-validation/spec.md)
- [../ensemble-and-search/spec.md](../ensemble-and-search/spec.md)
- [../report-and-method-writeup/spec.md](../report-and-method-writeup/spec.md)

## Requirements

### Observed current behavior

- Configs are layered by merging the base config with optional model, train, augmentation, and inference overrides.
- Training runs are written to timestamped directories under `project.output_root`.
- A modern successful run can include:
  - `config.yaml`
  - `train.log`
  - `history.csv`
  - `history.json`
  - `summary.json`
  - `run_metadata.json`
  - `model_metadata.json`
  - `label_to_idx.json`
  - `idx_to_label.json`
  - checkpoints under `checkpoints/`
- Seeds come from `project.seed`.
- Deterministic mode is optional through `project.deterministic` and defaults to `false`.
- Batch scripts run jobs sequentially and accept either direct checkpoint paths or run-dir-plus-checkpoint-name references, depending on the script.
- Batch scripts fail fast on ambiguous job definitions such as:
  - both `ckpt` and `run_dir`+`checkpoint_name`
  - both `init_ckpt` and `init_from_run_dir`
- `continue_on_error` is configurable. If false, the queue stops on the first failure.
- Newer runs may contain stage-gate metadata and dense checkpoint inventories. Older runs may not.

### Required preserved behavior

- A recipe is only reproducible in this repo if the config snapshot and run artifacts are saved together.
- A recipe is only adoptable for future homework-facing work if the evidence bundle includes:
  - the training config snapshot
  - `summary.json`
  - parameter-count evidence in `model_metadata.json`
  - the selected-branch validation result
  - if the recipe is submission-facing, a successful inference artifact showing `prediction.csv`
- For ensemble adoption, the evidence bundle should also include the relevant manifest, diversity output, or ensemble-search summary.
- Seeds and deterministic mode must remain explicit user-controlled configuration, not hidden defaults inside scripts.
- Batch execution must remain sequential and single-GPU practical by default.
- Ambiguous batch job definitions must continue to fail rather than guess the user's intent.

### Must not break

- A successful run must remain self-describing from its own directory.
- Checkpoint names must remain stable enough for validation and inference automation that refers to `best`, `best_raw`, `best_ema`, and `last`.
- Future changes must not blur the distinction between:
  - recipe reproducibility
  - bitwise determinism
  - leaderboard reproducibility
- Legacy runs that predate newer metadata files must not be silently treated as fully compliant with the current evidence contract.

## Scenarios

### Scenario: Re-running a recipe with the same config

- Given a saved run directory with `config.yaml`
- When a user reruns the same recipe with the same seed
- Then the repo should reproduce the same declared recipe and artifact structure
- Even if exact floating-point results differ when `project.deterministic` is false

### Scenario: Deterministic mode raises the reproducibility bar

- Given `project.deterministic: true`
- When the same run is repeated in the same environment
- Then stronger repeatability is expected than in the default mode
- But the repo still does not guarantee portability across different PyTorch/CUDA environments

### Scenario: Batch config is ambiguous and must fail

- Given a validation or inference batch job that provides both a direct `ckpt` and `run_dir` plus `checkpoint_name`
- When the batch script parses the job
- Then the job must fail instead of guessing which checkpoint reference to use

### Scenario: Batch queue stops on first failure by default

- Given a batch config with `continue_on_error: false`
- When one job exits non-zero
- Then later jobs must not run unless the user explicitly enables continue-on-error behavior

### Scenario: Dense checkpoint retention records fragile peaks

- Given `save_every_epoch: true` and `keep_top_k > 0`
- When a run finishes
- Then the retained epoch checkpoints and the metric used to retain them must be recorded in `summary.json`
- And `run_metadata.json` should expose the checkpoint inventory

### Scenario: Recipe adoption requires more than a public leaderboard anecdote

- Given a user wants to adopt a new recipe as the repo's preferred baseline
- When that decision is documented
- Then the minimum evidence must include the local run artifacts defined in this spec
- And a public leaderboard improvement by itself is not enough

### Scenario: Legacy run is missing newer metadata

- Given an older run directory without `run_metadata.json`
- When it is compared against a new run
- Then it may still be useful historical evidence
- But it does not satisfy the full current reproducibility contract

## Known ambiguities or gaps

- Default runs are seeded but not fully deterministic because `project.deterministic` defaults to `false`.
- The environment files do not fully pin the CUDA-enabled PyTorch build used in practice.
- There is no central experiment registry beyond output directories and their local metadata files.
- Batch scripts do not provide resource locking, queue recovery, or automatic retries.

## Non-goals

- This spec does not require CI, a database-backed tracker, or a hosted experiment platform.
- This spec does not require automatic report generation from run directories.
- This spec does not define architecture legality; that belongs to the homework and model-architecture specs.
