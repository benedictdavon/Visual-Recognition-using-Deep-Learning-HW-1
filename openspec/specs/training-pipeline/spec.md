# Training Pipeline Spec

## Overview

This spec captures the current training contract of the repository.
It is derived from [scripts/train.py](../../../scripts/train.py), [src/data/dataset.py](../../../src/data/dataset.py), [src/data/transforms.py](../../../src/data/transforms.py), [src/data/samplers.py](../../../src/data/samplers.py), [src/models/builder.py](../../../src/models/builder.py), [src/losses/losses.py](../../../src/losses/losses.py), [src/engine/trainer.py](../../../src/engine/trainer.py), and [src/utils/checkpoint.py](../../../src/utils/checkpoint.py).

This spec must be read together with:
- [../homework-requirements/spec.md](../homework-requirements/spec.md)
- [../model-architecture/spec.md](../model-architecture/spec.md)
- [../data-and-split/spec.md](../data-and-split/spec.md)
- [../evaluation-and-validation/spec.md](../evaluation-and-validation/spec.md)

## Requirements

### Observed current behavior

- `scripts/train.py` merges `configs/config.yaml` with optional override YAML files, sets the seed, creates the run directory, and saves the merged config before training begins.
- The training script always prepares both a training dataframe and a validation dataframe through `prepare_dataframes`.
- The current local default dataset uses the explicit folder validation split under `data/val`.
- Training uses `FlexibleImageDataset` plus transform builders from `src/data/transforms.py`.
- The training dataloader:
  - shuffles when no sampler is provided
  - disables shuffle when a sampler is provided
  - uses `drop_last=True` for the training loader
- The validation dataloader does not shuffle.
- Supported optimizer names are `adamw` and `sgd`.
- Supported scheduler names are `cosine` and `none`.
- Supported loss names are:
  - `cross_entropy`
  - `focal`
  - `balanced_softmax`
  - `logit_adjusted_ce`
- Supported sampler names are:
  - `none`
  - `weighted_random`
- Model warm-starting is available through:
  - `--init-ckpt`
  - `train.init_checkpoint`
  - optional `--init-use-ema` / `train.init_use_ema`
- Current warm-start behavior is asymmetric with validation and inference:
  - if `init_use_ema` is requested and the checkpoint contains `ema_state_dict`, EMA weights are used
  - if `init_use_ema` is requested and the checkpoint does not contain `ema_state_dict`, the code silently falls back to the standard state dict
- A successful training run currently writes:
  - `config.yaml`
  - `train.log`
  - `history.csv`
  - `history.json`
  - `summary.json`
  - `run_metadata.json`
  - `model_metadata.json`
  - `label_to_idx.json`
  - `idx_to_label.json`
  - `checkpoints/last.ckpt`
  - `checkpoints/best.ckpt`
  - `checkpoints/best_raw.ckpt`
  - `checkpoints/best_ema.ckpt` when EMA is enabled
  - optional `checkpoints/epoch_XXX.ckpt` files when dense checkpointing is enabled

### Required preserved behavior

- Training must remain config-driven and launched through `scripts/train.py`.
- The default training path must remain practical on a single GPU and non-distributed.
- Training must continue to rely on the repo's data-preparation path rather than ad hoc data loading in scripts.
- Model legality for homework use is governed by [../model-architecture/spec.md](../model-architecture/spec.md), not by successful model construction alone.
- A successful training run must remain self-describing from its output directory. At minimum it must preserve the artifact set listed above.
- Validation must continue to run every epoch and drive checkpoint selection.
- The selected model-selection metric, source, mode, and value must remain visible in logs and persisted in `summary.json`.
- Unsupported optimizer, scheduler, loss, sampler, or model names must continue to fail fast rather than silently fall back.
- Tail-aware losses that require valid class counts must continue to reject invalid class-count inputs rather than silently degrading to plain cross-entropy.

### Must not break

- Training must not silently ignore the explicit validation ownership rules defined in [../data-and-split/spec.md](../data-and-split/spec.md).
- Training must not bypass `src/models/builder.py` when constructing homework-facing models.
- Successful runs must not omit `run_metadata.json`, label maps, or `model_metadata.json`.
- The default path must not require DDP, multi-GPU launchers, or external orchestration.
- The current warm-start semantics must not be changed silently. Any future change to the `init_use_ema` fallback behavior must be called out explicitly because automation may already depend on the current asymmetry.

## Scenarios

### Scenario: Successful run writes the required artifact suite

- Given a valid configuration and a completed training run
- When the run exits successfully
- Then the run directory must contain the config snapshot, logs, summaries, label maps, model metadata, and checkpoint files required by this spec

### Scenario: Explicit validation folder remains active

- Given the default local config with `dataset.val_dir: val`
- When `scripts/train.py` runs
- Then training must use `data/val` as validation
- And it must not create a new random train/val split from `data/train`

### Scenario: `init_use_ema` loads EMA weights when available

- Given `--init-ckpt <path>` and `--init-use-ema`
- And the checkpoint contains `ema_state_dict`
- When training starts
- Then model initialization must use the EMA weights from that checkpoint

### Scenario: `init_use_ema` silently falls back to raw weights today

- Given `--init-ckpt <path>` and `--init-use-ema`
- And the checkpoint does not contain `ema_state_dict`
- When training starts
- Then the current code falls back to the standard state dict without raising
- And future automation must not assume that `init_use_ema` guarantees actual EMA initialization

### Scenario: Unsupported sampler fails fast

- Given `sampler.name` is set to a value other than `none` or `weighted_random`
- When `scripts/train.py` builds the train sampler
- Then the run must fail before training starts

### Scenario: Tail-aware loss rejects invalid class counts

- Given `loss.name: balanced_softmax` or `loss.name: logit_adjusted_ce`
- And at least one class count is non-positive
- When `build_loss` is called
- Then the run must fail rather than silently use an incorrect loss

### Scenario: Code-supported architecture is not automatically homework-safe

- Given a configuration whose `model.name` is supported by code
- When a user considers it for homework use
- Then the run is still governed by the legality tiers in [../model-architecture/spec.md](../model-architecture/spec.md)
- And successful construction alone is not sufficient evidence of homework compliance

### Scenario: Dense checkpoint retention follows the selected metric

- Given `train.checkpointing.save_every_epoch: true` and `keep_top_k > 0`
- When training runs
- Then retained `epoch_XXX.ckpt` files must be filtered according to the configured model-selection metric and mode
- And the retained set must be recorded in `summary.json`

## Known ambiguities or gaps

- `scripts/train.py` exposes `--resume`, but the flag is reserved and not wired into the current training loop.
- `SoftTargetCrossEntropy` exists in `src/losses/losses.py`, but the current mixup/cutmix path uses interpolated hard-target losses instead.
- The training loop does not implement gradient accumulation, distributed training, or multi-GPU coordination.
- The warm-start log line does not distinguish between true EMA initialization and fallback-to-raw behavior when `init_use_ema` is requested but EMA weights are absent.

## Non-goals

- This spec does not require resume support, distributed training, or automatic hyperparameter search.
- This spec does not define validation policy in isolation; that belongs to the evaluation and data/split specs.
- This spec does not bless every code-supported architecture for homework use.
