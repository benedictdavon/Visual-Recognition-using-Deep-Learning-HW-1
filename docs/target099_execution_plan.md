# Target 0.99 Execution Plan

Source of truth: `docs/handoff_trial_summary_target099.md`

## Repo Structure Map

### Entry points
- `scripts/train.py`: main training entrypoint, config merge, run-dir creation, checkpoint init, train/val loader build.
- `scripts/validate.py`: checkpoint validation and detailed val artifact export.
- `scripts/infer.py`: test inference, optional TTA, probability/logit artifact export, `prediction.csv` generation.
- `scripts/train_batch.py`: sequential multi-run launcher for training.
- `scripts/validate_batch.py`: sequential multi-run launcher for validation.
- `scripts/infer_batch.py`: sequential multi-run launcher for inference.
- `scripts/ensemble.py`: current simple hard/soft voting utility.

### Core runtime modules
- `src/data/dataset.py`: dataset discovery, folder/CSV handling, train/val split, label mapping, dataframe preparation.
- `src/data/transforms.py`: train/eval transform construction, including current FixRes-style size overrides.
- `src/engine/trainer.py`: optimizer, scheduler, train loop, validation loop call, checkpoint saving, history/summary export.
- `src/engine/evaluator.py`: validation metrics computation.
- `src/engine/inference.py`: probability prediction and current horizontal-flip TTA.
- `src/losses/losses.py`: current CE and focal loss builder.
- `src/models/builder.py`: model construction and parameter-limit enforcement.
- `src/models/resnet_variants.py`: supported residual-family backbones and legal ResNet-based modifications.
- `src/utils/checkpoint.py`: checkpoint save/load/init helpers.
- `src/utils/metrics.py`: metric helpers.
- `src/submission/make_submission.py`: final `prediction.csv` formatting and write path.

### Config and artifact layout
- `configs/config.yaml`: base config.
- `configs/experiments/*.yaml`: one-file train experiment configs. Current strongest base recipe is `configs/experiments/resnet101_strong_v2_fast*.yaml`.
- `configs/batch/*.yaml`: sequential job queues for train/validate/infer.
- `configs/inference/*.yaml`: current TTA/default inference settings.
- `outputs/<run_dir>/`: current training run artifacts (`config.yaml`, `history.csv`, `history.json`, `summary.json`, `model_metadata.json`, `checkpoints/`).
- `outputs/validate_batch/*/`: current validation analysis exports.
- `outputs/infer_batch/*/test_probs_with_ids.npz`: current ensemble-ready probability pool referenced by the handoff.
- `outputs/ensemble_sweep/*`: prior sweep outputs cited in the handoff.

## Mapping Handoff Recommendations To Actual Files

### Better model-selection metrics
- Implement in `src/engine/evaluator.py` and `src/utils/metrics.py`.
- Consume and act on them in `src/engine/trainer.py`.
- Surface them in `scripts/validate.py` exports and run summaries.

### Denser checkpointing around early useful epochs
- Implement in `src/engine/trainer.py` with helper support in `src/utils/checkpoint.py` if needed.
- Expose via config in `configs/config.yaml` and Stage 0 experiment configs.

### Tail-aware loss probes
- Implement `balanced_softmax` and `logit_adjusted_ce` in `src/losses/losses.py`.
- Derive training class counts from `prepare_dataframes()` output in `scripts/train.py` and `scripts/validate.py`.

### Sampler probes
- Add weighted sampler support near dataloader construction in `scripts/train.py`.
- Put sampler math in a small new helper under `src/data/`.

### Short low-LR FixRes refresh
- Reuse current warm-start path already present in `scripts/train.py` plus existing config fields:
  - `dataset.image_size`
  - `augmentation.eval.resize`
  - `augmentation.eval.center_crop`
  - `train.init_checkpoint`
  - `train.init_use_ema`
  - `mixup_cutmix.enabled`
- Formalize with a dedicated template config and documentation.

### Diversity analysis utilities
- Add new standalone analysis script instead of rewriting `scripts/ensemble.py`.
- Inputs should be saved val/test logits or probs from existing validation/inference paths.

### Safer ensemble search utilities
- Add a second standalone script for greedy forward selection and coarse weight search.
- Keep existing `scripts/ensemble.py` backward compatible as the simple baseline tool.

## Repo-Specific Constraints Discovered

- There is no existing automated test suite. Sanity checks must be lightweight script-level checks.
- The current validation loop only computes `loss`, `acc1`, `acc5`, and macro per-class accuracy; it does not yet expose NLL or ECE.
- Best-checkpoint selection is currently accuracy-driven and EMA-preferred when enabled.
- Checkpoint retention is currently limited to `best.ckpt`, `best_raw.ckpt`, `best_ema.ckpt`, and `last.ckpt`.
- Current TTA is horizontal-flip only, implemented in `src/engine/inference.py`.
- Current FixRes support is already functional through config overrides and checkpoint warm-starting; this should be extended, not rewritten.
- Current ensemble search is minimal and does not score candidates by diversity, rescue count, or NLL.
- `wide_resnet50_2` is not currently implemented in `src/models/resnet_variants.py`; if added, it must be clearly marked as legality-sensitive/optional in docs and configs.
- The workspace is not a Git repository, so change tracking must rely on the new progress log and delivery summary docs.

## Sequential Implementation Order

### Phase A
1. Finish audit and create this execution plan.
2. Initialize `docs/target099_progress_log.md`.

### Phase B
1. Add validation metrics: `val_acc`, `val_macro_recall`, `val_nll`, `val_ece`.
2. Add configurable model-selection metric support while preserving current default behavior.
3. Add checkpoint density support: `save_every_epoch`, `keep_top_k`.
4. Add structured per-run metadata output with rationale, stage-gate status, checkpoint inventory, and metrics summary.
5. Create `docs/experiments/stage0.md`, `stage1.md`, `stage2.md`.

### Phase C
1. Add Balanced Softmax.
2. Add logit-adjusted cross entropy.
3. Keep plain CE path unchanged.
4. Document tail-aware loss usage.

### Phase D
1. Add weighted random sampler with configurable power.
2. Wire sampler config into actual train loader build.
3. Document sampler probes.

### Phase E
1. Verify and extend short FixRes refresh config support.
2. Add ready-to-run tail-aware-base -> short FixRes template.
3. Document the short refresh workflow.

### Phase F
1. Add diversity analysis script for val/test logits/probs.
2. Add thresholded include/exclude logic matching the handoff’s diversity-first diagnosis.
3. Document workflow.

### Phase G
1. Add greedy ensemble search script with anchor locking, coarse weights, NLL-aware scoring, and pool cap 4.
2. Preserve `prediction.csv` output path when test probs are available.
3. Document workflow and safeguards.

### Phase H
1. Add Stage 0 configs for low-cost probes derived from the strongest current base recipe.
2. Add exact commands, kill criteria, and success criteria in a Stage 0 runbook.

### Phase I
1. Add Stage 1 and Stage 2 config/document skeletons only.
2. Do not launch or fully expand later stages before Stage 0 results exist.

### Phase J
1. Write a final delivery summary.
2. Update progress log with final status and remaining work.

## Current Baseline For Deriving Stage 0

- Base recipe anchor: `configs/experiments/resnet101_strong_v2_fast.yaml`
- Existing seed variants:
  - `configs/experiments/resnet101_strong_v2_fast_seed42.yaml`
  - `configs/experiments/resnet101_strong_v2_fast_seed3407.yaml`
  - `configs/experiments/resnet101_strong_v2_fast_seed2026.yaml`
- Existing short FixRes reference:
  - `configs/experiments/fixres320_seed42_short3.yaml`
  - `configs/experiments/fixres320_seed3407_short3.yaml`
  - `configs/experiments/fixres320_seed2026_short3.yaml`

Stage 0 should therefore be implemented as minimal deltas on top of the `resnet101_strong_v2_fast` recipe unless the handoff explicitly requires otherwise.
