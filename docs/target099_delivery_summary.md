# Target 0.99 Delivery Summary

## Implemented

- Repo audit and execution mapping:
  - `docs/target099_execution_plan.md`
  - `docs/target099_progress_log.md`
- Validation/model-selection infrastructure:
  - `val_acc`
  - `val_macro_recall`
  - `val_nll`
  - `val_ece`
  - configurable model-selection metric
  - dense checkpoint retention
  - structured `run_metadata.json`
- Tail-aware training probes:
  - `balanced_softmax`
  - `logit_adjusted_ce`
  - weighted random sampler with inverse-power weighting
- Short FixRes refresh support:
  - explicit warm-start fields in base config
  - short-refresh template and Stage 0 branch configs
- Diversity-first ensemble tooling:
  - `scripts/diversity_report.py`
  - `scripts/ensemble_search.py`
  - `val_probs_with_ids.npz` export from validation
- Stage framework:
  - Stage 0 runnable configs and runbook
  - Stage 1 scaffold configs
  - Stage 2 scaffold configs
- Optional model support:
  - `wide_resnet50_2`

## Post-Delivery Additions

- LDAM-DRW late reweighting path implemented and validated
- branch-aware diversity-first residual probe workflow implemented and validated
- `timm`-backed `resnetv2_101x1_bit.goog_in21k_ft_in1k` support added to the legal residual-family path
- strong `target099` BiT experiment config added and exercised on real training runs

## Post-Delivery Experiment Results

- LDAM-DRW full run:
  - `outputs/target099_ldam_drw/20260330_151455_target099_ldam_drw_resnet101`
  - selected result:
    - `val_acc = 89.00`
    - `val_nll = 3.3001`
  - verdict:
    - implementation is correct
    - experiment path is not competitive
- ResNeXt50 diversity probe rerun:
  - `outputs/target099_diversity_probe/20260330_224556_target099_diversity_probe_resnext50_32x3d_full`
  - best raw:
    - `val_acc = 89.00`
    - `val_nll = 0.4936`
  - best EMA:
    - `val_acc = 19.00`
    - `val_nll = 3.7905`
  - verdict:
    - raw branch remains a possible ensemble candidate
    - EMA branch is dead
    - not a new anchor
- BiT ResNetV2-101x1 strong run:
  - `outputs/bit_resnetv2_101x1/20260331_120013_target099_bit_resnetv2_101x1_strong`
  - training crashed during `epoch_025.ckpt` write, but `best_ema.ckpt` survived
  - validated EMA result:
    - `val_acc = 92.33`
    - `val_nll = 0.3806`
    - `val_ece = 0.0791`
  - verdict:
    - current best local single-model checkpoint

## Current Single-Model Ranking

1. BiT `best_ema.ckpt`
   - `92.33 / 0.3806`
2. Stage 0 FixRes from `s0_r1` selected checkpoint
   - `91.00 / 0.3986`
3. CRT FixRes-from-rebalance selected checkpoint
   - `90.33 / 0.3587`

## Main Files Changed

### Runtime code
- `scripts/train.py`
- `scripts/validate.py`
- `scripts/infer.py`
- `scripts/diversity_report.py`
- `scripts/ensemble_search.py`
- `src/engine/evaluator.py`
- `src/engine/trainer.py`
- `src/losses/losses.py`
- `src/data/dataset.py`
- `src/data/samplers.py`
- `src/models/resnet_variants.py`
- `src/utils/branch_provenance.py`
- `src/utils/metrics.py`
- `src/utils/run_metadata.py`

### Configs
- `configs/config.yaml`
- `configs/experiments/target099/*.yaml`
- `configs/experiments/target099_fixres320_short_refresh_template.yaml`
- `configs/model/resnetv2_101x1_bit_goog_in21k_ft_in1k.yaml`
- `configs/batch/*.yaml`
- `configs/manifests/*.yaml`

### Docs
- `docs/target099_execution_plan.md`
- `docs/target099_progress_log.md`
- `docs/target099_delivery_summary.md`
- `docs/experiments/stage0.md`
- `docs/experiments/stage1.md`
- `docs/experiments/stage2.md`
- `docs/experiments/stage0_runbook.md`
- `docs/experiments/losses_tail_aware.md`
- `docs/experiments/sampler_tail_aware.md`
- `docs/experiments/fixres_short_refresh.md`
- `docs/experiments/diversity_first_ensemble.md`
- `docs/experiments/ensemble_search.md`
- `docs/experiments/ldam_drw.md`
- `docs/experiments/diversity_first_residual_probe.md`
- `docs/experiments/bit_resnetv2_101x1.md`
- `README.md`

### Dependencies
- `requirements.txt`

## How To Run Stage 0

Use the exact commands in `docs/experiments/stage0_runbook.md`.

Core train commands:

```bash
python scripts/train.py --config configs/config.yaml --train-config configs/experiments/target099/stage0_s0_r1_resnet101_logit_adjusted.yaml --output-dir outputs/target099_stage0
python scripts/train.py --config configs/config.yaml --train-config configs/experiments/target099/stage0_s0_r2_resnet101_balanced_softmax.yaml --output-dir outputs/target099_stage0
python scripts/train.py --config configs/config.yaml --train-config configs/experiments/target099/stage0_s0_r3_resnet101_sampler_inv_sqrt.yaml --output-dir outputs/target099_stage0
```

Optional pilot:

```bash
python scripts/train.py --config configs/config.yaml --train-config configs/experiments/target099/stage0_s0_r5_wide_resnet50_2_optional.yaml --output-dir outputs/target099_stage0
```

Follow-up short FixRes:
- run exactly one of the `stage0_s0_r4_fixres320_from_s0r*.yaml` configs based on the winning Stage 0 base

## Diversity Workflow

1. Validate promising checkpoints so `val_probs_with_ids.npz` exists.
2. Save test probs with `scripts/infer.py --save-probs`.
3. Build a manifest and run:

```bash
python scripts/diversity_report.py --manifest outputs/diversity_manifest.yaml --output-dir outputs/diversity_stage0
```

## Ensemble Search Workflow

Run the greedy NLL-aware search after diversity filtering:

```bash
python scripts/ensemble_search.py --manifest outputs/diversity_manifest.yaml --anchor <anchor_name> --diversity-summary outputs/diversity_stage0/diversity_summary.csv --output-dir outputs/ensemble_search_stage0
```

If all selected models include test probabilities, the script writes:
- `prediction.csv`
- `ensemble_test_probs_with_ids.npz`

## What Remains

- submission decisions from the current checkpoints
- optional ensemble work if the single-model submissions plateau
- optional cleanup of checkpoint retention to reduce large run-directory storage pressure

## Repo-Forced Deviation

- A fully runnable train+val refit template is intentionally deferred.
- Reason:
  - the current training loop still expects a validation loader for checkpoint selection/reporting
  - adding a validation-free training path now would be broader than the approved Stage 0 scope
