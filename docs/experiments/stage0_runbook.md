# Stage 0 Runbook

Source of truth for why these runs exist: `docs/handoff_trial_summary_target099.md`

Anchor baseline for Stage 0 comparisons:
- current strongest base recipe: `resnet101_strong_v2_fast_seed2026`
- historical local reference:
  - `val_acc`: about `90.0`
  - `val_macro_recall`: about `90.0`

Common Stage 0 discipline across S0-R1 / S0-R2 / S0-R3:
- single seed: `2026`
- model selection: `val_nll`
- dense checkpoint retention:
  - `save_every_epoch: true`
  - `keep_top_k: 5`
- required post-train inspection:
  - `summary.json`
  - `run_metadata.json`
  - `history.csv`
  - retained `epoch_*.ckpt`

## Run Order

1. `S0-R1` logit-adjusted CE
2. `S0-R2` Balanced Softmax
3. `S0-R3` inverse-sqrt sampler + plain CE
4. Pick the winning base recipe from `S0-R1..R3`
5. `S0-R4` short FixRes refresh using the matching branch config
6. `S0-R5` optional WideResNet50-2 pilot only if legality is accepted

## S0-R1

- Config: `configs/experiments/target099/stage0_s0_r1_resnet101_logit_adjusted.yaml`
- Delta vs current baseline:
  - `loss.name`: `cross_entropy -> logit_adjusted_ce`
  - `loss.logit_adjusted_tau: 1.0`
  - `train.model_selection.metric: val_nll`
  - dense checkpoint retention enabled
- Expected runtime:
  - about `85-95 min` on the current single-GPU setup
- Train command:

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --train-config configs/experiments/target099/stage0_s0_r1_resnet101_logit_adjusted.yaml \
  --output-dir outputs/target099_stage0
```

## S0-R2

- Config: `configs/experiments/target099/stage0_s0_r2_resnet101_balanced_softmax.yaml`
- Delta vs current baseline:
  - `loss.name`: `cross_entropy -> balanced_softmax`
  - `train.model_selection.metric: val_nll`
  - dense checkpoint retention enabled
- Expected runtime:
  - about `85-95 min`
- Train command:

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --train-config configs/experiments/target099/stage0_s0_r2_resnet101_balanced_softmax.yaml \
  --output-dir outputs/target099_stage0
```

## S0-R3

- Config: `configs/experiments/target099/stage0_s0_r3_resnet101_sampler_inv_sqrt.yaml`
- Delta vs current baseline:
  - `sampler.name`: `weighted_random`
  - `sampler.power`: `0.5` (inverse-sqrt)
  - `loss.name`: stays `cross_entropy`
  - `train.model_selection.metric: val_nll`
  - dense checkpoint retention enabled
- Expected runtime:
  - about `85-95 min`
- Train command:

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --train-config configs/experiments/target099/stage0_s0_r3_resnet101_sampler_inv_sqrt.yaml \
  --output-dir outputs/target099_stage0
```

## S0-R4

Run exactly one branch after `S0-R1..R3` finish.

### If S0-R1 wins
- Config: `configs/experiments/target099/stage0_s0_r4_fixres320_from_s0r1.yaml`

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --train-config configs/experiments/target099/stage0_s0_r4_fixres320_from_s0r1.yaml \
  --output-dir outputs/target099_stage0 \
  --init-ckpt outputs/target099_stage0/<winning_s0_r1_run_dir>/checkpoints/best.ckpt \
  --init-use-ema
```

### If S0-R2 wins
- Config: `configs/experiments/target099/stage0_s0_r4_fixres320_from_s0r2.yaml`

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --train-config configs/experiments/target099/stage0_s0_r4_fixres320_from_s0r2.yaml \
  --output-dir outputs/target099_stage0 \
  --init-ckpt outputs/target099_stage0/<winning_s0_r2_run_dir>/checkpoints/best.ckpt \
  --init-use-ema
```

### If S0-R3 wins
- Config: `configs/experiments/target099/stage0_s0_r4_fixres320_from_s0r3.yaml`

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --train-config configs/experiments/target099/stage0_s0_r4_fixres320_from_s0r3.yaml \
  --output-dir outputs/target099_stage0 \
  --init-ckpt outputs/target099_stage0/<winning_s0_r3_run_dir>/checkpoints/best.ckpt \
  --init-use-ema
```

- Expected runtime:
  - about `10-15 min`
- Delta vs winning base:
  - `dataset.image_size: 320`
  - `train/scheduler epochs: 2`
  - low LR refresh
  - mixup/cutmix disabled
  - dense checkpoint retention around the early peak

## S0-R5

- Config: `configs/experiments/target099/stage0_s0_r5_wide_resnet50_2_optional.yaml`
- Status:
  - optional
  - legality-sensitive
- Only run if the homework interpretation clearly accepts `torchvision.models.wide_resnet50_2` as within the allowed ResNet family.
- Expected runtime:
  - about `70-100 min`
- Train command:

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --train-config configs/experiments/target099/stage0_s0_r5_wide_resnet50_2_optional.yaml \
  --output-dir outputs/target099_stage0
```

## Post-Train Validation Commands

After each serious Stage 0 run, validate the raw and EMA checkpoints so that `val_probs_with_ids.npz` exists for diversity analysis.

Raw checkpoint:

```bash
python scripts/validate.py \
  --config configs/config.yaml \
  --train-config <same_stage0_train_config> \
  --ckpt outputs/target099_stage0/<run_dir>/checkpoints/best_raw.ckpt \
  --output-dir outputs/validate_stage0/<run_name>_raw
```

EMA checkpoint:

```bash
python scripts/validate.py \
  --config configs/config.yaml \
  --train-config <same_stage0_train_config> \
  --ckpt outputs/target099_stage0/<run_dir>/checkpoints/best_ema.ckpt \
  --use-ema \
  --output-dir outputs/validate_stage0/<run_name>_ema
```

If a run passes Stage 0 and should enter diversity/ensemble comparison, save test probabilities too:

```bash
python scripts/infer.py \
  --config configs/config.yaml \
  --train-config <same_stage0_train_config> \
  --ckpt outputs/target099_stage0/<run_dir>/checkpoints/best_ema.ckpt \
  --use-ema \
  --tta \
  --save-probs \
  --output-dir outputs/infer_stage0/<run_name>_ema_tta
```

## Kill Criteria

- Kill a base probe if `val_acc < 88.5`.
- Kill a base probe if both `val_macro_recall` and `val_nll` are clearly worse than the CE anchor.
- Kill a FixRes refresh if it worsens `val_nll` and fails to preserve early useful checkpoints.
- Kill the optional WideResNet50-2 pilot immediately if legality is uncertain or if early metrics trail badly.

## Success Criteria

- A base probe is Stage-1-worthy if it stays within about `0.5` of the historical strong-v2 `val_acc` anchor and shows a better tail/calibration story through `val_macro_recall`, `val_nll`, or later rescue/diversity behavior.
- A short FixRes refresh is successful only if it improves or preserves the selected base without obvious calibration drift.

## Artifacts To Inspect Before Moving On

- `summary.json`
  - selected checkpoint metric
  - `best_metrics_selected`
- `run_metadata.json`
  - stage-gate result
  - checkpoint inventory
- `history.csv`
  - look for early useful epochs and drift after the peak
- `outputs/validate_stage0/*/validate_metrics.json`
- `outputs/validate_stage0/*/val_probs_with_ids.npz`
- retained `epoch_*.ckpt` files kept by `keep_top_k`
