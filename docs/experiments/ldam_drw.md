# LDAM-DRW Experiment Path

This path keeps the legal `ResNet101` backbone fixed and changes only the imbalance strategy:

- use LDAM as the training loss
- keep ordinary training behavior in the early phase
- activate class-balanced reweighting only after an explicit late-stage boundary

The point is to test the missing imbalance family cleanly against the current anchors, not to introduce a new architecture variable.

## Configs

- Full experiment: `configs/experiments/target099/ldam_drw/resnet101_ldam_drw_full.yaml`
- Smoke verification: `configs/experiments/target099/ldam_drw/resnet101_ldam_drw_smoke.yaml`
- Invalid schedule check: `configs/experiments/target099/ldam_drw/resnet101_ldam_drw_invalid_schedule.yaml`

## Training Contract

- `loss.name: ldam` is now a first-class option
- LDAM requires valid per-class counts and fails before training if they are missing or invalid
- `loss.deferred_reweighting.enabled: true` is supported only with LDAM
- `loss.deferred_reweighting.start_epoch` is explicit and must fall in `[2, total_epochs]`
- before `start_epoch`, the loss runs without class-balanced reweighting
- at and after `start_epoch`, inverse-frequency class weights activate for the remainder of the run
- ordinary non-LDAM recipes keep their current behavior when deferred reweighting is not requested

## Recommended Commands

Full LDAM-DRW run:

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --train-config configs/experiments/target099/ldam_drw/resnet101_ldam_drw_full.yaml \
  --output-dir outputs/target099_ldam_drw
```

Smoke verification:

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --train-config configs/experiments/target099/ldam_drw/resnet101_ldam_drw_smoke.yaml \
  --output-dir outputs/target099_ldam_drw_verification
```

Validate a completed checkpoint:

```bash
python scripts/validate.py \
  --config configs/config.yaml \
  --train-config configs/experiments/target099/ldam_drw/resnet101_ldam_drw_full.yaml \
  --ckpt outputs/target099_ldam_drw/<run_dir>/checkpoints/best.ckpt \
  --output-dir outputs/validate_target099_ldam_drw
```

## Comparison Discipline

Compare LDAM-DRW directly against the current non-LDAM anchors:

- `target099_s0_r1_resnet101_logit_adjusted`
- `target099_s0_r2_resnet101_balanced_softmax`
- `target099_s0_r3_resnet101_sampler_inv_sqrt`

Do not promote the path based on a single metric. Inspect:

- `val_acc`
- `val_macro_recall`
- `val_nll`
- `val_ece`
- `history.csv` phase columns:
  - `loss_drw_enabled`
  - `loss_drw_active`
  - `loss_drw_start_epoch`
  - `loss_class_weight_source`

## Keep / Kill Criteria

Keep the path if one of the following is true:

- it improves `val_nll` materially while keeping `val_acc` within a small noise band of the current anchor
- it improves `val_macro_recall` without a clearly unacceptable `val_acc` collapse
- it underperforms as a standalone model but shows distinct enough behavior to matter for ensembling

Kill the path if both of the following hold:

- it fails to beat the current anchor on `val_nll` and `val_macro_recall`
- the late-stage boundary does not produce a defensible tradeoff in the recorded run artifacts

## Verification Notes

Healthy LDAM-DRW evidence should show all of the following:

- `summary.json` identifies `loss_runtime.loss_name: ldam`
- `run_metadata.json` includes `loss_runtime.deferred_reweighting`
- `history.csv` shows `loss_drw_active=false` before the boundary and `true` after it
- invalid schedule configs fail before training proceeds
- invalid class-count inputs fail in the loss builder instead of silently falling back to CE-family behavior

## Verification Evidence

Compatibility check for ordinary non-LDAM recipes:

- direct builder check still returns:
  - `CrossEntropyLoss`
  - `LogitAdjustedCrossEntropyLoss`

Positive smoke run:

- `outputs/target099_ldam_drw_verification/20260330_145555_target099_ldam_drw_smoke`
- observed summary metrics:
  - `val_acc`: `84.00`
  - `val_macro_recall`: `84.00`
  - `val_nll`: `3.2611`
  - `val_ece`: `0.7907`
- observed metadata:
  - `summary.json` and `run_metadata.json` both record `loss_name: ldam`
  - deferred reweighting is enabled with `start_epoch: 2`
  - active class weights are recorded once the late-stage boundary is reached
- observed phase flip in `history.csv`:
  - epoch 1: `loss_drw_active=False`, `loss_class_weight_source=none`
  - epoch 2: `loss_drw_active=True`, `loss_class_weight_source=deferred_reweighting`

Negative checks:

- invalid schedule config:
  - command failed before optimization with:
    - `ValueError: loss.deferred_reweighting.start_epoch must fall in [2, total_epochs] for LDAM-DRW; got start_epoch=3, total_epochs=2.`
- invalid class-count input:
  - direct builder call failed with:
    - `ValueError: class_counts length 2 does not match num_classes 3.`

## Full Run Outcome

The full `target099` run was completed at:

- `outputs/target099_ldam_drw/20260330_151455_target099_ldam_drw_resnet101`

Observed summary:

- selected checkpoint:
  - source: `ema`
  - metric: `val_nll`
  - `val_acc = 89.00`
  - `val_macro_recall = 89.00`
  - `val_nll = 3.3001`
  - `val_ece = 0.8478`
- best raw checkpoint:
  - `val_acc = 90.33`
  - `val_macro_recall = 90.33`
  - `val_nll = 3.5495`
  - `val_ece = 0.8714`

Interpretation:

- the implementation is functioning as intended
- the late reweighting boundary is visible and reproducible
- the actual model quality is not competitive with the current anchors
- this is not a calibration win either, because `val_nll` and `val_ece` are both dramatically worse than the stronger ResNet baselines

Verdict:

- keep the code path
- kill the experiment path for further `target099` submission work unless a future ensemble-specific hypothesis appears
