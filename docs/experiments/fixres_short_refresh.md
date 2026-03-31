# Short FixRes Refresh

The repo already supported the mechanics needed for FixRes-style refresh. This phase formalizes the exact short-refresh workflow required by the target099 plan.

## Supported Workflow

The short refresh now has explicit config support for:
- image-size override via `dataset.image_size`
- short duration via `train.epochs` and `scheduler.epochs`
- reduced LR via `optimizer.lr`
- mixup/cutmix disable via `mixup_cutmix.enabled: false`
- warm start from an existing checkpoint via:
  - `train.init_checkpoint`
  - `train.init_use_ema`
  - or CLI overrides `--init-ckpt` and `--init-use-ema`

## Template

- Config template: `configs/experiments/target099_fixres320_short_refresh_template.yaml`

This template is intentionally short, conservative, and aligned with the handoff’s diagnosis that useful FixRes gains appear early and overfit rapidly.

## Recommended Usage

1. Pick the winning Stage 0 base recipe.
2. Point the short-refresh run at that checkpoint.
3. Keep the refresh short.
4. Inspect `val_nll`, `val_ece`, and retained early checkpoints before extending anything.

Example command:

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --train-config configs/experiments/target099_fixres320_short_refresh_template.yaml \
  --init-ckpt outputs/<winning_stage0_run>/checkpoints/best.ckpt \
  --init-use-ema
```

## Notes

- If the winning Stage 0 base uses `balanced_softmax`, update the `loss` block in the template to match before running.
- If the winning Stage 0 base uses `logit_adjusted_ce`, keep or retune `loss.logit_adjusted_tau` as needed.
- The template defaults to dense checkpoint retention so early useful epochs are preserved for later ranking or ensemble work.
