# ResNeXt-50 32x3d Experiment Path

This document defines the repo-supported path for the custom `resnext50_32x3d` backbone.

It is intentionally conservative:

- the model is a legal ResNet-family extension only when it is clearly documented as a ResNeXt-style grouped-bottleneck residual network
- the custom `32x3d` width is not a standard torchvision checkpoint name
- the path should be killed early if it is clearly weaker than the current `resnet101` anchor

## Legality Tier

- Tier: `Tier 2`
- Interpretation:
  - allowed with report-visible explanation and citation
  - still subject to the strict `< 100M` parameter limit
  - still prohibited from using external data
- Current parameter count:
  - about `17.88M` in the repo's 100-class classifier setup

## Architecture Definition

- Base family: ResNeXt / residual bottleneck network
- Depth profile: ResNet-50 style `[3, 4, 6, 3]`
- Cardinality: `32`
- Width per group: `3`
- Model name in config: `resnext50_32x3d`

This repo builds the model through the same `torchvision.models.resnet.ResNet` / `Bottleneck` path used by standard ResNet-family models.

## Initialization Path

`resnext50_32x3d` does not have a standard torchvision pretrained checkpoint.

When `model.pretrained: true`, this repo requires:

```yaml
model:
  name: resnext50_32x3d
  pretrained: true
  custom_pretrained_init: resnext50_32x4d_slice
```

Behavior:

- the repo loads torchvision `resnext50_32x4d` weights
- it transfers them into the narrower `32x3d` grouped bottleneck by shape-aware slicing
- the initialization source is recorded in model metadata and logs
- validation, inference, and checkpoint-warm-start training skip this preload path when a checkpoint is the true weight source

Failure behavior:

- if `custom_pretrained_init` is omitted while `pretrained: true`, model build fails
- if torchvision `resnext50_32x4d` weights are unavailable, model build fails with an explicit error instead of silently pretending pretrained initialization succeeded

Fallback:

- set `model.pretrained: false` for a lower-confidence random-init path
- or redirect to the now-supported standard `resnext50_32x4d` implementation if the custom path proves too brittle or too weak

## Configs

- Smoke: `configs/experiments/resnext50_32x3d_smoke.yaml`
- Short probe: `configs/experiments/resnext50_32x3d_short_probe.yaml`
- Full run: `configs/experiments/resnext50_32x3d_full.yaml`
- FixRes refresh: `configs/experiments/resnext50_32x3d_fixres320.yaml`

## Run Order

1. Smoke
2. Short probe
3. Full run only if the short probe passes
4. Short FixRes refresh only if the full base run is already competitive
5. Validation/test probability export
6. Diversity analysis and ensemble search against the current ResNet101 anchor

## Stop And Continue Criteria

### Smoke

- Continue if:
  - model build succeeds
  - the explicit custom initialization path is accepted
  - forward/backward/checkpoint paths are intact
- Stop if:
  - model build fails
  - pretrained warm-start is unavailable and you do not want a random-init custom run

### Short Probe

- Config: `configs/experiments/resnext50_32x3d_short_probe.yaml`
- Continue if one of these is true:
  - selected `val_acc` is roughly within `1.5` to `2.0` points of the current base ResNet101 anchor
  - selected `val_nll` is close enough to keep the model credible
  - the model shows meaningful diversity potential versus the current anchor
- Stop if:
  - `val_acc` is clearly below the gate in `run_metadata.json`
  - `val_nll` and `val_macro_recall` are both clearly worse than the anchor
  - the model overfits earlier than the current FixRes path without compensating gains

### Full Run

- Config: `configs/experiments/resnext50_32x3d_full.yaml`
- Continue to FixRes only if:
  - the model is competitive as a single model
  - or it is slightly weaker but shows clear disagreement/rescue value for ensembling
- Stop if:
  - it remains clearly below the current anchor with no diversity upside

### FixRes Refresh

- Config: `configs/experiments/resnext50_32x3d_fixres320.yaml`
- Run only on top of the best completed base checkpoint
- Kill the refresh if:
  - selected `val_nll` drifts badly upward
  - the useful early-epoch checkpoint is not better than the base run

## Exact Commands

Train smoke:

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --train-config configs/experiments/resnext50_32x3d_smoke.yaml \
  --output-dir outputs/resnext50_32x3d
```

Train short probe:

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --train-config configs/experiments/resnext50_32x3d_short_probe.yaml \
  --output-dir outputs/resnext50_32x3d
```

Train full run:

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --train-config configs/experiments/resnext50_32x3d_full.yaml \
  --output-dir outputs/resnext50_32x3d
```

Train short FixRes refresh:

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --train-config configs/experiments/resnext50_32x3d_fixres320.yaml \
  --output-dir outputs/resnext50_32x3d \
  --init-ckpt outputs/resnext50_32x3d/<winning_base_run_dir>/checkpoints/best.ckpt \
  --init-use-ema
```

Validate raw branch:

```bash
python scripts/validate.py \
  --config configs/config.yaml \
  --train-config configs/experiments/resnext50_32x3d_full.yaml \
  --ckpt outputs/resnext50_32x3d/<run_dir>/checkpoints/best_raw.ckpt \
  --output-dir outputs/validate_resnext50_32x3d/<run_name>_raw
```

Validate EMA branch:

```bash
python scripts/validate.py \
  --config configs/config.yaml \
  --train-config configs/experiments/resnext50_32x3d_full.yaml \
  --ckpt outputs/resnext50_32x3d/<run_dir>/checkpoints/best_ema.ckpt \
  --use-ema \
  --output-dir outputs/validate_resnext50_32x3d/<run_name>_ema
```

Infer raw branch test probabilities:

```bash
python scripts/infer.py \
  --config configs/config.yaml \
  --train-config configs/experiments/resnext50_32x3d_full.yaml \
  --ckpt outputs/resnext50_32x3d/<run_dir>/checkpoints/best_raw.ckpt \
  --tta \
  --save-probs \
  --output-dir outputs/infer_resnext50_32x3d/<run_name>_raw_tta
```

Infer EMA branch test probabilities:

```bash
python scripts/infer.py \
  --config configs/config.yaml \
  --train-config configs/experiments/resnext50_32x3d_full.yaml \
  --ckpt outputs/resnext50_32x3d/<run_dir>/checkpoints/best_ema.ckpt \
  --use-ema \
  --tta \
  --save-probs \
  --output-dir outputs/infer_resnext50_32x3d/<run_name>_ema_tta
```

Diversity report:

```bash
python scripts/diversity_report.py \
  --manifest outputs/diversity_manifest.yaml \
  --output-dir outputs/diversity_resnext50_32x3d
```

Ensemble search:

```bash
python scripts/ensemble_search.py \
  --manifest outputs/diversity_manifest.yaml \
  --anchor <current_resnet101_anchor_name> \
  --diversity-summary outputs/diversity_resnext50_32x3d/diversity_summary.csv \
  --output-dir outputs/ensemble_resnext50_32x3d
```

## What To Inspect

- `summary.json`
- `run_metadata.json`
- `model_metadata.json`
- `history.csv`
- `validate_metrics.json`
- `val_probs_with_ids.npz`
- `test_probs_with_ids.npz`
- `diversity_summary.csv`
- `ensemble_search_summary.json`

## Report Notes

Describe this model as:

- a custom ResNeXt-style grouped-bottleneck extension of ResNet
- a narrower `32x3d` residual-family variant derived from the ResNeXt design
- a legal ResNet-family extension under the homework interpretation because the residual lineage remains explicit

Do not describe it as:

- a standard torchvision pretrained backbone
- a completely different non-ResNet architecture

Minimum citations:

- Kaiming He et al., "Deep Residual Learning for Image Recognition"
- Saining Xie et al., "Aggregated Residual Transformations for Deep Neural Networks"
- Optional if used in the final recipe:
  - "Bag of Tricks for Image Classification with Convolutional Neural Networks"
  - "Fixing the Train-Test Resolution Discrepancy"

## Smoke Verification Status

Smoke verification completed during implementation:

- offline builder smoke with `pretrained: false`: expected to be runnable
- offline grouped-width transfer smoke from a randomly initialized `resnext50_32x4d` source: passed the tensor-shape transfer path
- pretrained warm-start through torchvision default weights: not fully executable in the current restricted shell because downloading weights is blocked; the repo now fails clearly in that case instead of silently misreporting initialization status
