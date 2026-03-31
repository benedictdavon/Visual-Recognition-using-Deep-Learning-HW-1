# Tail-Aware Sampler

This repo now supports a configurable class-aware sampler for single-GPU training.

## Supported Sampler

### `weighted_random`
- Backend: `torch.utils.data.WeightedRandomSampler`
- Intended use: Stage 0 probe for long-tail balancing without changing the loss.

## Config

```yaml
sampler:
  name: weighted_random
  power: 0.5
  replacement: true
  num_samples: null
```

## Meaning Of `power`

- `0.0`: uniform sampling, equivalent to no class reweighting
- `0.5`: inverse-sqrt frequency weighting
- `1.0`: inverse-frequency weighting

The effective class weight is:

```text
weight(class) = 1 / count(class)^power
```

## Notes

- The sampler is wired directly into the training dataloader in `scripts/train.py`.
- It is only applied to the training loader; validation and inference stay unchanged.
- It is designed for the current single-GPU, non-distributed workflow.
- If a class has zero samples in the active training split, sampler construction raises an error instead of silently degrading.

## When To Use

- Use this probe when you want to test tail balancing at the data-loader level while keeping:
  - the backbone fixed
  - the augmentation stack fixed
  - the loss as plain CE

## Evaluation Guidance

- Compare against the plain-CE baseline using:
  - `val_acc`
  - `val_macro_recall`
  - `val_nll`
  - `val_ece`
- Prefer it only if it improves tail-sensitive metrics without clearly hurting overall calibration or stability.
