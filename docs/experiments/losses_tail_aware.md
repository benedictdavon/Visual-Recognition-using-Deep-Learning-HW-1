# Tail-Aware Losses

This repo now supports two train-prior-aware loss probes for the long-tail diagnosis documented in `docs/handoff_trial_summary_target099.md`.

## Supported Loss Names

### `balanced_softmax`
- Use when you want a train-prior-aware alternative to plain CE that directly accounts for class frequency inside the softmax normalization.
- Best fit in this repo: low-cost Stage 0 probe on top of the strongest `resnet101_strong_v2_fast` recipe.

### `logit_adjusted_ce`
- Use when you want a simpler train-prior logit-shift variant with configurable strength.
- Best fit in this repo: low-cost Stage 0 probe where you want to tune the prior effect with `tau`.

## Required Config Fields

### Balanced Softmax
```yaml
loss:
  name: balanced_softmax
  label_smoothing: 0.05
```

### Logit-Adjusted CE
```yaml
loss:
  name: logit_adjusted_ce
  label_smoothing: 0.05
  logit_adjusted_tau: 1.0
```

Notes:
- Both losses derive train priors from the actual training dataframe at runtime.
- There is no silent fallback to plain CE if class priors are unavailable or invalid.
- Plain CE remains unchanged:

```yaml
loss:
  name: cross_entropy
  label_smoothing: 0.05
```

## Expected Long-Tail Effect

- `balanced_softmax`:
  - can help when the plain CE baseline over-favors head classes under strong imbalance
  - should be compared against plain CE using both `val_macro_recall` and `val_nll`, not accuracy alone

- `logit_adjusted_ce`:
  - can provide a milder or more tunable prior-aware correction than switching to a different normalization objective
  - `logit_adjusted_tau` controls how strongly train priors affect the logits

## Evaluation Guidance

- Prefer comparing:
  - `val_acc`
  - `val_macro_recall`
  - `val_nll`
  - `val_ece`
- If a tail-aware loss slightly changes `val_acc` but meaningfully improves `val_macro_recall` and/or `val_nll`, keep it in consideration for Stage 1.
- If it hurts both calibration and recall, kill it early.
