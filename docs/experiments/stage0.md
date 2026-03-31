# Stage 0: Low-Cost Probes

Goal: test whether tail-aware objectives, tail-aware sampling, and a short FixRes refresh can break the current high-correlation plateau with minimal GPU cost.

## Planned Runs

### S0-R1
- Recipe: `ResNet101 strong-v2 + logit-adjusted CE`
- Purpose: test whether train-prior logit adjustment improves tail handling without changing the base architecture or augmentation stack.

### S0-R2
- Recipe: `ResNet101 strong-v2 + Balanced Softmax`
- Purpose: test a train-prior-aware alternative to plain CE for the documented train imbalance.

### S0-R3
- Recipe: `ResNet101 strong-v2 + weighted random sampler (inverse-sqrt) + plain CE`
- Purpose: isolate sampler-driven tail balancing from loss-driven tail balancing.

### S0-R4
- Recipe: `short low-LR FixRes refresh` on the best Stage 0 base result
- Purpose: preserve the handoff’s observed FixRes benefit while constraining overfit with a short refresh.

### S0-R5
- Recipe: `WideResNet50-2 pilot`
- Purpose: optional diversity probe only if the repo/legal framing supports it.
- Status: legality-sensitive and must remain explicitly optional.

## Success Criteria
- At least one Stage 0 base probe beats the current plain-CE anchor on at least one of:
  - `val_acc`
  - `val_macro_recall`
  - `val_nll`
- The best probe yields a plausible candidate for the short FixRes refresh.
- Dense checkpoint retention captures multiple useful early checkpoints for later diversity or snapshot analysis.

## Kill Criteria
- Stop a probe if it underperforms the plain-CE anchor early and also shows clearly worse `val_nll` or `val_macro_recall`.
- Stop a probe if it destabilizes optimization or causes obvious calibration collapse.
- Do not expand Stage 1 until Stage 0 produces a winning base recipe and a documented go/no-go decision.

## Required Artifacts
- run directory with `config.yaml`, `summary.json`, `run_metadata.json`, `history.csv`, and checkpoint inventory
- validation metrics including `val_acc`, `val_macro_recall`, `val_nll`, `val_ece`
- if advanced to ensemble analysis: val/test probability artifacts and diversity summary
