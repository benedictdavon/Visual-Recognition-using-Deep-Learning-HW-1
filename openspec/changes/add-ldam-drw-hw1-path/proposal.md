## Why

The repo has already explored three imbalance directions on the current `ResNet101` anchor: `logit_adjusted_ce`, `balanced_softmax`, and sampler-only rebalancing. What it has not tested is the LDAM-DRW family of late-stage reweighting, where representation learning stays natural early and class-balanced correction activates only after the model has learned a stronger shared feature space.

## What Changes

- Add an LDAM-based loss option for the current ResNet-family training pipeline.
- Add a deferred reweighting schedule so class-balanced weights can stay inactive early and turn on only in the later phase of training.
- Add a legal `ResNet101` HW1 experiment path that uses LDAM-DRW style late-stage reweighting without changing the backbone family.
- Add explicit config and fail-fast semantics for LDAM / DRW so the path cannot run with missing class-count inputs or ambiguous schedule boundaries.
- Add reproducibility requirements so runs record whether late-stage reweighting was active, when it activated, and what weighting regime was used.

## Capabilities

### New Capabilities
- `ldam-drw-late-reweighting`: A legal HW1 experiment path that keeps `ResNet101` as the backbone and adds LDAM loss with deferred late-stage class reweighting.

### Modified Capabilities
- `training-pipeline`: The training contract changes to support an LDAM loss option and an explicit deferred reweighting schedule.
- `experimentation-and-reproducibility`: The reproducibility contract changes so runs record the late-stage reweighting schedule, activation point, and weighting evidence needed for later adoption decisions.

## Impact

- Affected code is expected to center on `src/losses/losses.py`, `scripts/train.py`, training config parsing, and run metadata / summary persistence.
- New experiment configs will be needed under `configs/experiments/`.
- No new external dependency is required.
- The backbone, strict parameter budget, and no-external-data homework constraints remain unchanged.
