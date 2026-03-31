## Why

The current repo evidence says the best legal single-backbone path is still `ResNet101` with a `logit_adjusted_ce` base recipe, but the project remains stuck near the current leaderboard plateau with many shared mistakes across runs. That pattern suggests the representation is mostly strong enough, while the classifier still carries train-prior bias from the strongly imbalanced dataset and needs a more targeted long-tail correction.

## What Changes

- Add a staged HW1 recipe that keeps `ResNet101` as the backbone, trains the base model with the current `logit_adjusted_ce` path, and then runs a classifier-only rebalance stage from the saved base checkpoint.
- Add training-pipeline support for stage-specific freezing so the rebalance stage can keep the backbone frozen while updating only the classifier head.
- Add explicit warm-start and failure semantics for the rebalance stage so it cannot run ambiguously without a valid parent checkpoint.
- Add optional short FixRes refresh guidance after the rebalance stage as a separate continuation step, not an automatic default.
- Add reproducibility requirements for multi-stage recipe lineage so adoption decisions can compare base, rebalance, and optional FixRes stages with clear evidence.

## Capabilities

### New Capabilities
- `decoupled-classifier-rebalance`: A legal multi-stage ResNet101 training path that learns the base representation first, then rebalances only the classifier head before an optional short FixRes refresh.

### Modified Capabilities
- `training-pipeline`: The training contract changes to support stage-specific parameter freezing, classifier-only optimization, and fail-fast rebalance-stage warm-start rules.
- `experimentation-and-reproducibility`: The reproducibility contract changes so multi-stage recipes record parent-stage lineage, active trainable scope, and adoption evidence for each stage.

## Impact

- Affected code is expected to center on `scripts/train.py`, `src/engine/trainer.py`, optimizer / parameter-group setup, staged experiment configs under `configs/experiments/`, and run metadata / summary persistence.
- Validation and inference entrypoints should remain compatible with ordinary checkpoints, but staged runs will need clearer lineage and branch-selection evidence.
- No new external dependency is required.
- The backbone, parameter budget, and no-external-data homework guardrails remain unchanged.
