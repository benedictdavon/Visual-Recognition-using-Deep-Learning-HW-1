## Context

The current repo has already explored several legal HW1 directions around `ResNet101`, tail-aware losses, sampler-level balancing, and short FixRes refreshes. The strongest local evidence still centers on `ResNet101` with `logit_adjusted_ce`, while sampler-only rebalancing underperformed and architecture probes such as `wide_resnet50_2` did not justify replacing the anchor.

The dataset profile matters:

- `100` classes with severe training imbalance
- tiny balanced official validation (`3` samples per class)
- repeated overlap among mistakes across otherwise different runs

That evidence suggests the base representation is useful, but the classifier remains biased toward training priors. A decoupled classifier-rebalance stage is a natural next step because it attacks the classifier directly without changing the legal backbone family or assuming the whole representation must be relearned.

## Goals / Non-Goals

**Goals:**

- Add a legal multi-stage `ResNet101` path that preserves the current `logit_adjusted_ce` base recipe and then runs a classifier-only rebalance stage from a saved checkpoint.
- Keep the default workflow single-GPU practical and config-driven.
- Make the rebalance stage explicit, reproducible, and fail-fast when its prerequisites are missing or ambiguous.
- Preserve the option to run a short FixRes refresh after the rebalance stage without making it automatic.
- Record enough run lineage to compare base, rebalance, and optional FixRes stages cleanly.

**Non-Goals:**

- Replace the current architecture legality tiers.
- Add non-ResNet backbones, external data, or distributed-only workflows.
- Guarantee a leaderboard jump from the staged recipe.
- Redesign validation, inference, or ensemble tooling beyond what is needed to support staged run evidence.

## Decisions

### Decision: Keep `ResNet101` and the current `logit_adjusted_ce` base as the anchor

The staged recipe SHALL begin from the strongest current local base rather than from a new backbone family or a sampler-led recipe.

Rationale:

- `logit_adjusted_ce` already outperformed the sampler-only probe in the repo.
- The current evidence points more toward classifier bias and correlated mistakes than plain under-capacity.
- Keeping the anchor fixed isolates the value of the rebalance stage.

Alternatives considered:

- `balanced_softmax` as the anchor: close, but weaker in current repo evidence.
- sampler-led base path: underperformed and is less aligned with a decoupled classifier hypothesis.
- a new architecture anchor: increases variables and weakens the causal readout.

### Decision: Implement rebalance as a classifier-only stage with the backbone frozen

The rebalance stage SHALL update only the classifier head while keeping the backbone frozen.

Rationale:

- This matches the cRT-style hypothesis that representation quality is mostly sufficient while the classifier remains biased by class frequency.
- It keeps the stage cheap and single-GPU practical.
- It reduces the chance of destabilizing the already strong base representation.

Alternatives considered:

- full end-to-end fine-tuning with class reweighting: more flexible, but less targeted and harder to attribute.
- sampler-only late-stage rebalance: easier to add, but the earlier Stage 0 evidence makes it a weaker bet.
- architectural changes: potentially useful for diversity, but not the primary hypothesis for this change.

### Decision: Require an explicit parent checkpoint for the rebalance stage

The rebalance stage SHALL not run from random initialization or ambiguous state. It must start from a valid saved base checkpoint.

Rationale:

- The value of the stage depends on decoupling, not on retraining from scratch.
- Fail-fast semantics protect experiment discipline and make lineage easier to reason about.

Alternatives considered:

- silently falling back to ordinary training: rejected because it hides stage misconfiguration.
- allowing a no-checkpoint rebalance stage: rejected because it violates the design intent.

### Decision: Treat the optional short FixRes refresh as a separate continuation stage

The FixRes refresh SHALL remain optional and explicitly later than the classifier rebalance stage.

Rationale:

- The repo already shows that short FixRes refresh can improve NLL while also being fragile.
- Keeping it separate prevents the change from collapsing into a monolithic recipe with hard-to-interpret gains.

Alternatives considered:

- bake FixRes directly into the rebalance stage: rejected because it couples two distinct hypotheses.
- omit FixRes entirely: rejected because it is still one of the few proven useful knobs in the repo.

### Decision: Persist multi-stage lineage in run metadata and summaries

Runs produced by the staged path SHALL record their parent checkpoint / run, stage identity, and active trainable scope.

Rationale:

- Multi-stage recipes are hard to compare if ancestry is implicit.
- The official validation set is tiny, so adoption decisions need stronger artifact discipline, not weaker.

Alternatives considered:

- infer lineage only from directory names: rejected because it is brittle.
- document lineage only in prose: rejected because machine-readable evidence matters for later automation.

## Risks / Trade-offs

- [Tiny official validation can overrank a lucky rebalance stage] -> Mitigation: require lineage-rich artifacts and compare base vs rebalance vs optional FixRes as a bundle, not by one metric alone.
- [Classifier-only rebalance may help tail classes while hurting head-class calibration] -> Mitigation: keep `val_acc`, `val_macro_recall`, `val_nll`, and `val_ece` visible for each stage.
- [Freezing logic can become ambiguous or silently ineffective] -> Mitigation: make rebalance-stage checkpoint prerequisites and active trainable scope explicit and fail fast.
- [Optional FixRes can blur attribution] -> Mitigation: keep it as a distinct continuation stage with separate configs and outputs.

## Migration Plan

1. Add the staged-training controls and explicit warm-start semantics.
2. Add configs for the base-to-rebalance path and for the optional short FixRes continuation.
3. Persist stage lineage in run artifacts.
4. Validate that ordinary single-stage runs remain unchanged.

Rollback strategy:

- keep the new staged path opt-in
- if the rebalance stage proves weak or brittle, the repo can stop using the new configs without disturbing the current baseline path

## Open Questions

- Whether the rebalance stage should use plain CE, logit-adjusted CE, or another head-only objective is still an experimental choice rather than a fixed design promise.
- Whether the optional FixRes stage should warm-start from the rebalance-selected checkpoint or the best raw / EMA branch is intentionally left open for implementation-time evidence.
