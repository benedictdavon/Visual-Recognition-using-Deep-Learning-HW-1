## Context

The current repo already treats long-tail handling as a first-class problem. Existing probes cover train-prior-aware objectives (`logit_adjusted_ce`, `balanced_softmax`) and sampler-level balancing (`weighted_random`), but they do not cover the LDAM-DRW family introduced as a strong late-stage reweighting baseline for imbalanced recognition.

This matters because the dataset profile is mixed:

- strongly long-tailed training distribution
- tiny balanced official validation
- hard classes that are not purely tail classes

That combination weakens the case for aggressive whole-run balancing and strengthens the case for learning the representation naturally first, then applying a more targeted imbalance correction later.

## Goals / Non-Goals

**Goals:**

- Add an LDAM loss option that remains compatible with the current single-GPU, config-driven training pipeline.
- Add a deferred reweighting schedule that activates class-balanced weighting only after a configurable late-stage boundary.
- Provide a legal `ResNet101` experiment path for HW1 that isolates this imbalance hypothesis from architecture changes.
- Persist enough metadata to compare no-DRW, DRW, and possible follow-up stages cleanly.

**Non-Goals:**

- Change the repo's backbone legality rules or add new architecture families.
- Replace the current best recipe by policy rather than evidence.
- Introduce distributed-only training or external dependencies.
- Guarantee that LDAM-DRW will beat the current `logit_adjusted_ce` path.

## Decisions

### Decision: Treat LDAM-DRW as a training-strategy probe on top of `ResNet101`

The proposal keeps the backbone fixed and frames LDAM-DRW as a training-strategy change, not an architectural one.

Rationale:

- The missing experiment family is about imbalance optimization, not unused parameter budget.
- Fixing the backbone keeps the causal readout cleaner against the current best path.
- It stays in the safest legal part of the homework rules.

Alternatives considered:

- Pair LDAM-DRW with a new backbone immediately: rejected because it would blur the hypothesis.
- Apply the method first to a wider / grouped residual variant: rejected for the same reason.

### Decision: Add LDAM as an explicit new loss option rather than approximating it with existing losses

The design adds LDAM directly instead of trying to emulate it with current weighting knobs.

Rationale:

- The repo does not currently support margin-based imbalance correction.
- The paper-backed hypothesis is specifically about label-distribution-aware margins plus late reweighting.
- Reusing only existing CE-family losses would not actually test the intended family.

Alternatives considered:

- plain CE plus late class weights: simpler, but not a real LDAM-DRW path.
- balanced softmax with a schedule: closer to existing probes than to the missing experiment family.

### Decision: Make DRW activation epoch explicit and fail-fast

The schedule boundary where deferred reweighting activates SHALL be a clear config field, not an implicit convention.

Rationale:

- Tiny official validation already makes late-stage behavior hard to rank; hidden schedule behavior would make it worse.
- Explicit schedule boundaries improve reproducibility and ablation clarity.

Alternatives considered:

- hard-code a paper-style midpoint: rejected because the repo should stay config-driven.
- infer the boundary from total epochs silently: rejected because it weakens auditability.

### Decision: Keep ordinary training behavior unchanged when LDAM-DRW is not requested

The new path must be opt-in and must not disturb the current strong baseline behavior.

Rationale:

- The repo already has a working anchor.
- This change is an experiment family, not a mandatory migration.

Alternatives considered:

- repurpose existing class-weight fields automatically: rejected because it risks silently changing current runs.

## Risks / Trade-offs

- [LDAM-DRW may improve tail sensitivity while hurting head-class calibration] → Mitigation: keep `val_acc`, `val_macro_recall`, `val_nll`, and `val_ece` visible for all LDAM-DRW runs.
- [The tiny official validation set may overstate or understate the late-stage benefit] → Mitigation: require stage-aware metadata and compare against the strongest current anchor rather than a weak baseline.
- [Implementing margin-based loss incorrectly would create a misleading ablation] → Mitigation: make class-count requirements explicit and fail fast on invalid inputs.
- [Deferred reweighting may become an ambiguous hidden schedule] → Mitigation: persist the activation epoch and weighting regime in run artifacts.

## Migration Plan

1. Add the LDAM loss option and the deferred reweighting controls.
2. Add one or more `ResNet101` experiment configs that exercise the new path.
3. Persist schedule metadata in run artifacts.
4. Validate that existing non-LDAM recipes remain unchanged.

Rollback strategy:

- keep LDAM-DRW fully opt-in
- if the path proves weak or brittle, stop using the new configs without affecting the current baseline

## Open Questions

- Whether the late-stage weighting should reuse existing class-count-derived weights exactly or introduce a separate DRW-specific weighting strength remains an implementation choice.
- Whether the strongest practical comparison should be against plain CE or against the current `logit_adjusted_ce` anchor is an experiment-planning decision rather than a design guarantee.
