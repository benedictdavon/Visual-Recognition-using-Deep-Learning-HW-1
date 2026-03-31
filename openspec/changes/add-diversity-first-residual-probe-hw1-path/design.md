## Context

The repo already supports several legal Tier 2 residual-family options, including `ResNeXt101` variants and `ResNet101` with SE-style modifications. The practical issue is not lack of legal residual variants; it is that most candidate models so far have been too correlated with the current anchor to move the leaderboard. The `resnext50_32x3d` path also exposed another important detail: the raw branch can remain viable while the EMA branch collapses, so “EMA by default” is not a safe assumption for diversity probes.

This change therefore focuses on experiment discipline rather than architecture proliferation. One diversity-first probe is enough if the repo evaluates it in a way that can detect useful disagreement and rescue behavior.

## Goals / Non-Goals

**Goals:**

- Define a legal path for one already-supported Tier 2 residual-family probe to compete for ensemble inclusion rather than anchor status.
- Require branch-specific raw and EMA validation evidence before the probe is promoted.
- Require diversity evidence against the current `ResNet101` anchor as part of continuation and adoption decisions.
- Keep the workflow single-GPU practical and compatible with current validation, inference, and ensemble tooling.

**Non-Goals:**

- Add a new backbone family or stretch the homework legality boundary.
- Guarantee that the probe becomes a better single model than the current anchor.
- Replace the current anchor purely because the probe is newer or larger.
- Expand the proposal into a many-model search program.

## Decisions

### Decision: Use one already-supported Tier 2 residual-family candidate

The path assumes one carefully chosen candidate from the repo's existing legal Tier 2 residual-family space, such as:

- a `ResNeXt101` family branch already supported by the builder
- a `ResNet101` branch that uses supported SE-style modifications

Rationale:

- The repo already has the legal building blocks.
- The real hypothesis is about error decorrelation, not about adding another model registry entry.
- Keeping the probe inside existing support lowers implementation risk.

Alternatives considered:

- add a new custom residual model first: rejected because it adds architecture uncertainty before the diversity thesis is tested.
- test many candidates at once: rejected because it dilutes the evidence and spend.

### Decision: Treat raw and EMA as separate candidate branches

When a diversity probe produces both raw and EMA branches, both branches SHALL be evaluated explicitly rather than assuming the selected branch for ordinary single-model use is automatically the right ensemble candidate.

Rationale:

- The repo already has evidence that raw can remain useful when EMA is not.
- Diversity-first selection cares about rescue behavior and disagreement, not just one branch's local top-1 score.

Alternatives considered:

- always prefer EMA: rejected because it can hide a stronger diversity candidate.
- always prefer raw: rejected because EMA is still often useful and must be measured, not guessed.

### Decision: Continue the probe only if diversity evidence is real

The continuation gate for the probe is diversity-first:

- acceptable single-model local metric gap to the anchor
- meaningful disagreement and rescue behavior
- explicit branch provenance

Rationale:

- A slightly weaker single model can still be a valuable ensemble candidate.
- The plateau diagnosis points directly at correlation as the bottleneck.

Alternatives considered:

- rank purely by single-model `val_acc`: rejected because it misses the point of the probe.
- accept any candidate with disagreement only: rejected because random disagreement without rescue value is not enough.

### Decision: Keep the probe out of anchor promotion by default

The probe SHALL be framed as an ensemble-candidate path unless later evidence shows that it also deserves anchor consideration.

Rationale:

- This preserves decision discipline.
- It matches the actual stated reason for trying the architecture.

Alternatives considered:

- let the probe compete as the new default immediately: rejected because that would overstate the evidence.

## Risks / Trade-offs

- [The chosen candidate may still be too correlated with the anchor] → Mitigation: make disagreement and rescue evidence explicit before continuing.
- [Raw and EMA branch handling may become ambiguous in manifests and reports] → Mitigation: require branch-labeled provenance and distinct candidate naming.
- [A weak single-model result may tempt premature rejection of a useful ensemble branch] → Mitigation: require diversity-aware continuation gates, not only top-1 ranking.
- [The probe may consume time without changing the final ensemble] → Mitigation: keep the path narrow, one candidate at a time, with hard stop rules.

## Migration Plan

1. Add branch-aware experiment scaffolding for one already-supported Tier 2 residual-family probe.
2. Add validation / inference conventions or manifests that treat raw and EMA as distinct candidate branches where appropriate.
3. Add diversity-first continuation and adoption guidance tied to the existing diversity tooling.
4. Keep the current anchor path unchanged unless later evidence justifies broader use.

Rollback strategy:

- keep the probe opt-in
- if diversity evidence is weak, stop using the candidate without affecting the current baseline path

## Open Questions

- Whether the first candidate should be an existing `ResNeXt101` branch or an `SE-ResNet101` branch is intentionally left open to implementation-time judgment.
- Whether raw and EMA should always both be exported, or only when both branches are locally credible, is a workflow decision to settle during implementation.
