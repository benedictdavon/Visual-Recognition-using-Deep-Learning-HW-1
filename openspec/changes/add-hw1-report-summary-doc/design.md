## Context

The repository already contains run artifacts (`outputs/*`) and strategy/progress notes (`docs/progress_summary.md`, `docs/visual_recognition_hw1_strategy.md`), but there is no single report-prep document that is structured exactly for the course grading rubric. This creates risk that final PDF writing drifts from actual evidence or misses mandatory sections.

The requested change is documentation-focused: add one consolidated report-summary document and define a required structure that is directly mappable to the grading policy (Introduction, Method, Results, References, and additional experiment framing).

## Goals / Non-Goals

**Goals:**
- Define one canonical consolidated report-prep doc in `docs/`.
- Require a dedicated "Grading Policy - Report (15%)" section with the exact course expectations.
- Require experiment writeups to include hypothesis, why it may/may not work, and implication of results.
- Require traceability from each key claim to concrete run artifacts (logs/metrics/submission outputs).

**Non-Goals:**
- Auto-generate PDF or citations.
- Change model training/inference behavior.
- Add external dependencies.

## Decisions

### Decision 1: Single canonical doc path in `docs/`
- Choice: Maintain one report-prep summary markdown file as the canonical source before writing the final PDF.
- Rationale: Minimizes drift and avoids conflicting summaries across multiple docs.
- Alternative considered: Keep multiple ad-hoc notes. Rejected because it increases inconsistency risk.

### Decision 2: Mandatory grading-policy section as a fixed heading
- Choice: Include a verbatim heading block titled `Grading Policy - Report (15%)` and map each required report part to project evidence.
- Rationale: Prevents omission of course-required items during final report conversion.
- Alternative considered: Checklist-only appendix. Rejected because rubric content should be first-class, not hidden.

### Decision 3: Evidence-linked experiment summaries
- Choice: Every major experiment summary must include: hypothesis, mechanism expectation, observed result, and implication.
- Rationale: Matches homework requirement that "additional experiments" are analytical, not only hyperparameter tuning.
- Alternative considered: leaderboard-only table. Rejected as insufficiently explanatory.

### Decision 4: OpenSpec capability split
- Choice: Add a new capability (`report-summary-doc`) plus a modified capability (`report-and-method-writeup`) with additive requirements.
- Rationale: Keeps spec intent clear: one capability for document existence/structure and one for rubric-aligned reporting behavior.

## Risks / Trade-offs

- [Risk] Manual summaries may become stale after new runs.
  - Mitigation: Define explicit update checklist in tasks and include artifact paths in the doc.
- [Risk] Overly long summary may be hard to maintain.
  - Mitigation: Enforce section structure and concise evidence tables.
- [Risk] Claims may exceed what artifacts prove.
  - Mitigation: Require run-path references for each key performance claim.

## Migration Plan

1. Create the consolidated doc in `docs/` using the required section structure.
2. Port current findings from existing docs and run artifacts.
3. Add grading-policy section and experiment-analysis subsection.
4. Keep older docs as historical references; treat the new doc as canonical report-prep source.

## Open Questions

- Should the consolidated doc be updated automatically by script later, or remain manually curated for quality?
- Should the final PDF template mapping be added as a separate appendix in this repo?
