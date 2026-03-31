## Why

The project has many runs and fixes across multiple docs/logs, but we still need a single report-ready source that maps progress to the grading rubric required by the course. Creating this now reduces report risk and ensures every final claim is traceable to concrete run artifacts.

## What Changes

- Add one consolidated documentation artifact in `docs/` that summarizes findings, model progression, and evidence-backed results.
- Add an explicit "Grading Policy - Report (15%)" section in that doc, including required report sections and additional-experiment expectations.
- Add a traceability format that links each reported claim to run artifacts (logs, summaries, validation outputs, submission outputs).
- Define a repeatable update workflow so this document can be refreshed after new experiments without changing code behavior.

## Capabilities

### New Capabilities
- `report-summary-doc`: Maintain a single consolidated progress-and-findings document that is directly reusable for the final PDF report drafting stage.

### Modified Capabilities
- `report-and-method-writeup`: Add explicit requirement that report-prep documentation includes the grading-policy checklist section and hypothesis/method/result implication framing for additional experiments.

## Impact

- Affected files are documentation-focused (`docs/` and OpenSpec change artifacts only).
- No training/inference behavior or dependencies are changed.
- Improves auditability and reduces mismatch between claimed method/results and saved artifacts.
