## ADDED Requirements

### Requirement: Consolidated report-prep document SHALL exist
The repository SHALL maintain one consolidated markdown document in `docs/` that summarizes findings, experiment progression, and current best results for HW1 report preparation.

#### Scenario: Canonical summary document is present
- **WHEN** a reviewer inspects the repository docs
- **THEN** exactly one canonical report-summary markdown file is available and contains project progress context, not only command snippets

### Requirement: Grading policy section SHALL be explicitly included
The consolidated report-prep document SHALL include a section titled `Grading Policy - Report (15%)` that captures required report structure and penalties/expectations from the course instructions.

#### Scenario: Required grading-policy content is visible
- **WHEN** a student prepares the final PDF from the summary document
- **THEN** the document explicitly lists required report sections (Introduction, Method, Results, References) and additional-experiment expectations

### Requirement: Experiment writeups SHALL include scientific framing
For each major experiment branch, the consolidated report-prep document SHALL include hypothesis, why it may or may not work, measured result, and implication.

#### Scenario: Additional experiment is defensible
- **WHEN** an evaluator checks whether an additional experiment is more than hyperparameter tuning
- **THEN** the writeup includes hypothesis/mechanism/result/implication framing tied to concrete run outcomes

### Requirement: Key claims SHALL be traceable to artifacts
The consolidated report-prep document SHALL link key performance claims to concrete artifact paths under `outputs/` (for example `train.log`, `summary.json`, validation outputs, or submission outputs).

#### Scenario: Claim-to-artifact traceability
- **WHEN** a reader audits a reported score or conclusion
- **THEN** the reader can locate corresponding run artifacts from paths listed in the document
