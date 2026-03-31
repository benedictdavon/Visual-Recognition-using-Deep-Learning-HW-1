## ADDED Requirements

### Requirement: Branch-labeled candidates are valid search inputs
The ensemble and diversity workflow SHALL allow raw and EMA branches from the same diversity-first probe to be treated as distinct candidates only when their branch provenance is explicit.

#### Scenario: Raw and EMA branches are included as separate candidates
- **WHEN** a diversity-first probe contributes both raw and EMA probability artifacts to a candidate pool
- **THEN** the candidate identities SHALL distinguish the raw branch from the EMA branch
- **AND** the pool SHALL not treat them as one unlabeled candidate

#### Scenario: Unlabeled branch provenance is invalid for candidate promotion
- **WHEN** a candidate artifact from a diversity-first probe lacks explicit raw or EMA branch provenance
- **THEN** that artifact SHALL not be considered a fully reproducible ensemble candidate

### Requirement: Diversity-first candidate admission prioritizes rescue-aware evidence
For diversity-first residual probes, candidate admission to later ensemble work SHALL prioritize rescue-aware diversity evidence rather than only single-model validation rank.

#### Scenario: Candidate admission uses diversity evidence
- **WHEN** a diversity-first probe is compared against the current anchor
- **THEN** disagreement and rescue-style evidence SHALL be part of the candidate-admission decision

#### Scenario: Weak diversity evidence blocks admission
- **WHEN** a diversity-first probe fails to show useful rescue or disagreement value
- **THEN** it SHALL not be promoted merely because it is a legal residual-family variant
