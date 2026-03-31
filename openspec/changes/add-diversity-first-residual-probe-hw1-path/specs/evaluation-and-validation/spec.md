## ADDED Requirements

### Requirement: Diversity probes require branch-specific validation evidence
When a diversity-first residual-family probe exposes both raw and EMA branches, the validation workflow SHALL preserve explicit branch-specific evidence before the candidate is promoted for ensemble consideration.

#### Scenario: Raw and EMA are evaluated separately
- **WHEN** a diversity-first probe run contains both raw and EMA weight branches
- **THEN** the repo SHALL support evaluating the raw branch and the EMA branch as distinct validation candidates
- **AND** candidate promotion SHALL not rely on an implicit assumption that one branch is automatically preferred

#### Scenario: Branch provenance remains explicit in validation artifacts
- **WHEN** validation artifacts are produced for a diversity-first probe branch
- **THEN** the branch identity SHALL remain explicit enough for later diversity and ensemble comparison

### Requirement: Ensemble-candidate promotion requires more than selected-epoch ambiguity
For diversity-first probes, candidate promotion SHALL use explicit branch evidence rather than relying only on the selected epoch represented by `best.ckpt`.

#### Scenario: Selected epoch does not replace branch choice
- **WHEN** a user considers a diversity-first probe for ensemble inclusion
- **THEN** the candidate decision SHALL use explicit raw or EMA branch evidence
- **AND** it SHALL not rely only on the fact that `best.ckpt` exists
