## ADDED Requirements

### Requirement: Diversity-first probe evidence bundle is branch-aware
The reproducibility contract SHALL require diversity-first residual probes to preserve branch-aware evidence for any raw or EMA branch considered for ensemble use.

#### Scenario: Branch-aware evidence is recorded for candidate promotion
- **WHEN** a diversity-first probe branch is promoted as an ensemble candidate
- **THEN** the evidence bundle SHALL include branch-specific validation artifacts
- **AND** it SHALL identify whether the promoted branch is raw or EMA

#### Scenario: Probe adoption requires diversity artifacts
- **WHEN** a diversity-first residual-family probe is adopted for later homework-facing ensemble work
- **THEN** the evidence bundle SHALL include the relevant diversity artifact or manifest used to justify that adoption

### Requirement: Diversity-first probe remains distinguishable from anchor replacement
The reproducibility record SHALL make it clear whether the probe is being kept as an ensemble candidate or being argued as a stronger new anchor.

#### Scenario: Ensemble-candidate framing is preserved
- **WHEN** a diversity-first probe is documented for future use
- **THEN** the saved evidence SHALL allow a reviewer to tell that the path was justified for ensemble inclusion rather than automatic anchor replacement
