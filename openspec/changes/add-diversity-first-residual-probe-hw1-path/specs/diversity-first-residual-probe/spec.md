## ADDED Requirements

### Requirement: Diversity-first residual-family probe path
The system SHALL support a legal experiment path for one already-supported Tier 2 residual-family candidate whose continuation is justified primarily by diversity evidence against the current `ResNet101` anchor rather than by single-model ranking alone.

#### Scenario: Probe is treated as an ensemble candidate first
- **WHEN** a user runs the diversity-first residual-family probe path
- **THEN** the candidate SHALL be evaluated first as a potential ensemble partner for the current anchor
- **AND** it SHALL not be treated as the new default anchor without later evidence

#### Scenario: Probe remains inside the legal residual-family envelope
- **WHEN** a user selects the candidate for the diversity-first probe
- **THEN** the candidate SHALL come from the repo's already-supported legal Tier 2 residual-family space
- **AND** the change SHALL not require a non-ResNet backbone

### Requirement: Probe continuation depends on meaningful diversity
The diversity-first residual-family probe SHALL continue only when its branch-specific evidence shows meaningful disagreement or rescue value against the current anchor without unacceptable local-metric collapse.

#### Scenario: Slightly weaker single model can still continue
- **WHEN** the probe trails the anchor modestly on local single-model metrics
- **THEN** the probe MAY still continue if its diversity evidence shows meaningful rescue or disagreement value

#### Scenario: Probe is stopped when diversity value is absent
- **WHEN** the probe remains weak on local metrics and also fails to show meaningful diversity value against the anchor
- **THEN** the probe SHALL be stopped rather than expanded
