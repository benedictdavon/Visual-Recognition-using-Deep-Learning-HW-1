## ADDED Requirements

### Requirement: LDAM-DRW experiment path
The system SHALL support a legal HW1 experiment path in which `ResNet101` remains the backbone and imbalance handling is introduced through LDAM loss plus deferred late-stage reweighting.

#### Scenario: LDAM-DRW path keeps the backbone unchanged
- **WHEN** a user runs the LDAM-DRW experiment path
- **THEN** the backbone SHALL remain within the existing legal ResNet-family path
- **AND** the change SHALL be treated as a training-strategy experiment rather than an architecture change

#### Scenario: LDAM-DRW path is optional
- **WHEN** a user runs an ordinary non-LDAM recipe
- **THEN** the repo SHALL not implicitly activate the LDAM-DRW path

### Requirement: Deferred late-stage reweighting remains explicit
The LDAM-DRW experiment path SHALL expose the late-stage reweighting boundary explicitly so users can tell when class-balanced reweighting becomes active.

#### Scenario: Reweighting activates only after the configured boundary
- **WHEN** a user configures an LDAM-DRW run with a deferred reweighting boundary
- **THEN** class-balanced reweighting SHALL remain inactive before that boundary
- **AND** it SHALL become active only after the configured late-stage point

#### Scenario: Schedule configuration is required to be interpretable
- **WHEN** a user inspects an LDAM-DRW run after completion
- **THEN** the run artifacts SHALL make the deferred reweighting behavior interpretable from saved metadata
