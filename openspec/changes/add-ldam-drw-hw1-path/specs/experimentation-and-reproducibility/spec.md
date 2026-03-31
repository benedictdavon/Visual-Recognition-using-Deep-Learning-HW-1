## ADDED Requirements

### Requirement: LDAM-DRW schedule metadata is persisted
Runs that use LDAM-DRW SHALL record enough metadata to identify the loss family, whether deferred reweighting was enabled, and when late-stage reweighting became active.

#### Scenario: Completed LDAM-DRW run records schedule details
- **WHEN** an LDAM-DRW run completes successfully
- **THEN** its artifacts SHALL identify the loss as LDAM
- **AND** they SHALL record whether deferred reweighting was enabled
- **AND** they SHALL record the activation boundary used for late-stage reweighting

#### Scenario: Non-LDAM runs do not claim LDAM-DRW evidence
- **WHEN** a run does not use LDAM-DRW
- **THEN** its artifacts SHALL not imply that a deferred reweighting schedule was active

### Requirement: Adoption evidence for LDAM-DRW is phase-aware
The reproducibility contract SHALL require LDAM-DRW adoption decisions to preserve the evidence needed to compare the late-stage schedule against non-DRW anchors.

#### Scenario: LDAM-DRW adoption includes schedule-aware evidence
- **WHEN** a user promotes an LDAM-DRW recipe for future homework-facing use
- **THEN** the evidence bundle SHALL include the config snapshot and run artifacts that expose the deferred reweighting behavior
- **AND** the bundle SHALL remain sufficient to compare the recipe against current non-LDAM anchors
