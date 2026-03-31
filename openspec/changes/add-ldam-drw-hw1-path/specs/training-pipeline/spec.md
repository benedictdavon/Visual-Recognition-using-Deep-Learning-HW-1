## ADDED Requirements

### Requirement: LDAM loss is supported as a first-class training option
The training pipeline SHALL support an LDAM loss option that derives its margin behavior from valid class-count inputs and remains compatible with the current config-driven training path.

#### Scenario: LDAM runs with valid class counts
- **WHEN** a user configures the training pipeline to use LDAM and valid class counts are available
- **THEN** training SHALL build the LDAM loss explicitly rather than silently degrading to an existing CE-family loss

#### Scenario: LDAM rejects invalid class-count inputs
- **WHEN** a user configures the training pipeline to use LDAM and class-count inputs are missing or invalid
- **THEN** the run SHALL fail before training proceeds

### Requirement: Deferred reweighting schedule is supported explicitly
The training pipeline SHALL support a deferred reweighting schedule in which class-balanced weighting is inactive for the early training phase and activates at a configured later phase.

#### Scenario: Deferred reweighting leaves early training unchanged
- **WHEN** the current epoch is earlier than the configured deferred reweighting boundary
- **THEN** the pipeline SHALL behave as the pre-reweighting phase for that LDAM-DRW recipe

#### Scenario: Deferred reweighting activates at the configured boundary
- **WHEN** the current epoch reaches or passes the configured deferred reweighting boundary
- **THEN** the pipeline SHALL activate the configured class-balanced weighting behavior for the remainder of the run

#### Scenario: Ordinary runs remain unchanged when DRW is disabled
- **WHEN** a user runs a recipe without LDAM-DRW controls
- **THEN** ordinary training behavior SHALL remain unchanged from the current repo path
