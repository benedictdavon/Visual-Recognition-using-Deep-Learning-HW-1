## ADDED Requirements

### Requirement: Stage-specific trainable scope control
The training pipeline SHALL support an explicit stage configuration that freezes the backbone and optimizes only the classifier head for decoupled classifier rebalance.

#### Scenario: Backbone freezing is active during classifier rebalance
- **WHEN** the training pipeline runs a classifier-rebalance stage
- **THEN** backbone parameters SHALL be excluded from optimization updates
- **AND** classifier-head parameters SHALL remain trainable

#### Scenario: Ordinary runs remain unchanged when stage freezing is disabled
- **WHEN** the training pipeline runs a normal single-stage recipe without classifier-rebalance controls
- **THEN** training behavior SHALL remain unchanged from the current ordinary end-to-end path

### Requirement: Rebalance-stage warm-start semantics are explicit
The training pipeline SHALL require an explicit parent checkpoint for a classifier-rebalance stage and SHALL fail fast if that checkpoint is missing or incompatible.

#### Scenario: Rebalance stage fails fast on incompatible checkpoint
- **WHEN** a user provides a checkpoint that does not match the requested staged recipe
- **THEN** the pipeline SHALL fail before optimization begins
- **AND** it SHALL not silently reinterpret the request as an ordinary training run

#### Scenario: Rebalance stage preserves builder-based legality checks
- **WHEN** a classifier-rebalance stage builds its model from a parent checkpoint
- **THEN** model construction SHALL still route through the repo's normal builder path
- **AND** the usual homework legality and parameter-budget checks SHALL remain in force
