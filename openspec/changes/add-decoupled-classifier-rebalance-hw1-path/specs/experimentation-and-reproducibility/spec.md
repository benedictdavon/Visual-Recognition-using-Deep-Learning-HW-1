## ADDED Requirements

### Requirement: Multi-stage run lineage is machine-readable
Multi-stage recipes SHALL record enough lineage to identify the current stage, the parent checkpoint or run, and the active trainable scope used for that stage.

#### Scenario: Rebalance stage records ancestry
- **WHEN** a classifier-rebalance run completes successfully
- **THEN** its run artifacts SHALL identify the stage as classifier rebalance
- **AND** they SHALL record the parent checkpoint or parent run used to initialize the stage
- **AND** they SHALL record that only the classifier head was trainable

#### Scenario: Optional FixRes continuation records ancestry
- **WHEN** a short FixRes continuation runs after classifier rebalance
- **THEN** its artifacts SHALL record the rebalance checkpoint or run it came from
- **AND** users SHALL be able to distinguish the continuation from the earlier stages by machine-readable metadata

### Requirement: Adoption evidence for staged recipes remains stage-aware
The reproducibility contract SHALL treat a staged recipe as an evidence bundle rather than a single anecdotal run.

#### Scenario: Staged recipe adoption includes base and rebalance evidence
- **WHEN** a user promotes a staged recipe for future homework-facing use
- **THEN** the evidence bundle SHALL include the base-stage run artifacts and the rebalance-stage run artifacts
- **AND** if a FixRes continuation is part of the claimed recipe, its evidence SHALL also be included

#### Scenario: Stage-aware evidence is required even if leaderboard movement exists
- **WHEN** a staged recipe shows promising leaderboard behavior
- **THEN** that behavior alone SHALL not replace the requirement for local stage-aware evidence
