## ADDED Requirements

### Requirement: Classifier-only rebalance stage
The system SHALL support a legal multi-stage `ResNet101` recipe in which a saved base checkpoint can launch a classifier-rebalance stage that keeps the backbone frozen and updates only the classifier head.

#### Scenario: Rebalance stage starts from a valid base checkpoint
- **WHEN** a user launches the decoupled classifier-rebalance stage with a valid `ResNet101` base checkpoint
- **THEN** the run SHALL initialize from that checkpoint
- **AND** the backbone SHALL remain frozen during optimization
- **AND** the classifier head SHALL remain trainable

#### Scenario: Rebalance stage rejects missing parent checkpoint
- **WHEN** a user requests the decoupled classifier-rebalance stage without a valid parent checkpoint
- **THEN** the run SHALL fail before training begins
- **AND** the repo SHALL not silently fall back to ordinary end-to-end training

### Requirement: Optional post-rebalance FixRes continuation
The system SHALL allow the decoupled recipe to continue into a separate short FixRes-style refresh stage after classifier rebalance, while keeping that continuation explicit and optional.

#### Scenario: FixRes continuation remains a separate stage
- **WHEN** a user runs the optional short FixRes continuation
- **THEN** the continuation SHALL be represented as a distinct stage from both the base run and the classifier-rebalance stage
- **AND** its outputs SHALL remain distinguishable from earlier stages

#### Scenario: Base and rebalance stages remain usable without FixRes
- **WHEN** a user stops after the base run or after the classifier-rebalance stage
- **THEN** the recipe SHALL remain valid without requiring the optional FixRes continuation
