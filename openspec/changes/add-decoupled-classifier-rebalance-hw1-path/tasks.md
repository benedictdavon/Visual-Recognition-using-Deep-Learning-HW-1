## 1. Staged Training Controls

- [x] 1.1 Add explicit staged-training config support for a classifier-rebalance phase that can distinguish ordinary end-to-end runs from backbone-frozen runs.
- [x] 1.2 Update the training entrypoint to require a valid parent checkpoint for the classifier-rebalance stage and fail fast when the checkpoint is missing or incompatible.
- [x] 1.3 Update optimizer / trainable-parameter setup so classifier-rebalance runs freeze the backbone and optimize only the classifier head while ordinary runs remain unchanged.

## 2. Run Lineage And Artifacts

- [x] 2.1 Persist stage identity, parent checkpoint or parent run reference, and active trainable scope in staged run metadata and summaries.
- [x] 2.2 Keep existing checkpoint selection and ordinary run artifact behavior compatible with current validation and inference workflows.
- [x] 2.3 Make the optional short FixRes continuation record its rebalance-stage ancestry distinctly from the base run.

## 3. Experiment Scaffolding

- [x] 3.1 Add staged experiment configs for the current `ResNet101 + logit_adjusted_ce` base and the classifier-rebalance continuation.
- [x] 3.2 Add an optional short FixRes continuation config that warm-starts from the staged rebalance output instead of treating FixRes as an automatic default.
- [x] 3.3 Add documentation or runbook notes that explain the cRT-style hypothesis, stage gates, and report-safe homework framing for this path.

## 4. Verification

- [x] 4.1 Run smoke-level verification that a classifier-rebalance stage launches from a valid base checkpoint and trains only the classifier head.
- [x] 4.2 Verify that the rebalance stage fails fast on missing or incompatible checkpoints rather than silently degrading to ordinary training.
- [x] 4.3 Record the staged recipe evidence and next-step recommendation so base, rebalance, and optional FixRes stages can be compared as one adoption bundle.
