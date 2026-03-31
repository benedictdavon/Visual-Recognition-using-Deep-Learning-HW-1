# Repo Architecture Spec

## Overview

This spec documents the structural invariants that matter for future work in this brownfield repository.
It is grounded in the current root layout, the CLI scripts in `scripts/`, the implementation modules in `src/`, the config tree in `configs/`, and the current spec location under `openspec/specs/`.

## Requirements

### Observed current behavior

- Runtime entrypoints live under `scripts/`:
  - training: `train.py`, `train_batch.py`
  - validation: `validate.py`, `validate_batch.py`
  - inference: `infer.py`, `infer_batch.py`
  - ensemble/search utilities: `ensemble.py`, `ensemble_search.py`, `diversity_report.py`
  - data inspection: `inspect_dataset.py`
- Reusable code lives under `src/`:
  - `src/data/`
  - `src/models/`
  - `src/losses/`
  - `src/engine/`
  - `src/utils/`
  - `src/submission/`
- Configs live under `configs/`, with:
  - base config in `configs/config.yaml`
  - reusable partial configs in `configs/model/`, `configs/train/`, `configs/aug/`, `configs/inference/`
  - experiment and batch manifests in `configs/experiments/` and `configs/batch/`
- Historical artifacts and run outputs live under `outputs/`.
- Specs live under `openspec/specs/`. The top-level `specs/` directory is currently unused and not authoritative.
- The repo is not packaged as an installable module. CLI scripts add the repo root to `sys.path` at runtime.

### Required preserved behavior

- Thin CLI scripts should remain thin. Business logic belongs in `src/`, not as script-only implementations.
- Config-driven behavior should continue to flow from `configs/config.yaml` plus layered overrides.
- Training, validation, inference, and submission behavior should remain discoverable from the existing script entrypoints.
- New specs should live under `openspec/specs/` rather than creating a second authoritative spec tree elsewhere.
- `docs/` may contain strategy notes, experiment notes, and handoff material, but those docs should not silently replace the contract defined in OpenSpec.

### Must not break

- `scripts/train.py` must remain the primary training entrypoint.
- `scripts/validate.py` must remain the primary checkpoint-validation entrypoint.
- `scripts/infer.py` must remain the primary inference-to-submission entrypoint.
- `src/models/builder.py` must remain the parameter-budget enforcement boundary.
- `src/submission/make_submission.py` must remain the submission row/CSV construction boundary.
- `outputs/` must remain the default home for generated run artifacts rather than source directories or docs.

## Scenarios

### Scenario: New experiment config is added

- Given a new YAML under `configs/experiments/`
- When a user runs `scripts/train.py --config configs/config.yaml --train-config <experiment>`
- Then the run should execute without editing source files
- And the merged config snapshot should be written into the run directory

### Scenario: New runtime helper is added

- Given a helper for data loading, model construction, evaluation, or artifact bookkeeping
- When it is added to the repo
- Then it should live under the matching `src/` package
- And the CLI script should remain a thin wrapper around it

### Scenario: New spec is added

- Given a new contract area that needs formal documentation
- When a new spec is created
- Then it should be added under `openspec/specs/`
- And it should cross-reference related specs rather than duplicating them in `docs/`

### Scenario: Notebook-only logic is proposed

- Given a future workflow that only exists in a notebook or ad hoc shell commands
- When that workflow becomes required for training, validation, inference, or submission
- Then the change is invalid until the behavior is represented in the normal script-plus-`src/` path

## Known ambiguities or gaps

- `docs/` contains both homework source material and strategy notes. Not every markdown file there is normative.
- `outputs/` contains historical runs from multiple phases, but there is no central registry or manifest covering all past artifacts.
- The empty top-level `specs/` directory can confuse future contributors unless they notice that `openspec/specs/` is the real spec tree.

## Non-goals

- This spec does not require a package refactor, a dependency-manager migration, or CI.
- This spec does not require every historical doc in `docs/` to be rewritten as a spec.
- This spec does not define experiment policy by itself; that belongs to the homework, model, training, evaluation, and reproducibility specs.
