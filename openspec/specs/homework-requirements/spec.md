# Homework Requirements Spec

## Overview

This spec is the top-level homework contract for this repository.
It is grounded in the current repo docs and implementation, especially [docs/descriptions.md](../../../docs/descriptions.md), [README.md](../../../README.md), [configs/config.yaml](../../../configs/config.yaml), [src/models/builder.py](../../../src/models/builder.py), and [src/submission/make_submission.py](../../../src/submission/make_submission.py).

This spec is the guardrail for all other specs in `openspec/specs/`.

## Requirements

### Observed current behavior

- The repo is built for a 100-class image classification homework with local data under `data/train`, `data/val`, and `data/test`.
- The repo currently supports multiple residual-family backbones and extensions in code, including vanilla ResNet depths, wider/cardinality variants, and residual-block modifications such as ResNet-D, SE, CBAM, and stochastic depth.
- The repo enforces the parameter budget in code through `src/models/builder.py`, using a strict `< limit` check. A model with parameter count greater than or equal to `model.param_limit_million` is rejected.
- The repo writes leaderboard-ready predictions as `prediction.csv`.
- The default runtime model is single-process, single-machine, and practical for one GPU. Batch scripts run jobs sequentially rather than in parallel.
- The repo does not programmatically enforce homework-report obligations such as citations or PDF packaging. Those obligations come from the homework docs and user workflow.

### Required preserved behavior

- Future work must preserve the homework constraints as hard requirements:
  - backbone remains in the ResNet family
  - parameter count remains strictly below 100M
  - no external data
  - submission output remains `prediction.csv`
  - the default path remains practical on a single GPU
- Future work must keep the model-selection and submission paths compatible with the current 100-class label space.
- Future work must not treat successful code execution as proof of homework compliance. Homework compliance is narrower than "the code can build the model."

#### Legality tiers for this repo

- `Tier 1: safest`
  - Vanilla torchvision-style ResNet backbones already supported here:
    - `resnet50`
    - `resnet101`
    - `resnet152`
  - Standard training-recipe changes that do not alter backbone family.
- `Tier 2: allowed with report-visible explanation and citation`
  - Residual-family extensions whose lineage remains clearly ResNet-based under the current TA interpretation:
    - `resnext50_32x3d`
    - `resnext50_32x4d`
    - `wide_resnet50_2`
    - `resnext101_64x4d`
    - `resnext101_32x8d`
    - `model.resnetd: true`
    - `model.attention: se`
    - `model.attention: cbam`
    - `model.se_mode: bottleneck`
    - `model.drop_path_rate > 0`
    - registered profile names such as `resnet101d_bse_sd` and `resnext101_32x8d_d_bse_sd`
  - These are allowed only when the report clearly explains the residual-family lineage, cites the modification or family, and states what changed in this repo.
- `Tier 3: risky or requires explicit justification before homework use`
  - New model names or architectural changes whose residual lineage is not obvious from the implementation
  - Architectures that rely on non-residual feature extractors
  - Hybrid or third-party additions that are only loosely inspired by ResNet
  - Any change that code supports but the report cannot defend as a clear ResNet-family extension

### Must not break

- The parameter-limit check in `src/models/builder.py` must remain the final enforcement point for model size.
- `prediction.csv` must remain the exact submission filename produced by the repo's inference path.
- Submission rows must continue to map back to homework label IDs rather than leaking internal training indices.
- The default workflow must not become multi-GPU-only, distributed-only, or external-service-dependent.
- Code support for a model must not be documented as automatically homework-safe without regard to legality tier and report obligations.
- Future changes must not silently widen the homework contract beyond:
  - ResNet-family backbones
  - no external data
  - strict parameter budget
  - current single-GPU practicality

## Scenarios

### Scenario: Tier 1 vanilla backbone remains homework-safe

- Given `model.name: resnet101`
- And `model.param_limit_million: 100`
- When `scripts/train.py` builds the model
- Then the run is homework-safe on architecture grounds if the parameter count is below 100M
- And the run must still be able to produce `prediction.csv` through `scripts/infer.py`

### Scenario: Tier 2 residual-family extension is allowed with explanation

- Given a run that uses `resnext50_32x3d`, `resnext50_32x4d`, `resnext101_64x4d`, `wide_resnet50_2`, or a ResNet-D / SE / CBAM / bottleneck-SE / stochastic-depth extension
- When that run is used for the homework
- Then it is allowed only if the report explains the residual-family lineage, cites the relevant idea, and states the actual repo-side change
- And the run must still satisfy the strict parameter budget and no-external-data rules

### Scenario: Parameter-limit violation is rejected

- Given a model configuration whose parameter count is greater than or equal to `model.param_limit_million`
- When `src/models/builder.py` builds the model
- Then model construction must fail before training proceeds

### Scenario: Code-supported model is not automatically Tier 1

- Given a model name that the code can build
- When homework legality is evaluated
- Then code support alone is insufficient
- And the model must be classified by legality tier, parameter budget, and report defensibility

### Scenario: External data proposal is rejected

- Given a proposal to add external data, external pseudo-label corpora, or non-homework datasets
- When the change is evaluated against this spec
- Then the change must be rejected as homework-incompatible even if it is technically easy to implement

### Scenario: Submission filename contract remains exact

- Given any single-model or ensemble inference path
- When the final submission artifact is written
- Then the file consumed for competition upload must still be named exactly `prediction.csv`

### Scenario: Single-GPU practicality is preserved

- Given a future change that only works with multi-GPU or distributed training
- When it is proposed as the default repo workflow
- Then the change is invalid unless a single-GPU practical path remains the default homework path

## Known ambiguities or gaps

- The repo does not store legality-tier metadata inside model configs or run summaries. Legality remains a spec and report responsibility rather than a runtime-enforced field.
- The repo does not enforce the "no external data" rule programmatically. This remains a process constraint.
- The repo does not automate final zip packaging for CodaBench or E3 submission.
- Homework-report rules such as English PDF, GitHub link, and citation quality are documented but not enforced by scripts.

## Non-goals

- This spec does not bless every future residual-looking idea as homework-safe.
- This spec does not define the final report text for the user.
- This spec does not require leaderboard-specific strategy beyond protecting the homework contract.
