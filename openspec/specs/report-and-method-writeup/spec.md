# Report And Method-Writeup Spec

## Overview

This spec defines the minimum documentation contract needed to defend homework-facing results produced by this repository.
It is grounded in [docs/descriptions.md](../../../docs/descriptions.md), the current run artifacts written by training/validation/inference scripts, and the legality policy defined by [../homework-requirements/spec.md](../homework-requirements/spec.md) and [../model-architecture/spec.md](../model-architecture/spec.md).

## Requirements

### Observed current behavior

- The homework docs require an English PDF report with sections covering:
  - introduction
  - method
  - results
  - references
- The homework docs also require meaningful additional experiments. Pure learning-rate, batch-size, or optimizer tuning is explicitly not enough by itself.
- The repo currently writes artifacts that can support report claims:
  - architecture and parameter count: `config.yaml`, `model_metadata.json`
  - training behavior: `train.log`, `history.csv`, `summary.json`
  - validation behavior: `validate_metrics.json`, confusion-matrix outputs, per-class analysis
  - inference/submission behavior: `prediction.csv`, `inference_summary.json`
  - ensemble behavior: diversity summaries and ensemble-search outputs when those scripts are used
- The repo does not automatically generate report text, citations, or a final PDF.

### Required preserved behavior

- Any non-trivial architecture change used for homework-facing results must be report-visible.
- Tier 2 architecture choices from [../model-architecture/spec.md](../model-architecture/spec.md) require:
  - explicit naming in the report
  - a short explanation of what changed in the residual architecture
  - citation of the family or modification idea when appropriate
- The report must not claim a method was used unless the repo artifacts show that it was actually run.
- Public leaderboard performance may be reported, but it must not be the sole evidence for a method claim in this spec set.
- Claims of improvement should be backed by concrete repo artifacts such as:
  - config snapshots
  - summaries
  - validation outputs
  - inference outputs
  - ensemble summaries when relevant
- The report must not use code support as the sole argument that a model is homework-legal.
- The report must not imply use of external data, hidden labels, or train+val refit unless those workflows were actually executed in a homework-legal way.

### Must not break

- Future repo changes must continue to save enough artifacts to support reportable method and result claims.
- Architecture extensions must not be described vaguely as "still ResNet" without explaining the actual change.
- Additional experiments must not be documented as purely hyperparameter tuning when the homework docs explicitly reject that framing.
- A reported final method must remain reproducible enough that a reader can trace it back to concrete repo configs and outputs.

## Scenarios

### Scenario: Tier 2 residual-family extension is reported responsibly

- Given a homework-facing run uses `wide_resnet50_2`, `resnext101_*`, ResNet-D, SE, CBAM, or stochastic depth
- When the report describes the method
- Then it must explain the residual-family lineage, describe the change made in this repo, and cite the idea when appropriate

### Scenario: Report claim is backed by artifacts

- Given the report states that a recipe improved validation or leaderboard performance
- When that claim is audited
- Then the corresponding run config and output artifacts should exist in the repo or its documented outputs

### Scenario: Public leaderboard result alone is not enough

- Given the report claims a method is better only because one public leaderboard score increased
- When that claim is evaluated against this spec
- Then the evidence is incomplete unless it is paired with the local artifacts expected by the reproducibility and evaluation specs

### Scenario: Unsupported architectural claim is invalid

- Given the report describes a model as homework-safe only because "the code supports it"
- When legality is evaluated
- Then the claim is invalid without the architecture-policy reasoning defined by the homework and model specs

### Scenario: Additional experiment is too weak

- Given an "additional experiment" only changes batch size, learning rate, or optimizer
- When the report frames it as satisfying the homework's additional-experiment requirement
- Then that framing is invalid under the documented homework rules

### Scenario: Method claim exceeds what was actually run

- Given the report claims train+val refit, repeated holdout, or external-data use
- When the repo artifacts do not show that workflow
- Then the claim is invalid

## Known ambiguities or gaps

- The repo does not enforce citation completeness or PDF formatting.
- The repo does not automatically connect run directories to report sections.
- The homework docs define report expectations, but interpretation quality still depends on the author.

## Non-goals

- This spec does not prescribe writing style or page layout.
- This spec does not generate citations automatically.
- This spec does not replace the homework PDF instructions in the course materials.
