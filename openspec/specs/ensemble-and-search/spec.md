# Ensemble And Search Spec

## Overview

This spec defines the current contract for hard-vote ensembling, soft-vote ensembling, diversity ranking, and greedy ensemble search.
It is derived from [scripts/ensemble.py](../../../scripts/ensemble.py), [scripts/diversity_report.py](../../../scripts/diversity_report.py), [scripts/ensemble_search.py](../../../scripts/ensemble_search.py), and the current probability-artifact schemas produced by validation and inference.

This spec must be read together with:
- [../inference-and-submission/spec.md](../inference-and-submission/spec.md)
- [../evaluation-and-validation/spec.md](../evaluation-and-validation/spec.md)
- [../experimentation-and-reproducibility/spec.md](../experimentation-and-reproducibility/spec.md)

## Requirements

### Observed current behavior

- `scripts/ensemble.py` supports:
  - hard voting over prediction CSV files
  - soft voting over `.npz` probability artifacts
- In hard mode:
  - every CSV must contain the configured ID and target columns
  - all candidate CSVs must have identical ID order
  - optional weights are applied as weighted voting over target values
- In soft mode:
  - inputs come from `.npz` artifacts
  - supported arrays are `probs` or `logits`
  - `sample_ids` are required
  - candidate arrays must have matching shape and sample order
  - label-name output requires an `idx_to_label` mapping
- `scripts/diversity_report.py` supports candidate ranking against an anchor using:
  - validation disagreement
  - optional test disagreement
  - rescue count
  - optional JS divergence
  - validation metric gap to best
  - family-based duplicate filtering
- `diversity_report.py` accepts candidate artifacts as `.npz` or `.csv`, but rescue-count computation requires validation targets to be available.
- `scripts/ensemble_search.py` performs greedy forward selection over candidate probability artifacts:
  - anchor is mandatory
  - candidate loading is optionally filtered by a `diversity_summary.csv`
  - weights are searched over a discrete grid
  - selection is NLL-aware with an accuracy-drop guard
  - final `prediction.csv` is written only when all selected candidates provide test probabilities
- `ensemble_search.py` requires NPZ-based validation artifacts with targets; it does not accept CSV validation artifacts for the search itself.

### Required preserved behavior

- Hard-vote and soft-vote behavior must remain distinct. Missing soft-vote artifacts must not silently fall back to hard voting.
- Probability-based tools must continue to require `sample_ids` alignment across candidates.
- Candidate pools for soft voting, diversity ranking, and greedy search are only valid if they share:
  - the same class ordering
  - the same label space
  - the same sample-ID universe for the compared split
- `ensemble_search.py` may assume probability artifacts come from the same class order even though the current code does not fully validate this.
- Output contracts should remain stable:
  - `ensemble.py`: `prediction.csv`
  - `diversity_report.py`: `diversity_summary.csv`, `diversity_summary.json`
  - `ensemble_search.py`: `search_trace.csv`, `ensemble_search_summary.json`, and optional `prediction.csv` plus `ensemble_test_probs_with_ids.npz`

### Must not break

- Mismatched sample IDs or incompatible tensor shapes must continue to fail fast.
- Missing `idx_to_label` mapping in label-name soft-vote mode must continue to fail fast.
- Greedy search must not invent a final submission when selected candidates do not all provide test probabilities.
- Diversity filtering and greedy search must remain separate stages. `ensemble_search.py` may use a diversity summary when provided, but it must not pretend to recompute diversity policy on its own.

## Scenarios

### Scenario: Hard-vote ensemble succeeds on aligned CSVs

- Given multiple prediction CSV files with the same ID order
- When `scripts/ensemble.py --mode=hard` runs
- Then it must emit `prediction.csv`
- And the output IDs must preserve the shared candidate order

### Scenario: Hard-vote ensemble rejects mismatched ID order

- Given two prediction CSV files with different ID order
- When `scripts/ensemble.py --mode=hard` runs
- Then it must fail rather than silently reorder rows

### Scenario: Soft-vote ensemble requires probability artifacts

- Given `scripts/ensemble.py --mode=soft`
- When `--prob-files` are missing
- Then the command must fail rather than degrade to hard-vote behavior

### Scenario: Soft-vote ensemble requires label mapping for label-name output

- Given `scripts/ensemble.py --mode=soft --use-label-name`
- And no `idx_to_label.json` can be resolved
- When the command runs
- Then it must fail rather than emit ambiguous labels

### Scenario: Diversity report requires usable validation targets

- Given a diversity manifest whose validation artifacts do not contain targets in either the anchor or the candidate
- When `scripts/diversity_report.py` computes rescue count
- Then it must fail rather than fabricate rescue metrics

### Scenario: Ensemble search requires anchor and NPZ validation artifacts

- Given `scripts/ensemble_search.py`
- When the anchor is missing from the candidate pool or a validation artifact lacks targets
- Then the search must fail

### Scenario: Search can rank candidates without writing a submission

- Given selected candidates provide validation probabilities but not test probabilities
- When `scripts/ensemble_search.py` finishes
- Then it may still write `ensemble_search_summary.json`
- But `prediction_csv` must be `null` and no final submission should be implied

### Scenario: Label-space mismatch is invalid even if shape matches

- Given two NPZ probability artifacts with the same class dimension but different class ordering
- When they are added to a candidate pool
- Then the pool is invalid
- And current scripts are allowed to rely on the user to preserve consistent label space because they do not fully validate it

### Scenario: Corrupted diversity filter may distort search input

- Given a stale or manually edited `diversity_summary.csv`
- When `scripts/ensemble_search.py` uses it as a filter
- Then the resulting candidate pool reflects that file's contents
- And users must treat the diversity summary itself as part of the reproducibility evidence

## Known ambiguities or gaps

- Soft-vote and search tools validate sample alignment more strongly than class-order alignment.
- `diversity_report.py` can load CSV artifacts, while `ensemble_search.py` requires NPZ probability artifacts. The split in accepted formats is intentional but easy to misuse.
- None of these utilities automate final zip packaging for submission.

## Non-goals

- This spec does not require more advanced ensemble search algorithms.
- This spec does not define leaderboard strategy beyond artifact and alignment safety.
- This spec does not require the repo to auto-discover candidate pools from `outputs/`.
