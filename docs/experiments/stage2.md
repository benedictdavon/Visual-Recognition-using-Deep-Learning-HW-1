# Stage 2: Gated Expensive Runs

Goal: reserve the highest-cost or interpretation-sensitive paths for only the strongest evidence-backed candidates.

## Planned Runs

### S2-R1
- Recipe: repeated-holdout or repeated ranking on the locked candidate pool
- Purpose: reduce overreaction to the tiny 300-image val set when final ranking remains ambiguous.

### S2-R2
- Recipe: train+val refit after hyperparameters are locked
- Purpose: optional last-mile data usage path.
- Status: legality-sensitive and must be clearly flagged before use.

### S2-R3
- Recipe: grouped-convolution fallback only if explicitly judged legal and still needed
- Purpose: late diversity fallback only after cheaper legal paths fail.
- Status: interpretation-sensitive and optional.

## Scaffold Configs

- `configs/experiments/target099/stage2_repeated_holdout_kfold_template.yaml`
- `configs/experiments/target099/stage2_grouped_conv_resnext101_32x8d_optional.yaml`

## Repo Constraint

- A fully runnable `train+val refit` template is intentionally not added yet.
- Reason:
  - the current training loop still assumes a validation loader for checkpoint selection and reporting
  - Stage 0 / Stage 1 evidence should justify that extra change before the repo takes on a validation-free training path
- Status:
  - documented as a gated optional path
  - deferred rather than implemented as dead config

## Success Criteria
- Only enter Stage 2 when Stage 1 has already produced a small, high-quality, diversity-vetted candidate pool.
- Every Stage 2 run has a documented legality note and a concrete expected payoff.

## Kill Criteria
- Do not enter Stage 2 for runs that duplicate existing predictions or consume substantial compute without a clear diversity thesis.
- Drop any legality-sensitive path immediately if compliance is uncertain.

## Required Artifacts
- explicit legality notes
- comparison against the Stage 1 anchor pool
- diversity summary and ensemble-search outputs before any final submission choice
