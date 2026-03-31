# Stage 1: Medium-Cost Core Runs

Goal: spend the main training budget only after Stage 0 identifies a winning base recipe.

## Planned Runs

### S1-R1 / S1-R2
- Recipe: full train of the winning Stage 0 tail-aware base, two seeds
- Purpose: verify that the Stage 0 signal is real and not a single-seed artifact.

### S1-R3
- Recipe: short FixRes refresh on the best Stage 1 seed(s)
- Purpose: keep the proven resolution-refresh pattern, but only after the base recipe is locked.

### S1-R4
- Recipe: full `WideResNet50-2` only if the pilot passes and legality remains acceptable
- Purpose: introduce additional diversity without committing before a cheap pilot passes.

### S1-R5
- Recipe: early-peak snapshot retention / soup candidate preparation if repo behavior supports it
- Purpose: exploit the documented early useful epoch zone without broad refactors.

## Scaffold Configs

- `configs/experiments/target099/stage1_winner_seed42_template.yaml`
- `configs/experiments/target099/stage1_winner_seed3407_template.yaml`
- `configs/experiments/target099/stage1_fixres320_winner_template.yaml`
- `configs/experiments/target099/stage1_wide_resnet50_2_full_optional.yaml`

Note:
- the seed templates default to `logit_adjusted_ce`
- if Stage 0 picks a different winner, update the loss and sampler blocks to match before running
- snapshot/soup remains scaffold-only in this repo; use the retained `epoch_*.ckpt` files as the starting point

## Success Criteria
- Winning Stage 0 recipe remains competitive across two seeds.
- Short FixRes refresh improves or preserves the best seed without severe calibration drift.
- At least two non-near-duplicate candidates are available for diversity-first ensemble search.

## Kill Criteria
- Stop Stage 1 expansion if the Stage 0 winner does not reproduce across seeds.
- Skip FixRes continuation if short refresh consistently hurts `val_nll` or degrades diversity.
- Keep optional branches optional unless the cheap pilot or prior stage gate clearly passes.

## Required Artifacts
- multi-seed run metadata
- retained early-epoch checkpoints where enabled
- validation exports for diversity and rescue-count analysis
