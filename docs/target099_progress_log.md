# Target 0.99 Progress Log

## 2026-03-25

### Phase A started
- Read `docs/handoff_trial_summary_target099.md` as the strategy source of truth.
- Audited the active repo structure: training, validation, inference, dataset, model, loss, metrics, checkpoint, submission, and ensemble paths.
- Confirmed current artifact locations for:
  - run outputs under `outputs/<run_dir>/`
  - validation exports under `outputs/validate_batch/*`
  - soft-vote probability pools under `outputs/infer_batch/*/test_probs_with_ids.npz`
  - prior ensemble sweeps under `outputs/ensemble_sweep/*`

### Phase A findings
- Existing strengths:
  - training/inference pipeline is already functional and reproducible
  - EMA, warm-starting, short FixRes configs, and horizontal-flip TTA already exist
  - current output layout already saves config snapshots and basic summaries
- Missing for the approved plan:
  - model-selection metrics beyond accuracy
  - dense checkpoint retention around early peaks
  - structured run metadata with stage-gate status
  - tail-aware losses
  - tail-aware sampler
  - diversity-first analysis utility
  - greedy, NLL-aware ensemble search
- Repo-specific constraint:
  - no automated test suite is present, so changes need script-level sanity validation

### Phase A completed
- Added `docs/target099_execution_plan.md` with the repo mapping and sequential implementation order.

### Next
- Phase B: implement metrics/model-selection/checkpoint/metadata support before any Stage 0 configs.

### Phase B completed
- Added validation metrics support in the live training/validation path:
  - `val_acc`
  - `val_macro_recall`
  - `val_nll`
  - `val_ece`
- Added configurable model-selection settings while preserving the default accuracy-based behavior.
- Added dense checkpoint support with:
  - `train.checkpointing.save_every_epoch`
  - `train.checkpointing.keep_top_k`
- Added structured run metadata via `run_metadata.json`, including:
  - config snapshot path
  - metrics summary
  - checkpoint inventory
  - rationale/notes fields from config
  - stage-gate evaluation result
- Added stage-discipline docs:
  - `docs/experiments/stage0.md`
  - `docs/experiments/stage1.md`
  - `docs/experiments/stage2.md`
- Ran lightweight sanity checks:
  - `python -m py_compile ...` on edited modules
  - evaluator metadata smoke test
  - trainer fit smoke test inside workspace

### Next
- Phase C: add Balanced Softmax and logit-adjusted CE with train-prior wiring.

### Phase C completed
- Added train-prior-driven loss support in the live path:
  - `balanced_softmax`
  - `logit_adjusted_ce`
- Preserved the existing plain CE path without changing its behavior.
- Wired train class counts from `bundle.train_df` into both `scripts/train.py` and `scripts/validate.py`.
- Added `loss.logit_adjusted_tau` to the base config.
- Added documentation:
  - `docs/experiments/losses_tail_aware.md`
- Ran lightweight sanity checks:
  - `python -m py_compile` on the updated files
  - direct loss-construction smoke test for both new losses

### Next
- Phase D: add weighted random sampler support with configurable power.

### Phase D completed
- Added configurable weighted random sampling in the live training loader path.
- New config surface:
  - `sampler.name`
  - `sampler.power`
  - `sampler.replacement`
  - `sampler.num_samples`
- Implemented inverse-power class weighting, including the Stage 0 target setting of inverse-sqrt frequency (`power: 0.5`).
- Logged active sampler metadata at train startup.
- Added documentation:
  - `docs/experiments/sampler_tail_aware.md`
- Ran lightweight sanity checks:
  - `python -m py_compile` on the updated files
  - direct sampler-construction smoke test

### Next
- Phase E: formalize the short FixRes refresh workflow and template config.

### Phase E completed
- Formalized short-refresh support in the base config by adding:
  - `train.init_checkpoint`
  - `train.init_use_ema`
- Added a short FixRes refresh template:
  - `configs/experiments/target099_fixres320_short_refresh_template.yaml`
- Added documentation:
  - `docs/experiments/fixres_short_refresh.md`
- Verified that the existing code path already supports the required mechanics:
  - image-size override
  - short duration
  - low LR
  - mixup/cutmix disable
  - warm-start from checkpoint

### Next
- Phase F: add diversity analysis tooling for disagreement and rescue-count ranking.

### Phase F completed
- Extended `scripts/validate.py` to export `val_probs_with_ids.npz` when analysis is enabled.
- Added diversity-first candidate ranking utility:
  - `scripts/diversity_report.py`
- Implemented candidate scoring features:
  - val disagreement vs anchor
  - test disagreement vs anchor
  - rescue count on val
  - optional JS divergence on val/test distributions
  - thresholded include/exclude logic
  - family-based duplicate filtering
- Added documentation:
  - `docs/experiments/diversity_first_ensemble.md`
- Ran lightweight sanity checks:
  - `python -m py_compile` on the updated scripts
  - synthetic manifest + synthetic artifact smoke run

### Next
- Phase G: add greedy, NLL-aware ensemble search tooling.

### Phase G completed
- Added greedy, NLL-aware ensemble search utility:
  - `scripts/ensemble_search.py`
- Implemented:
  - greedy forward selection
  - coarse discrete weight search
  - locked anchor support
  - pool-size cap
  - NLL-first selection with accuracy guardrail
  - optional filtering through diversity summary output
  - final `prediction.csv` generation when test probs are available
- Added documentation:
  - `docs/experiments/ensemble_search.md`
- Ran lightweight sanity checks:
  - `python -m py_compile` on the new script
  - synthetic ensemble-search smoke run

### Next
- Phase H: create Stage 0 configs and the detailed Stage 0 runbook.

### Phase H completed
- Added Stage 0 experiment configs under `configs/experiments/target099/`:
  - `stage0_s0_r1_resnet101_logit_adjusted.yaml`
  - `stage0_s0_r2_resnet101_balanced_softmax.yaml`
  - `stage0_s0_r3_resnet101_sampler_inv_sqrt.yaml`
  - `stage0_s0_r4_fixres320_from_s0r1.yaml`
  - `stage0_s0_r4_fixres320_from_s0r2.yaml`
  - `stage0_s0_r4_fixres320_from_s0r3.yaml`
  - `stage0_s0_r5_wide_resnet50_2_optional.yaml`
- Added detailed Stage 0 runbook:
  - `docs/experiments/stage0_runbook.md`
- Added live model support for `wide_resnet50_2` so the optional pilot config is runnable rather than dead config.
- Ran lightweight integrity checks:
  - YAML load check for all new Stage 0 configs
  - `build_model()` smoke check for `wide_resnet50_2`

### Next
- Phase I: add Stage 1 / Stage 2 scaffold configs and finish delivery docs.

### Phase I completed
- Added Stage 1 scaffold configs:
  - `stage1_winner_seed42_template.yaml`
  - `stage1_winner_seed3407_template.yaml`
  - `stage1_fixres320_winner_template.yaml`
  - `stage1_wide_resnet50_2_full_optional.yaml`
- Added Stage 2 scaffold configs:
  - `stage2_repeated_holdout_kfold_template.yaml`
  - `stage2_grouped_conv_resnext101_32x8d_optional.yaml`
- Updated the dataset loader so an explicit `dataset.val_dir: null` can disable auto-pickup of `data/val`, making the repeated-holdout template live.
- Updated `docs/experiments/stage1.md` and `docs/experiments/stage2.md` with scaffold file references and the deferred train+val-refit note.
- Updated `README.md` to list `wide_resnet50_2` as a supported live model path.
- Ran lightweight integrity checks:
  - `python -m py_compile` on the updated code
  - YAML load check across all `configs/experiments/target099/*.yaml`

### Next
- Phase J: write delivery summary, finalize progress log, and prepare the handoff response.

### Phase J completed
- Added final handoff summary:
  - `docs/target099_delivery_summary.md`
- Finalized the repo-side documentation for:
  - Stage 0 execution
  - diversity evaluation
  - ensemble search
  - Stage 1 / Stage 2 scaffolding

### Final Status
- Phase A through Phase I implementation is complete at the repository-preparation level.
- Stage 0 is fully configured and ready to run.
- Later GPU experiments are recorded below.

## 2026-03-30 to 2026-03-31

### Post-delivery experiment work started
- Moved from repo-preparation into actual `target099` runs and post-handoff path extensions.
- Implemented and exercised two OpenSpec-backed experiment additions:
  - LDAM-DRW late reweighting path
  - diversity-first residual probe path
- Added one non-OpenSpec residual-family backbone extension:
  - `resnetv2_101x1_bit.goog_in21k_ft_in1k` via `timm`

### LDAM-DRW path completed and evaluated
- Added first-class LDAM support plus deferred reweighting activation metadata and logging.
- Verified the smoke path and negative guards:
  - invalid schedule fails before training
  - invalid class counts fail in the loss builder
- Ran the full experiment:
  - run: `outputs/target099_ldam_drw/20260330_151455_target099_ldam_drw_resnet101`
  - selected checkpoint:
    - `val_acc = 89.00`
    - `val_nll = 3.3001`
    - `val_ece = 0.8478`
  - best raw checkpoint:
    - `val_acc = 90.33`
    - `val_nll = 3.5495`
- Verdict:
  - implementation is correct
  - experiment result is not competitive
  - path should be killed for primary submission use

### Diversity-first residual probe path completed and evaluated
- Added branch-aware artifact provenance for raw versus EMA handling in validation, inference, diversity reporting, and ensemble search.
- Verified the initial branch-aware probe workflow against the existing `resnext50_32x3d` branch pair.
- Ran a fresh full `resnext50_32x3d` probe:
  - run: `outputs/target099_diversity_probe/20260330_224556_target099_diversity_probe_resnext50_32x3d_full`
  - best raw checkpoint:
    - `val_acc = 89.00`
    - `val_nll = 0.4936`
    - `val_ece = 0.0713`
  - best EMA checkpoint:
    - `val_acc = 19.00`
    - `val_nll = 3.7905`
- Verdict:
  - the raw branch remains a possible diversity/ensemble candidate
  - the EMA branch should be discarded
  - the architecture should not replace the current anchor

### BiT residual-family path integrated and evaluated
- Added support for `resnetv2_101x1_bit.goog_in21k_ft_in1k` to the legal residual-family builder path.
- Added a dedicated strong `target099` config:
  - `configs/experiments/target099/resnetv2_101x1_bit_goog_in21k_ft_in1k_strong.yaml`
- Ran one weak baseline recipe first:
  - run: `outputs/bit_resnetv2_101x1/20260331_000827_baseline_resnet50`
  - result:
    - `val_acc = 85.67`
  - diagnosis:
    - bad recipe mismatch, not a useful verdict on the backbone
- Ran the strong `target099` BiT recipe:
  - run: `outputs/bit_resnetv2_101x1/20260331_120013_target099_bit_resnetv2_101x1_strong`
  - training crashed during `epoch_025.ckpt` write, but earlier checkpoints were preserved
  - validated `best_ema.ckpt`:
    - `val_acc = 92.33`
    - `val_nll = 0.3806`
    - `val_ece = 0.0791`
    - `val_acc5 = 98.67`
- Verdict:
  - current best local single-model result
  - good enough to submit without retraining the remaining epochs
  - use `best_ema.ckpt`, not the partial `epoch_025.ckpt`

### Current standing after additional experiments
- Strongest local single-model checkpoint:
  - BiT `best_ema.ckpt` at `92.33 / 0.3806`
- Strongest earlier anchor still worth keeping:
  - Stage 0 FixRes from `s0_r1` selected checkpoint at `91.00 / 0.3986`
- Calibration-first alternate checkpoint still worth keeping:
  - CRT FixRes from rebalance at `90.33 / 0.3587`

### Current repo status
- Repo preparation is complete.
- Additional experiment notes are now backed by actual run artifacts instead of plan-only docs.
- Long GPU training has now been launched, evaluated, and documented in this environment.
