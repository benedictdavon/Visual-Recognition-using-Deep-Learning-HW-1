# Diversity-First Residual Probe

This path locks one already-supported Tier 2 residual-family candidate into an ensemble-probe role first. It does not assume the probe becomes the new anchor.

## Chosen Candidate

- Candidate: `resnext50_32x3d`
- Why this one:
  - already supported by the repo
  - already under the homework parameter limit
  - already exposed the raw-versus-EMA split that motivated this change
  - cheaper to probe than a bigger grouped-convolution fallback

## Probe Configs

- Smoke:
  - `configs/experiments/target099/diversity_first_residual_probe/resnext50_32x3d_probe_smoke.yaml`
- Short probe:
  - `configs/experiments/target099/diversity_first_residual_probe/resnext50_32x3d_probe_short.yaml`
- Full probe:
  - `configs/experiments/target099/diversity_first_residual_probe/resnext50_32x3d_probe_full.yaml`

The short and full configs are framed deliberately as ensemble-candidate probes. Their local `stage_gate` is only a guardrail. It is not the final continuation decision.

## Branch Export Convention

Treat raw and EMA as separate branches when both are credible:

- raw branch:
  - checkpoint: `best_raw.ckpt`
  - branch label: `raw`
- EMA branch:
  - checkpoint: `best_ema.ckpt`
  - branch label: `ema`

Validation and inference now save:

- `artifact_provenance.json`
- branch metadata inside `validate_metrics.json` or `inference_summary.json`
- `branch` inside `val_probs_with_ids.npz` or `test_probs_with_ids.npz`

That keeps later manifests branch-aware without relying only on a directory name.

## Validation / Inference Templates

- Validation batch template:
  - `configs/batch/validate_target099_resnext50_32x3d_diversity_probe_raw_ema_template.yaml`
- Inference batch template:
  - `configs/batch/infer_target099_resnext50_32x3d_diversity_probe_raw_ema_tta_template.yaml`

The template job names are intentionally branch-labeled:

- `target099_resnext50_32x3d_probe_raw`
- `target099_resnext50_32x3d_probe_ema`
- `target099_resnext50_32x3d_probe_raw_tta`
- `target099_resnext50_32x3d_probe_ema_tta`

## Manifest Convention

Use:

- `configs/manifests/target099_resnext50_32x3d_diversity_probe_vs_anchor_template.yaml`

Key rules:

- `require_branch_provenance: true`
- `treat_branch_variants_as_distinct: true`
- keep `family` shared across probe branches
- keep `selection_group` branch-specific

That allows raw and EMA to remain distinct candidates without pretending they are different architecture families.

## Continuation Gate

Continue the probe only if both layers below are acceptable:

1. Local guardrail:
   - the probe stays within a modest validation gap of the anchor
2. Diversity evidence:
   - branch-specific disagreement is real
   - rescue count is real
   - the branch is explicitly labeled

The diversity layer is encoded in the manifest thresholds and surfaced by `scripts/diversity_report.py` through:

- `passes_thresholds`
- `probe_status`
- `selection_reason`

Interpretation:

- `probe_status == continue_probe`
  - the branch is credible enough for later ensemble work
- `probe_status == stop_probe`
  - kill the probe branch even if the architecture is legal
- `probe_status == hold_near_duplicate_group`
  - keep records, but do not expand because the branch is too duplicative

## Stop Rules

Stop the probe if any of these hold:

- local metric gap to the anchor exceeds the configured tolerance
- validation disagreement is weak
- test disagreement is weak when test probabilities are available
- rescue count is weak
- branch provenance is missing

The point is to reject weak diversity candidates early instead of treating legality as sufficient justification.

## Verification Outcome

Verification was run against the current FixRes anchor using the existing full `resnext50_32x3d` probe artifacts.

- Raw branch validation:
  - `val_acc = 89.00`
  - `val_nll = 0.5522`
  - branch provenance stayed explicit as `raw`
- EMA branch validation:
  - `val_acc = 19.00`
  - `val_nll = 3.6984`
  - branch provenance stayed explicit as `ema`

The branch-aware diversity report produced the expected split:

- `target099_resnext50_32x3d_probe_raw`
  - `passes_thresholds = true`
  - `probe_status = continue_probe`
  - `rescue_count = 5`
- `target099_resnext50_32x3d_probe_ema`
  - `passes_thresholds = false`
  - `probe_status = stop_probe`
  - failed on the local gap guardrail

That is the intended behavior for this path. The change is not claiming the architecture is broadly better. It is only proving that branch-specific diversity evidence can rescue one branch while killing another.

Under the current search guardrails, ensemble search still keeps only the anchor:

- selected model set:
  - `target099_fixres_anchor_ema`
- stop condition:
  - `stop_no_nll_gain`

So the current conclusion is:

- keep the raw branch as a legitimate future ensemble candidate
- discard the EMA branch for this probe family
- do not promote the probe to a new default anchor

## Later Full Rerun

A later full rerun reproduced the same branch split:

- run:
  - `outputs/target099_diversity_probe/20260330_224556_target099_diversity_probe_resnext50_32x3d_full`
- best raw checkpoint:
  - `val_acc = 89.00`
  - `val_nll = 0.4936`
  - `val_ece = 0.0713`
- best EMA checkpoint:
  - `val_acc = 19.00`
  - `val_nll = 3.7905`
  - `val_ece = 0.1255`

That rerun makes the branch-level conclusion harder to dispute:

- raw is still a plausible future diversity candidate
- EMA is still unusable
- this family remains a probe-only branch, not the new single-model anchor

## Example Commands

Train short probe:

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --train-config configs/experiments/target099/diversity_first_residual_probe/resnext50_32x3d_probe_short.yaml \
  --output-dir outputs/target099_diversity_probe
```

Validate both branches:

```bash
python scripts/validate_batch.py \
  --batch-config configs/batch/validate_target099_resnext50_32x3d_diversity_probe_raw_ema_template.yaml
```

Infer both branches:

```bash
python scripts/infer_batch.py \
  --batch-config configs/batch/infer_target099_resnext50_32x3d_diversity_probe_raw_ema_tta_template.yaml
```

Run diversity report:

```bash
python scripts/diversity_report.py \
  --manifest configs/manifests/target099_resnext50_32x3d_diversity_probe_vs_anchor_template.yaml \
  --output-dir outputs/diversity_target099_resnext50_32x3d_probe
```

Run ensemble search after diversity filtering:

```bash
python scripts/ensemble_search.py \
  --manifest configs/manifests/target099_resnext50_32x3d_diversity_probe_vs_anchor_template.yaml \
  --anchor target099_fixres_anchor_ema \
  --diversity-summary outputs/diversity_target099_resnext50_32x3d_probe/diversity_summary.csv \
  --output-dir outputs/ensemble_target099_resnext50_32x3d_probe
```

## Evidence Bundle Required For Inclusion

Do not justify ensemble inclusion with `best.ckpt` alone. The minimum evidence bundle is:

- probe `summary.json`
- probe `run_metadata.json`
- branch-specific `validate_metrics.json`
- branch-specific `artifact_provenance.json`
- branch-specific `val_probs_with_ids.npz`
- branch-specific `test_probs_with_ids.npz`
- branch-aware diversity manifest
- `diversity_summary.csv`
- `diversity_summary.json`
- `ensemble_search_summary.json` if the branch enters search

If that bundle is missing, the probe should remain a note, not a promoted ensemble candidate.
