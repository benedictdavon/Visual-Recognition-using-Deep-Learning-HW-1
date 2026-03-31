# Diversity-First Ensemble Filtering

The handoff diagnosis points to a correlation ceiling, not a simple missing weight tweak. This repo now includes a diversity-first ranking utility to filter candidate models before ensemble search.

## Script

- `scripts/diversity_report.py`

## What It Computes

Per candidate, relative to a chosen anchor:
- argmax disagreement on validation
- argmax disagreement on test
- rescue count on validation:
  - anchor wrong
  - candidate right
- optional mean JS divergence on validation distributions
- optional mean JS divergence on test distributions
- validation gap to the best candidate under the chosen metric

## Inputs

The script uses a YAML manifest with entries such as:

```yaml
anchor: resnet101_v2_fast_seed2026_ema
metric: val_acc
require_branch_provenance: true
treat_branch_variants_as_distinct: true
thresholds:
  val_gap_tolerance: 1.0
  min_val_disagreement: 0.03
  min_test_disagreement: 0.03
  min_rescue_count: 2
  min_js_divergence: 0.0
  max_per_family: 2
candidates:
  - name: resnet101_v2_fast_seed2026_ema
    family: base3
    branch: ema
    selection_group: base3:ema
    val_metrics: outputs/validate_batch/resnet101_v2_fast_seed2026_ema/validate_metrics.json
    val_artifact: outputs/validate_batch/resnet101_v2_fast_seed2026_ema/val_probs_with_ids.npz
    test_artifact: outputs/infer_batch/resnet101_v2_fast_seed2026_ema_tta/test_probs_with_ids.npz
  - name: fixres320_seed2026_ema
    family: fixres320
    branch: ema
    selection_group: fixres320:ema
    val_metrics: outputs/validate_batch/fixres320_seed2026_ema/validate_metrics.json
    val_artifact: outputs/validate_batch/fixres320_seed2026_ema/val_probs_with_ids.npz
    test_artifact: outputs/infer_batch/fixres320_seed2026_ema_tta/test_probs_with_ids.npz
```

Supported artifact formats:
- `.npz` probability/logit artifacts
- `.csv` prediction artifacts for disagreement-only fallback paths

## Usage

```bash
python scripts/diversity_report.py \
  --manifest outputs/diversity_manifest.yaml \
  --output-dir outputs/diversity_stage0 \
  --val-gap-tolerance 1.0 \
  --min-val-disagreement 0.03 \
  --min-test-disagreement 0.03 \
  --min-rescue-count 2 \
  --min-js-divergence 0.0 \
  --max-per-family 1
```

## Outputs

- `diversity_summary.csv`
- `diversity_summary.json`

The CSV ranks candidates by usefulness for ensemble inclusion and marks:
- threshold failures
- branch-aware duplicate filtering through `selection_group`
- `probe_status` for continue / stop / hold decisions
- final inclusion recommendation

## Practical Guidance

- Treat `0.03` disagreement as 3%.
- Do not include candidates that are close in validation metric but nearly identical to the anchor.
- Prefer candidates with real rescue count, not only small random disagreement.
- Rerun `scripts/validate.py` after this phase if older validation directories do not yet contain `val_probs_with_ids.npz`.
