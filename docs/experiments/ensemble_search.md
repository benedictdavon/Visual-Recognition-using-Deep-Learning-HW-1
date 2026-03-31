# Ensemble Search

This repo now includes a separate ensemble-search utility for the handoff’s final-selection phase.

## Script

- `scripts/ensemble_search.py`

## Supported Search Behavior

- greedy forward selection
- coarse discrete weight search only
- anchor model locked into the pool
- NLL-aware candidate acceptance
- pool cap via `--max-pool-size` (default `4`)
- optional filtering through `diversity_summary.csv`

## Required Inputs

The script expects a manifest with `.npz` probability or logit artifacts for validation, and optionally test:

```yaml
candidates:
  - name: resnet101_v2_fast_seed2026_ema
    family: base3
    branch: ema
    selection_group: base3:ema
    val_metrics: outputs/validate_batch/resnet101_v2_fast_seed2026_ema/validate_metrics.json
    val_artifact: outputs/validate_batch/resnet101_v2_fast_seed2026_ema/val_probs_with_ids.npz
    test_artifact: outputs/infer_batch/resnet101_v2_fast_seed2026_ema_tta/test_probs_with_ids.npz
```

## Usage

Basic example:

```bash
python scripts/ensemble_search.py \
  --manifest outputs/diversity_manifest.yaml \
  --anchor resnet101_v2_fast_seed2026_ema \
  --diversity-summary outputs/diversity_stage0/diversity_summary.csv \
  --output-dir outputs/ensemble_search_stage0 \
  --weight-grid 0.5 1.0 1.5 2.0 \
  --min-nll-gain 0.0001 \
  --max-acc-drop 0.5
```

## Anti-Overfitting Safeguards

- coarse weight grid only
- locked anchor
- `--min-nll-gain` required to accept an added model
- `--max-acc-drop` guard to stop obviously overfit additions
- pool cap 4 by default
- optional pre-filtering through the diversity report instead of searching all candidates blindly

## Outputs

- `search_trace.csv`
- `ensemble_search_summary.json`
- `prediction.csv` if all selected candidates provide test probabilities
- `ensemble_test_probs_with_ids.npz` when test fusion is possible

## Practical Guidance

- Use diversity filtering first, then ensemble search.
- Keep raw and EMA as separate candidates only when the manifest labels branch provenance explicitly.
- Treat `val_nll` as the primary search objective and `val_acc` as a guardrail, not the other way around.
- Do not include extra models just because a 300-image val split gives a tiny accuracy bump.
