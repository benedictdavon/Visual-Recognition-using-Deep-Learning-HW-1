"""Diversity-first candidate ranking for ensemble inclusion."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.misc import ensure_dir, load_yaml, save_json  # noqa: E402


@dataclass
class PredictionArtifact:
    """Aligned prediction artifact loaded from CSV or NPZ outputs."""

    sample_ids: np.ndarray
    preds: np.ndarray
    targets: np.ndarray | None = None
    probs: np.ndarray | None = None


_METRIC_ALIASES = {
    "val_acc": ["val_acc", "acc1"],
    "val_macro_recall": ["val_macro_recall", "macro_recall", "macro_per_class_acc"],
    "val_nll": ["val_nll", "nll"],
    "val_ece": ["val_ece", "ece"],
}
_VALID_BRANCHES = {"raw", "ema", "selected_raw", "selected_ema", "last_raw", "last_ema", "single"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for diversity-first candidate ranking."""
    parser = argparse.ArgumentParser(
        description="Rank ensemble candidates by diversity versus an anchor."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="YAML manifest describing anchor and candidates.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/diversity")
    parser.add_argument(
        "--anchor", type=str, default=None, help="Override anchor candidate name from manifest."
    )
    parser.add_argument(
        "--metric", type=str, default=None, help="Validation metric used for gap filtering."
    )
    parser.add_argument("--metric-mode", choices=["auto", "max", "min"], default="auto")
    parser.add_argument("--val-gap-tolerance", type=float, default=None)
    parser.add_argument("--min-val-disagreement", type=float, default=None)
    parser.add_argument("--min-test-disagreement", type=float, default=None)
    parser.add_argument("--min-rescue-count", type=int, default=None)
    parser.add_argument("--min-js-divergence", type=float, default=None)
    parser.add_argument("--max-per-family", type=int, default=None)
    return parser.parse_args()


def _resolve_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _metric_mode(metric_name: str, configured_mode: str) -> str:
    if configured_mode != "auto":
        return configured_mode
    if metric_name in {"val_acc", "val_macro_recall"}:
        return "max"
    if metric_name in {"val_nll", "val_ece"}:
        return "min"
    raise ValueError(f"Cannot infer mode for metric '{metric_name}'.")


def _softmax_numpy(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _to_string_array(values: Any) -> np.ndarray:
    return np.asarray(values).astype(str)


def _load_npz_artifact(path: Path) -> PredictionArtifact:
    arr = np.load(path, allow_pickle=True)
    if "sample_ids" not in arr:
        raise ValueError(f"{path} is missing 'sample_ids'.")
    sample_ids = _to_string_array(arr["sample_ids"])
    probs = None
    if "probs" in arr:
        probs = arr["probs"].astype(np.float64)
    elif "logits" in arr:
        probs = _softmax_numpy(arr["logits"].astype(np.float64))

    if "preds" in arr:
        preds = _to_string_array(arr["preds"])
    elif probs is not None:
        preds = _to_string_array(np.argmax(probs, axis=1))
    else:
        raise ValueError(f"{path} is missing usable prediction arrays.")

    targets = _to_string_array(arr["targets"]) if "targets" in arr else None
    return PredictionArtifact(sample_ids=sample_ids, preds=preds, targets=targets, probs=probs)


def _pick_column(df: pd.DataFrame, candidates: list[str], path: Path) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(f"{path} is missing expected columns: {candidates}")


def _load_csv_artifact(path: Path) -> PredictionArtifact:
    df = pd.read_csv(path)
    id_col = _pick_column(df, ["sample_id", "image_name", "id"], path)
    pred_col = _pick_column(df, ["pred_idx", "pred_label"], path)
    target_col = None
    for candidate in ["true_idx", "true_label"]:
        if candidate in df.columns:
            target_col = candidate
            break
    preds = _to_string_array(df[pred_col].tolist())
    targets = _to_string_array(df[target_col].tolist()) if target_col is not None else None
    return PredictionArtifact(
        sample_ids=_to_string_array(df[id_col].tolist()),
        preds=preds,
        targets=targets,
        probs=None,
    )


def load_artifact(path: Path) -> PredictionArtifact:
    """Load a prediction artifact from CSV or NPZ storage."""
    if path.suffix.lower() == ".npz":
        return _load_npz_artifact(path)
    if path.suffix.lower() == ".csv":
        return _load_csv_artifact(path)
    raise ValueError(f"Unsupported artifact format: {path}")


def _aligned_indices(anchor_ids: np.ndarray, candidate_ids: np.ndarray) -> np.ndarray:
    candidate_lookup = {sample_id: idx for idx, sample_id in enumerate(candidate_ids.tolist())}
    try:
        indices = np.array(
            [candidate_lookup[sample_id] for sample_id in anchor_ids.tolist()], dtype=np.int64
        )
    except KeyError as exc:
        raise ValueError(
            f"Candidate artifact is missing sample_id '{exc.args[0]}' required by the anchor."
        ) from exc
    if len(candidate_lookup) != len(anchor_ids):
        raise ValueError("Anchor and candidate artifacts must contain the same number of samples.")
    return indices


def _js_divergence(
    anchor_probs: np.ndarray | None, candidate_probs: np.ndarray | None
) -> float | None:
    if anchor_probs is None or candidate_probs is None:
        return None
    eps = 1e-12
    p = np.clip(anchor_probs, eps, 1.0)
    q = np.clip(candidate_probs, eps, 1.0)
    m = 0.5 * (p + q)
    js = 0.5 * np.sum(p * np.log(p / m), axis=1) + 0.5 * np.sum(q * np.log(q / m), axis=1)
    return float(np.mean(js))


def _load_metric_value(path: Path, metric_name: str) -> float:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for candidate in _METRIC_ALIASES[metric_name]:
        if candidate not in data:
            continue
        value = float(data[candidate])
        if candidate == "macro_per_class_acc" and metric_name == "val_macro_recall":
            return value * 100.0
        return value
    raise ValueError(f"{path} does not contain metric '{metric_name}'.")


def _candidate_sort_key(row: Dict[str, Any]) -> tuple:
    return (
        int(row["passes_thresholds"]),
        float(row["rescue_count"]),
        float(row.get("val_disagreement", 0.0) or 0.0),
        float(row.get("test_disagreement", 0.0) or 0.0),
        float(row.get("val_js_divergence", 0.0) or 0.0),
        -float(row["val_gap_to_best"]),
    )


def _normalize_branch(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    branch = str(value).strip().lower()
    if branch not in _VALID_BRANCHES:
        raise ValueError(
            f"Unsupported branch value '{value}'. Expected one of: {sorted(_VALID_BRANCHES)}"
        )
    return branch


def _selection_group_name(
    entry: Dict[str, Any],
    family: str,
    branch: str | None,
    treat_branch_variants_as_distinct: bool,
) -> str:
    configured = entry.get("selection_group")
    if configured not in {None, ""}:
        return str(configured)
    if treat_branch_variants_as_distinct and branch is not None:
        return f"{family}:{branch}"
    return family


def main() -> int:
    """Rank candidate checkpoints by validation quality and prediction diversity."""
    args = parse_args()
    manifest = load_yaml(args.manifest)
    candidates = manifest.get("candidates", [])
    if not candidates:
        raise ValueError("Manifest must contain a non-empty 'candidates' list.")
    require_branch_provenance = bool(manifest.get("require_branch_provenance", False))
    treat_branch_variants_as_distinct = bool(
        manifest.get("treat_branch_variants_as_distinct", False)
    )
    threshold_cfg = manifest.get("thresholds", {})
    val_gap_tolerance = float(
        args.val_gap_tolerance
        if args.val_gap_tolerance is not None
        else threshold_cfg.get("val_gap_tolerance", 1.0)
    )
    min_val_disagreement = float(
        args.min_val_disagreement
        if args.min_val_disagreement is not None
        else threshold_cfg.get("min_val_disagreement", 0.02)
    )
    min_test_disagreement = float(
        args.min_test_disagreement
        if args.min_test_disagreement is not None
        else threshold_cfg.get("min_test_disagreement", 0.02)
    )
    min_rescue_count = int(
        args.min_rescue_count
        if args.min_rescue_count is not None
        else threshold_cfg.get("min_rescue_count", 1)
    )
    min_js_divergence = float(
        args.min_js_divergence
        if args.min_js_divergence is not None
        else threshold_cfg.get("min_js_divergence", 0.0)
    )
    max_per_family = int(
        args.max_per_family
        if args.max_per_family is not None
        else threshold_cfg.get("max_per_family", 1)
    )

    metric_name = args.metric or manifest.get("metric", "val_acc")
    if metric_name not in _METRIC_ALIASES:
        raise ValueError(f"Unsupported metric '{metric_name}'.")
    metric_mode = _metric_mode(metric_name, args.metric_mode)

    anchor_name = args.anchor or manifest.get("anchor")
    if not anchor_name:
        raise ValueError("Manifest must define 'anchor' or --anchor must be provided.")

    candidate_map = {entry["name"]: entry for entry in candidates}
    if anchor_name not in candidate_map:
        raise ValueError(f"Anchor '{anchor_name}' not found in manifest candidates.")

    output_dir = ensure_dir(args.output_dir)

    loaded_candidates = []
    best_metric_value = None
    for entry in candidates:
        val_metrics_path = _resolve_path(entry.get("val_metrics"))
        val_artifact_path = _resolve_path(entry.get("val_artifact"))
        if val_metrics_path is None or val_artifact_path is None:
            raise ValueError(
                f"Candidate '{entry['name']}' requires both 'val_metrics' and 'val_artifact'."
            )

        family = str(entry.get("family", entry["name"]))
        branch = _normalize_branch(entry.get("branch"))
        val_metric_value = _load_metric_value(val_metrics_path, metric_name)
        val_artifact = load_artifact(val_artifact_path)
        test_artifact_path = _resolve_path(entry.get("test_artifact"))
        test_artifact = (
            load_artifact(test_artifact_path) if test_artifact_path is not None else None
        )
        loaded_candidates.append(
            {
                "name": entry["name"],
                "family": family,
                "branch": branch,
                "selection_group": _selection_group_name(
                    entry=entry,
                    family=family,
                    branch=branch,
                    treat_branch_variants_as_distinct=treat_branch_variants_as_distinct,
                ),
                "branch_provenance_explicit": branch is not None,
                "val_metric": val_metric_value,
                "val_artifact": val_artifact,
                "test_artifact": test_artifact,
            }
        )
        if best_metric_value is None:
            best_metric_value = val_metric_value
        elif metric_mode == "max":
            best_metric_value = max(best_metric_value, val_metric_value)
        else:
            best_metric_value = min(best_metric_value, val_metric_value)

    anchor_entry = next(entry for entry in loaded_candidates if entry["name"] == anchor_name)
    anchor_val = anchor_entry["val_artifact"]
    anchor_test = anchor_entry["test_artifact"]

    rows = []
    for entry in loaded_candidates:
        val_indices = _aligned_indices(anchor_val.sample_ids, entry["val_artifact"].sample_ids)
        candidate_val_preds = entry["val_artifact"].preds[val_indices]
        candidate_val_targets = (
            entry["val_artifact"].targets[val_indices]
            if entry["val_artifact"].targets is not None
            else None
        )
        if anchor_val.targets is None and candidate_val_targets is None:
            raise ValueError(
                "Val artifacts must contain ground-truth targets for rescue-count computation."
            )
        val_targets = (
            anchor_val.targets if anchor_val.targets is not None else candidate_val_targets
        )

        if candidate_val_targets is not None and anchor_val.targets is not None:
            if not np.array_equal(candidate_val_targets, anchor_val.targets):
                raise ValueError(f"Val targets mismatch for candidate '{entry['name']}'.")

        candidate_val_probs = (
            entry["val_artifact"].probs[val_indices]
            if entry["val_artifact"].probs is not None
            else None
        )
        val_disagreement = float(np.mean(anchor_val.preds != candidate_val_preds))
        rescue_count = int(
            np.sum((anchor_val.preds != val_targets) & (candidate_val_preds == val_targets))
        )
        val_js_divergence = _js_divergence(anchor_val.probs, candidate_val_probs)

        test_disagreement = None
        test_js_divergence = None
        if anchor_test is not None and entry["test_artifact"] is not None:
            test_indices = _aligned_indices(
                anchor_test.sample_ids, entry["test_artifact"].sample_ids
            )
            candidate_test_preds = entry["test_artifact"].preds[test_indices]
            candidate_test_probs = (
                entry["test_artifact"].probs[test_indices]
                if entry["test_artifact"].probs is not None
                else None
            )
            test_disagreement = float(np.mean(anchor_test.preds != candidate_test_preds))
            test_js_divergence = _js_divergence(anchor_test.probs, candidate_test_probs)

        if metric_mode == "max":
            val_gap_to_best = float(best_metric_value - entry["val_metric"])
        else:
            val_gap_to_best = float(entry["val_metric"] - best_metric_value)

        threshold_failures = []
        if val_gap_to_best > val_gap_tolerance:
            threshold_failures.append("val_gap")
        if entry["name"] != anchor_name and val_disagreement < min_val_disagreement:
            threshold_failures.append("val_disagreement")
        if (
            entry["name"] != anchor_name
            and test_disagreement is not None
            and test_disagreement < min_test_disagreement
        ):
            threshold_failures.append("test_disagreement")
        if entry["name"] != anchor_name and rescue_count < min_rescue_count:
            threshold_failures.append("rescue_count")
        if (
            entry["name"] != anchor_name
            and val_js_divergence is not None
            and val_js_divergence < min_js_divergence
        ):
            threshold_failures.append("val_js_divergence")
        if require_branch_provenance and entry["branch"] is None:
            threshold_failures.append("branch_provenance")

        rows.append(
            {
                "name": entry["name"],
                "family": entry["family"],
                "branch": entry["branch"],
                "selection_group": entry["selection_group"],
                "branch_provenance_explicit": entry["branch_provenance_explicit"],
                "is_anchor": entry["name"] == anchor_name,
                "val_metric_name": metric_name,
                "val_metric": float(entry["val_metric"]),
                "val_gap_to_best": val_gap_to_best,
                "val_disagreement": val_disagreement,
                "test_disagreement": test_disagreement,
                "rescue_count": rescue_count,
                "val_js_divergence": val_js_divergence,
                "test_js_divergence": test_js_divergence,
                "passes_thresholds": len(threshold_failures) == 0,
                "threshold_failures": ",".join(threshold_failures),
                "selected_for_ensemble": entry["name"] == anchor_name,
                "selection_reason": "anchor" if entry["name"] == anchor_name else "",
                "probe_status": "anchor" if entry["name"] == anchor_name else "review",
            }
        )

    ranked_rows = sorted(rows, key=_candidate_sort_key, reverse=True)
    group_counts: dict[str, int] = {}
    for row in ranked_rows:
        if row["is_anchor"]:
            continue
        if not row["passes_thresholds"]:
            row["selection_reason"] = row["threshold_failures"] or "thresholds"
            row["probe_status"] = "stop_probe"
            continue
        selection_group = row["selection_group"]
        group_count = group_counts.get(selection_group, 0)
        if group_count >= max_per_family:
            row["selection_reason"] = "near_duplicate_selection_group"
            row["probe_status"] = "hold_near_duplicate_group"
            continue
        row["selected_for_ensemble"] = True
        row["selection_reason"] = "selected"
        row["probe_status"] = "continue_probe"
        group_counts[selection_group] = group_count + 1

    ranked_df = pd.DataFrame(ranked_rows)
    ranked_df["rank"] = range(1, len(ranked_df) + 1)
    ranked_df.to_csv(output_dir / "diversity_summary.csv", index=False)

    report = {
        "anchor": anchor_name,
        "metric": metric_name,
        "metric_mode": metric_mode,
        "best_metric_value": best_metric_value,
        "thresholds": {
            "val_gap_tolerance": val_gap_tolerance,
            "min_val_disagreement": min_val_disagreement,
            "min_test_disagreement": min_test_disagreement,
            "min_rescue_count": min_rescue_count,
            "min_js_divergence": min_js_divergence,
            "max_per_family": max_per_family,
        },
        "require_branch_provenance": require_branch_provenance,
        "treat_branch_variants_as_distinct": treat_branch_variants_as_distinct,
        "selected_candidates": ranked_df.loc[ranked_df["selected_for_ensemble"], "name"].tolist(),
        "continued_probe_candidates": ranked_df.loc[
            ranked_df["probe_status"] == "continue_probe", "name"
        ].tolist(),
        "stopped_probe_candidates": ranked_df.loc[
            ranked_df["probe_status"] == "stop_probe", "name"
        ].tolist(),
        "held_probe_candidates": ranked_df.loc[
            ranked_df["probe_status"] == "hold_near_duplicate_group",
            "name",
        ].tolist(),
    }
    save_json(report, output_dir / "diversity_summary.json")
    print(f"Saved diversity summary to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
