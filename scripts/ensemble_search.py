"""Greedy, NLL-aware ensemble search over probability artifacts."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.submission.make_submission import (  # noqa: E402
    build_prediction_dataframe,
    save_prediction_csv,
)
from src.utils.metrics import (  # noqa: E402
    compute_confusion_matrix,
    expected_calibration_error,
    macro_recall_from_confusion_matrix,
)
from src.utils.misc import ensure_dir, load_yaml, save_json  # noqa: E402


@dataclass
class CandidateArtifact:
    """Validation and test probabilities for one ensemble candidate."""

    name: str
    family: str
    branch: str | None
    selection_group: str
    val_metric: float
    val_sample_ids: np.ndarray
    val_targets: np.ndarray
    val_probs: np.ndarray
    test_sample_ids: np.ndarray | None
    test_probs: np.ndarray | None


@dataclass
class TrialResult:
    """One greedy-search trial over the current ensemble pool."""

    candidate: CandidateArtifact
    candidates: list[CandidateArtifact]
    weights: list[float]
    metrics: dict[str, float]


_METRIC_ALIASES = {
    "val_acc": ["val_acc", "acc1"],
    "val_macro_recall": ["val_macro_recall", "macro_recall", "macro_per_class_acc"],
    "val_nll": ["val_nll", "nll"],
    "val_ece": ["val_ece", "ece"],
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for greedy ensemble search."""
    parser = argparse.ArgumentParser(description="Greedy NLL-aware ensemble search.")
    parser.add_argument(
        "--manifest", type=str, required=True, help="YAML manifest describing candidates."
    )
    parser.add_argument("--output-dir", type=str, default="outputs/ensemble_search")
    parser.add_argument(
        "--anchor", type=str, required=True, help="Anchor candidate name to lock into the pool."
    )
    parser.add_argument(
        "--diversity-summary", type=str, default=None, help="Optional diversity_summary.csv filter."
    )
    parser.add_argument("--max-pool-size", type=int, default=4)
    parser.add_argument("--weight-grid", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--min-nll-gain", type=float, default=1e-4)
    parser.add_argument("--max-acc-drop", type=float, default=0.5)
    parser.add_argument("--id-column", type=str, default="image_name")
    parser.add_argument("--target-column", type=str, default="pred_label")
    parser.add_argument(
        "--use-label-name",
        dest="use_label_name",
        action="store_true",
        help="Map indices back to original labels when saving prediction.csv.",
    )
    parser.add_argument(
        "--no-use-label-name",
        dest="use_label_name",
        action="store_false",
        help="Write raw class indices in prediction.csv.",
    )
    parser.set_defaults(use_label_name=True)
    parser.add_argument("--idx-to-label", type=str, default=None)
    return parser.parse_args()


def _resolve_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _softmax_numpy(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _load_prob_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    arr = np.load(path, allow_pickle=True)
    if "sample_ids" not in arr:
        raise ValueError(f"{path} is missing 'sample_ids'.")
    sample_ids = arr["sample_ids"].astype(str)
    if "probs" in arr:
        probs = arr["probs"].astype(np.float64)
    elif "logits" in arr:
        probs = _softmax_numpy(arr["logits"].astype(np.float64))
    else:
        raise ValueError(f"{path} is missing both 'probs' and 'logits'.")
    targets = arr["targets"].astype(np.int64) if "targets" in arr else None
    return sample_ids, probs, targets


def _aligned_order(anchor_ids: np.ndarray, candidate_ids: np.ndarray) -> np.ndarray:
    lookup = {sample_id: idx for idx, sample_id in enumerate(candidate_ids.tolist())}
    try:
        indices = np.array([lookup[sample_id] for sample_id in anchor_ids.tolist()], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(
            f"Candidate artifact is missing sample_id '{exc.args[0]}' required by the anchor."
        ) from exc
    if len(lookup) != len(anchor_ids):
        raise ValueError("All candidate artifacts must cover the same sample set.")
    return indices


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


def _load_idx_to_label(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): str(v) for k, v in data.items()}


def _resolve_idx_to_label(
    manifest_candidates: list[dict], override_path: str | None
) -> dict[int, str] | None:
    if override_path:
        return _load_idx_to_label(_resolve_path(override_path))
    for entry in manifest_candidates:
        test_artifact = _resolve_path(entry.get("test_artifact"))
        if test_artifact is None:
            continue
        candidate = test_artifact.parent / "idx_to_label.json"
        if candidate.exists():
            return _load_idx_to_label(candidate)
    return None


def _compute_metrics(probs: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    preds = np.argmax(probs, axis=1)
    acc = float(np.mean(preds == targets) * 100.0)
    conf_mat = compute_confusion_matrix(
        targets.tolist(), preds.tolist(), num_classes=probs.shape[1]
    )
    macro_recall = macro_recall_from_confusion_matrix(conf_mat, as_percentage=True)
    nll = float(-np.log(np.clip(probs[np.arange(len(targets)), targets], 1e-12, 1.0)).mean())
    ece = expected_calibration_error(probs, targets)
    return {
        "val_acc": acc,
        "val_macro_recall": macro_recall,
        "val_nll": nll,
        "val_ece": ece,
    }


def _is_better(candidate_metrics: dict[str, float], best_metrics: dict[str, float]) -> bool:
    if candidate_metrics["val_nll"] < best_metrics["val_nll"] - 1e-12:
        return True
    if candidate_metrics["val_nll"] > best_metrics["val_nll"] + 1e-12:
        return False
    if candidate_metrics["val_acc"] > best_metrics["val_acc"] + 1e-12:
        return True
    if candidate_metrics["val_acc"] < best_metrics["val_acc"] - 1e-12:
        return False
    if candidate_metrics["val_macro_recall"] > best_metrics["val_macro_recall"] + 1e-12:
        return True
    if candidate_metrics["val_macro_recall"] < best_metrics["val_macro_recall"] - 1e-12:
        return False
    return candidate_metrics["val_ece"] < best_metrics["val_ece"] - 1e-12


def _fuse_probs(
    candidates: list[CandidateArtifact], weights: tuple[float, ...], use_test: bool
) -> tuple[np.ndarray, np.ndarray]:
    sample_ids = candidates[0].test_sample_ids if use_test else candidates[0].val_sample_ids
    probs = np.zeros_like(
        candidates[0].test_probs if use_test else candidates[0].val_probs, dtype=np.float64
    )
    for candidate, weight in zip(candidates, weights):
        source_probs = candidate.test_probs if use_test else candidate.val_probs
        if source_probs is None:
            raise ValueError(
                f"Candidate '{candidate.name}' is missing {'test' if use_test else 'val'} probabilities."
            )
        probs += source_probs * float(weight)
    probs /= float(sum(weights))
    return sample_ids, probs


def _search_best_weights(
    candidates: list[CandidateArtifact], weight_grid: list[float]
) -> tuple[list[float], dict[str, float]]:
    best_weights = [1.0] * len(candidates)
    _, baseline_probs = _fuse_probs(candidates, tuple(best_weights), use_test=False)
    best_metrics = _compute_metrics(baseline_probs, candidates[0].val_targets)
    for weights in itertools.product(weight_grid, repeat=len(candidates)):
        if sum(weights) <= 0:
            continue
        _, fused_probs = _fuse_probs(candidates, weights, use_test=False)
        metrics = _compute_metrics(fused_probs, candidates[0].val_targets)
        if _is_better(metrics, best_metrics):
            best_weights = list(weights)
            best_metrics = metrics
    return best_weights, best_metrics


def _load_candidates(
    manifest: dict, anchor_name: str, diversity_summary: Path | None
) -> list[CandidateArtifact]:
    selected_names = None
    if diversity_summary is not None:
        df = pd.read_csv(diversity_summary)
        if "name" not in df.columns or "selected_for_ensemble" not in df.columns:
            raise ValueError(
                "diversity_summary.csv must contain 'name' and 'selected_for_ensemble' columns."
            )
        selected_names = set(df.loc[df["selected_for_ensemble"], "name"].astype(str).tolist())
        selected_names.add(anchor_name)

    candidates = manifest.get("candidates", [])
    if not candidates:
        raise ValueError("Manifest must contain a non-empty 'candidates' list.")

    loaded = []
    anchor_ids = None
    anchor_test_ids = None
    for entry in candidates:
        name = str(entry["name"])
        if selected_names is not None and name not in selected_names:
            continue
        val_metrics_path = _resolve_path(entry.get("val_metrics"))
        val_artifact_path = _resolve_path(entry.get("val_artifact"))
        if val_metrics_path is None or val_artifact_path is None:
            raise ValueError(f"Candidate '{name}' requires 'val_metrics' and 'val_artifact'.")

        val_ids, val_probs, val_targets = _load_prob_npz(val_artifact_path)
        if val_targets is None:
            raise ValueError(f"Candidate '{name}' val artifact must contain targets.")
        test_artifact_path = _resolve_path(entry.get("test_artifact"))
        if test_artifact_path is not None:
            test_ids, test_probs, _ = _load_prob_npz(test_artifact_path)
        else:
            test_ids, test_probs = None, None

        family = str(entry.get("family", name))
        branch = (
            str(entry["branch"]).strip().lower() if entry.get("branch") not in {None, ""} else None
        )
        selection_group = str(
            entry.get("selection_group") or (f"{family}:{branch}" if branch else family)
        )
        candidate = CandidateArtifact(
            name=name,
            family=family,
            branch=branch,
            selection_group=selection_group,
            val_metric=_load_metric_value(val_metrics_path, "val_acc"),
            val_sample_ids=val_ids,
            val_targets=val_targets,
            val_probs=val_probs,
            test_sample_ids=test_ids,
            test_probs=test_probs,
        )
        loaded.append(candidate)
        if name == anchor_name:
            anchor_ids = val_ids
            anchor_test_ids = test_ids

    if anchor_ids is None:
        raise ValueError(f"Anchor '{anchor_name}' was not loaded from the manifest.")

    aligned = []
    for candidate in loaded:
        val_order = _aligned_order(anchor_ids, candidate.val_sample_ids)
        val_targets = candidate.val_targets[val_order]
        if not np.array_equal(
            val_targets, loaded[[c.name for c in loaded].index(anchor_name)].val_targets
        ):
            raise ValueError(f"Val targets mismatch for candidate '{candidate.name}'.")

        test_ids = candidate.test_sample_ids
        test_probs = candidate.test_probs
        if anchor_test_ids is not None and test_ids is not None:
            test_order = _aligned_order(anchor_test_ids, test_ids)
            test_ids = test_ids[test_order]
            test_probs = test_probs[test_order] if test_probs is not None else None

        aligned.append(
            CandidateArtifact(
                name=candidate.name,
                family=candidate.family,
                branch=candidate.branch,
                selection_group=candidate.selection_group,
                val_metric=candidate.val_metric,
                val_sample_ids=candidate.val_sample_ids[val_order],
                val_targets=val_targets,
                val_probs=candidate.val_probs[val_order],
                test_sample_ids=test_ids,
                test_probs=test_probs,
            )
        )
    return aligned


def main() -> int:
    """Search a small weighted ensemble pool around a fixed anchor model."""
    args = parse_args()
    if args.max_pool_size <= 0:
        raise ValueError("--max-pool-size must be positive.")
    if not args.weight_grid:
        raise ValueError("--weight-grid must contain at least one value.")

    manifest = load_yaml(args.manifest)
    diversity_summary = _resolve_path(args.diversity_summary) if args.diversity_summary else None
    candidates = _load_candidates(
        manifest, anchor_name=args.anchor, diversity_summary=diversity_summary
    )
    candidate_lookup = {candidate.name: candidate for candidate in candidates}
    if args.anchor not in candidate_lookup:
        raise ValueError(f"Anchor '{args.anchor}' is missing after candidate loading/filtering.")

    selected = [candidate_lookup[args.anchor]]
    selected_weights = [1.0]
    _, anchor_val_probs = _fuse_probs(selected, tuple(selected_weights), use_test=False)
    current_metrics = _compute_metrics(anchor_val_probs, selected[0].val_targets)
    search_trace = [
        {
            "step": 0,
            "action": "anchor",
            "selected_models": [selected[0].name],
            "weights": selected_weights,
            **current_metrics,
        }
    ]

    remaining = [candidate for candidate in candidates if candidate.name != args.anchor]
    while remaining and len(selected) < args.max_pool_size:
        best_trial: TrialResult | None = None
        for candidate in remaining:
            trial_candidates = selected + [candidate]
            trial_weights, trial_metrics = _search_best_weights(trial_candidates, args.weight_grid)
            trial_record = TrialResult(
                candidate=candidate,
                candidates=trial_candidates,
                weights=trial_weights,
                metrics=trial_metrics,
            )
            if best_trial is None or _is_better(trial_metrics, best_trial.metrics):
                best_trial = trial_record

        if best_trial is None:
            break

        nll_gain = current_metrics["val_nll"] - best_trial.metrics["val_nll"]
        acc_drop = current_metrics["val_acc"] - best_trial.metrics["val_acc"]
        if nll_gain < args.min_nll_gain:
            search_trace.append(
                {
                    "step": len(search_trace),
                    "action": "stop_no_nll_gain",
                    "candidate": best_trial.candidate.name,
                    "selected_models": [candidate.name for candidate in selected],
                    "weights": selected_weights,
                    **current_metrics,
                }
            )
            break
        if acc_drop > args.max_acc_drop:
            search_trace.append(
                {
                    "step": len(search_trace),
                    "action": "stop_acc_drop_guard",
                    "candidate": best_trial.candidate.name,
                    "selected_models": [candidate.name for candidate in selected],
                    "weights": selected_weights,
                    **current_metrics,
                }
            )
            break

        selected = best_trial.candidates
        selected_weights = best_trial.weights
        current_metrics = best_trial.metrics
        remaining = [
            candidate for candidate in remaining if candidate.name != best_trial.candidate.name
        ]
        search_trace.append(
            {
                "step": len(search_trace),
                "action": "add_candidate",
                "candidate": best_trial.candidate.name,
                "selected_models": [candidate.name for candidate in selected],
                "weights": selected_weights,
                **current_metrics,
            }
        )

    output_dir = ensure_dir(args.output_dir)
    pd.DataFrame(search_trace).to_csv(output_dir / "search_trace.csv", index=False)

    summary = {
        "anchor": args.anchor,
        "selected_models": [candidate.name for candidate in selected],
        "selected_families": [candidate.family for candidate in selected],
        "selected_branches": [candidate.branch for candidate in selected],
        "selected_selection_groups": [candidate.selection_group for candidate in selected],
        "weights": selected_weights,
        "val_metrics": current_metrics,
        "max_pool_size": args.max_pool_size,
        "weight_grid": args.weight_grid,
        "min_nll_gain": args.min_nll_gain,
        "max_acc_drop": args.max_acc_drop,
        "diversity_summary": str(diversity_summary) if diversity_summary is not None else None,
    }

    can_write_prediction = all(candidate.test_probs is not None for candidate in selected)
    if can_write_prediction:
        idx_to_label = _resolve_idx_to_label(manifest.get("candidates", []), args.idx_to_label)
        if args.use_label_name and idx_to_label is None:
            raise ValueError(
                "prediction.csv output with label names requires idx_to_label.json or --idx-to-label."
            )
        test_ids, fused_test_probs = _fuse_probs(selected, tuple(selected_weights), use_test=True)
        pred_indices = np.argmax(fused_test_probs, axis=1).tolist()
        submission_df = build_prediction_dataframe(
            sample_ids=test_ids.tolist(),
            pred_indices=pred_indices,
            id_column=args.id_column,
            target_column=args.target_column,
            idx_to_label=idx_to_label,
            use_label_name=args.use_label_name,
        )
        prediction_path = save_prediction_csv(
            submission_df, output_dir=output_dir, filename="prediction.csv"
        )
        np.savez_compressed(
            output_dir / "ensemble_test_probs_with_ids.npz",
            sample_ids=test_ids,
            probs=fused_test_probs,
        )
        summary["prediction_csv"] = str(prediction_path)
    else:
        summary["prediction_csv"] = None

    save_json(summary, output_dir / "ensemble_search_summary.json")
    print(f"Saved ensemble search outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
