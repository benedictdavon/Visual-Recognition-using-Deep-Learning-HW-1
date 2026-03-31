"""Run metadata and stage-gate helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.utils.misc import save_json


def list_checkpoint_files(checkpoint_dir: Path | str) -> list[str]:
    """Return checkpoint paths sorted by filename."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []
    return [str(path) for path in sorted(checkpoint_dir.glob("*.ckpt"))]


def evaluate_stage_gate(
    stage_gate_cfg: Dict[str, Any],
    available_metrics: Dict[str, Any],
    default_metric: str,
    default_mode: str,
) -> Dict[str, Any]:
    """Evaluate configured stage-gate thresholds against available metrics."""
    metric_value = stage_gate_cfg.get("metric") if stage_gate_cfg else None
    pass_threshold = stage_gate_cfg.get("pass_threshold") if stage_gate_cfg else None
    fail_threshold = stage_gate_cfg.get("fail_threshold") if stage_gate_cfg else None
    if not stage_gate_cfg or (metric_value in {None, ""} and pass_threshold is None and fail_threshold is None):
        return {
            "status": "not_configured",
            "metric": default_metric,
            "mode": default_mode,
            "measured_value": available_metrics.get(default_metric),
        }

    metric = str(metric_value or default_metric)
    mode = str(stage_gate_cfg.get("mode", default_mode)).lower()
    if mode not in {"auto", "min", "max"}:
        raise ValueError(f"Unsupported stage gate mode: {mode}")
    if mode == "auto":
        mode = default_mode

    if metric not in available_metrics:
        raise ValueError(f"Stage gate metric '{metric}' is not available in run summary.")

    value = available_metrics[metric]
    status = "review"

    if value is None:
        status = "metric_missing"
    elif mode == "max":
        if pass_threshold is not None and value >= float(pass_threshold):
            status = "passed"
        elif fail_threshold is not None and value <= float(fail_threshold):
            status = "failed"
    else:
        if pass_threshold is not None and value <= float(pass_threshold):
            status = "passed"
        elif fail_threshold is not None and value >= float(fail_threshold):
            status = "failed"

    return {
        "status": status,
        "metric": metric,
        "mode": mode,
        "measured_value": value,
        "pass_threshold": float(pass_threshold) if pass_threshold is not None else None,
        "fail_threshold": float(fail_threshold) if fail_threshold is not None else None,
    }


def write_run_metadata(
    run_dir: Path | str,
    config: Dict[str, Any],
    summary: Dict[str, Any],
    checkpoint_files: list[str],
) -> None:
    """Write structured run metadata JSON beside the run summary."""
    run_dir = Path(run_dir)
    experiment_cfg = config.get("experiment", {})
    selected_metrics = summary.get("best_metrics_selected", {})
    stage_gate = evaluate_stage_gate(
        stage_gate_cfg=experiment_cfg.get("stage_gate", {}),
        available_metrics=selected_metrics,
        default_metric=str(summary.get("best_metric_name", "val_acc")),
        default_mode=str(summary.get("best_metric_mode", "max")),
    )
    selected_metric_source = str(summary.get("best_metric_source", "raw"))
    selected_branch = "ema" if selected_metric_source == "ema" else "raw"
    branch_candidates = []
    raw_checkpoint = summary.get("best_raw_checkpoint")
    if raw_checkpoint:
        branch_candidates.append(
            {
                "branch": "raw",
                "checkpoint": raw_checkpoint,
                "epoch": summary.get("best_raw_epoch"),
                "metrics": summary.get("best_raw_metrics", {}),
            }
        )
    ema_checkpoint = summary.get("best_ema_checkpoint")
    if ema_checkpoint:
        branch_candidates.append(
            {
                "branch": "ema",
                "checkpoint": ema_checkpoint,
                "epoch": summary.get("best_ema_epoch"),
                "metrics": summary.get("best_ema_metrics", {}),
            }
        )

    metadata = {
        "experiment": {
            "stage": experiment_cfg.get("stage"),
            "rationale": experiment_cfg.get("rationale"),
            "notes": experiment_cfg.get("notes"),
            "tags": experiment_cfg.get("tags", []),
            "expected_artifacts": experiment_cfg.get("expected_artifacts", []),
            "stage_gate": stage_gate,
        },
        "loss_runtime": summary.get("loss_runtime", config.get("loss_runtime", {})),
        "staged_training": summary.get("staged_training", config.get("staged_training_runtime", {})),
        "artifacts": {
            "config_snapshot": str(run_dir / "config.yaml"),
            "summary_json": str(run_dir / "summary.json"),
            "history_csv": summary.get("history_path"),
            "checkpoint_files": checkpoint_files,
        },
        "branch_candidates": branch_candidates,
        "selected_branch": {
            "branch": selected_branch,
            "checkpoint": summary.get("best_checkpoint"),
            "selection_metric_source": selected_metric_source,
            "selection_metric_name": summary.get("best_metric_name"),
            "selection_metric_value": summary.get("best_metric_value"),
        },
        "metrics_summary": summary,
    }
    save_json(metadata, run_dir / "run_metadata.json")
