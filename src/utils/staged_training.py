"""Helpers for staged training configuration and lineage tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

_STAGE_ALIASES = {
    "": "single_stage",
    "single": "single_stage",
    "single_stage": "single_stage",
    "ordinary": "single_stage",
    "base": "base",
    "anchor_base": "base",
    "classifier_rebalance": "classifier_rebalance",
    "rebalance": "classifier_rebalance",
    "crt": "classifier_rebalance",
    "fixres": "fixres_refresh",
    "fixres_refresh": "fixres_refresh",
    "fixres_refresh_after_rebalance": "fixres_refresh",
}

_SCOPE_ALIASES = {
    "": "full_model",
    "all": "full_model",
    "full": "full_model",
    "full_model": "full_model",
    "classifier": "classifier_only",
    "classifier_only": "classifier_only",
    "head_only": "classifier_only",
}


def normalize_stage_name(value: Any) -> str:
    """Normalize stage-name aliases into a stable internal value."""
    key = str(value or "single_stage").strip().lower()
    return _STAGE_ALIASES.get(key, key)


def normalize_trainable_scope(value: Any) -> str:
    """Normalize trainable-scope aliases into a stable internal value."""
    key = str(value or "full_model").strip().lower()
    normalized = _SCOPE_ALIASES.get(key)
    if normalized is None:
        raise ValueError(f"Unsupported staged-training trainable_scope: {value}")
    return normalized


def normalize_stage_list(value: Any) -> list[str]:
    """Normalize optional expected-parent-stage config into a list."""
    if value is None:
        return []
    if isinstance(value, str):
        raw_values = [value]
    elif isinstance(value, Iterable):
        raw_values = list(value)
    else:
        raise ValueError(f"Unsupported expected_parent_stages value: {value!r}")
    return [normalize_stage_name(item) for item in raw_values if str(item).strip()]


def resolve_optional_path(value: Any) -> str | None:
    """Resolve an optional filesystem path to an absolute string."""
    if value in {None, ""}:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path.resolve())


def infer_run_dir_from_checkpoint(checkpoint_path: str | None) -> str | None:
    """Infer the run directory from a checkpoint path if it follows the repo layout."""
    if checkpoint_path in {None, ""}:
        return None
    path = Path(str(checkpoint_path))
    if path.parent.name == "checkpoints":
        return str(path.parent.parent.resolve())
    return str(path.parent.resolve())


def extract_checkpoint_stage_runtime(checkpoint: Dict[str, Any] | None) -> Dict[str, Any]:
    """Return normalized staged-training runtime info from a checkpoint when available."""
    if not isinstance(checkpoint, dict):
        return {}

    runtime = checkpoint.get("staged_training_runtime")
    if isinstance(runtime, dict):
        return runtime

    parent_config = checkpoint.get("config", {})
    if not isinstance(parent_config, dict):
        return {}

    staged_cfg = parent_config.get("staged_training", {})
    if not isinstance(staged_cfg, dict):
        staged_cfg = {}

    stage_name = normalize_stage_name(staged_cfg.get("stage_name"))
    trainable_scope = normalize_trainable_scope(staged_cfg.get("trainable_scope"))
    parent_checkpoint = resolve_optional_path(staged_cfg.get("parent_checkpoint"))

    lineage_cfg = staged_cfg.get("lineage", {})
    if not isinstance(lineage_cfg, dict):
        lineage_cfg = {}

    return {
        "stage_name": stage_name,
        "trainable_scope": trainable_scope,
        "parent_checkpoint": parent_checkpoint,
        "parent_use_ema": bool(staged_cfg.get("parent_use_ema", False)),
        "parent_run_dir": resolve_optional_path(staged_cfg.get("parent_run_dir"))
        or infer_run_dir_from_checkpoint(parent_checkpoint),
        "experiment_name": parent_config.get("project", {}).get("experiment_name"),
        "lineage": {
            "base_checkpoint": resolve_optional_path(lineage_cfg.get("base_checkpoint")),
            "base_run_dir": resolve_optional_path(lineage_cfg.get("base_run_dir")),
            "rebalance_checkpoint": resolve_optional_path(lineage_cfg.get("rebalance_checkpoint")),
            "rebalance_run_dir": resolve_optional_path(lineage_cfg.get("rebalance_run_dir")),
        },
    }


def build_stage_runtime(
    staged_cfg: Dict[str, Any],
    parent_checkpoint: Dict[str, Any] | None,
    trainable_param_names: list[str],
    trainable_param_count: int,
    frozen_param_count: int,
) -> Dict[str, Any]:
    """Build the runtime stage block persisted into summaries and checkpoints."""
    parent_runtime = extract_checkpoint_stage_runtime(parent_checkpoint)

    stage_name = normalize_stage_name(staged_cfg.get("stage_name"))
    parent_checkpoint_path = resolve_optional_path(staged_cfg.get("parent_checkpoint"))
    parent_run_dir = resolve_optional_path(
        staged_cfg.get("parent_run_dir")
    ) or infer_run_dir_from_checkpoint(parent_checkpoint_path)
    parent_experiment_name = parent_runtime.get("experiment_name")
    parent_stage_name = parent_runtime.get("stage_name")

    parent_lineage = (
        parent_runtime.get("lineage", {}) if isinstance(parent_runtime.get("lineage"), dict) else {}
    )
    explicit_lineage = staged_cfg.get("lineage", {})
    if not isinstance(explicit_lineage, dict):
        explicit_lineage = {}

    base_checkpoint = resolve_optional_path(
        explicit_lineage.get("base_checkpoint")
    ) or parent_lineage.get("base_checkpoint")
    base_run_dir = resolve_optional_path(
        explicit_lineage.get("base_run_dir")
    ) or parent_lineage.get("base_run_dir")
    rebalance_checkpoint = resolve_optional_path(
        explicit_lineage.get("rebalance_checkpoint")
    ) or parent_lineage.get("rebalance_checkpoint")
    rebalance_run_dir = resolve_optional_path(
        explicit_lineage.get("rebalance_run_dir")
    ) or parent_lineage.get("rebalance_run_dir")

    if parent_checkpoint_path and parent_stage_name in {None, "", "single_stage", "base"}:
        base_checkpoint = base_checkpoint or parent_checkpoint_path
        base_run_dir = base_run_dir or parent_run_dir
    if parent_checkpoint_path and parent_stage_name == "classifier_rebalance":
        rebalance_checkpoint = rebalance_checkpoint or parent_checkpoint_path
        rebalance_run_dir = rebalance_run_dir or parent_run_dir
        base_checkpoint = base_checkpoint or parent_lineage.get("base_checkpoint")
        base_run_dir = base_run_dir or parent_lineage.get("base_run_dir")

    return {
        "stage_name": stage_name,
        "trainable_scope": normalize_trainable_scope(staged_cfg.get("trainable_scope")),
        "expected_parent_stages": normalize_stage_list(staged_cfg.get("expected_parent_stages")),
        "parent_checkpoint": parent_checkpoint_path,
        "parent_use_ema": bool(staged_cfg.get("parent_use_ema", False))
        if parent_checkpoint_path
        else False,
        "parent_run_dir": parent_run_dir,
        "parent_experiment_name": parent_experiment_name,
        "parent_stage_name": parent_stage_name,
        "trainable_parameter_names": trainable_param_names,
        "trainable_parameter_count": int(trainable_param_count),
        "frozen_parameter_count": int(frozen_param_count),
        "lineage": {
            "base_checkpoint": base_checkpoint,
            "base_run_dir": base_run_dir,
            "rebalance_checkpoint": rebalance_checkpoint,
            "rebalance_run_dir": rebalance_run_dir,
        },
    }
