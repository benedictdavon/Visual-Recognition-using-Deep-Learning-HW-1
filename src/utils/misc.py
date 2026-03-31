"""General utility helpers."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


def ensure_dir(path: Path | str) -> Path:
    """Create a directory if needed and return it as Path."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_yaml(path: Path | str) -> Dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_yaml(data: Dict[str, Any], path: Path | str) -> None:
    """Save a dictionary to YAML."""
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_json(data: Dict[str, Any], path: Path | str) -> None:
    """Save a dictionary to JSON."""
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge updates into base and return a new dictionary."""
    merged = deepcopy(base)
    for key, value in updates.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def merge_yaml_configs(
    base_config_path: Path | str,
    extra_config_paths: Optional[Iterable[Path | str]] = None,
) -> Dict[str, Any]:
    """Load and merge YAML files. Later files override earlier values."""
    cfg = load_yaml(base_config_path)
    for path in extra_config_paths or []:
        cfg = deep_update(cfg, load_yaml(path))
    return cfg


def resolve_path(root: Path | str, maybe_relative: str | Path | None) -> Optional[Path]:
    """Resolve config path values against a root directory."""
    if maybe_relative is None:
        return None
    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    return Path(root) / path


def create_run_dir(output_root: Path | str, experiment_name: str) -> Path:
    """Create and return a timestamped run directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{timestamp}_{experiment_name}"
    return ensure_dir(run_dir)
