"""Helpers for branch-aware validation and inference artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def infer_artifact_provenance(checkpoint_path: str | Path, use_ema: bool) -> Dict[str, Any]:
    """Infer a stable branch label from the checkpoint name and weight source."""
    checkpoint = Path(checkpoint_path)
    checkpoint_name = checkpoint.stem.lower()

    if checkpoint_name == "best_raw":
        branch = "raw"
    elif checkpoint_name == "best_ema":
        branch = "ema"
    elif checkpoint_name == "best":
        branch = "selected_ema" if use_ema else "selected_raw"
    elif checkpoint_name == "last":
        branch = "last_ema" if use_ema else "last_raw"
    else:
        branch = "ema" if use_ema else "raw"

    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_name": checkpoint_name,
        "branch": branch,
        "used_ema_weights": bool(use_ema),
        "branch_is_explicit": branch in {"raw", "ema"},
    }
