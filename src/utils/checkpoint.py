"""Checkpoint utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from src.utils.misc import ensure_dir


def save_checkpoint(state: Dict[str, Any], path: Path | str) -> None:
    """Persist checkpoint to disk."""
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    torch.save(state, path_obj)


def load_checkpoint(path: Path | str, map_location: str = "cpu") -> Dict[str, Any]:
    """Load checkpoint from disk."""
    return torch.load(Path(path), map_location=map_location)


def load_model_weights(
    model: torch.nn.Module,
    checkpoint_path: Path | str,
    map_location: str = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """Load model weights from checkpoint and return full checkpoint data."""
    checkpoint = load_checkpoint(checkpoint_path, map_location=map_location)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=strict)
    return checkpoint


def initialize_model_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path | str,
    map_location: str = "cpu",
    strict: bool = True,
    use_ema: bool = False,
) -> Dict[str, Any]:
    """Initialize model weights from a checkpoint for fresh fine-tuning."""
    checkpoint = load_checkpoint(checkpoint_path, map_location=map_location)
    if use_ema and "ema_state_dict" in checkpoint:
        state_dict = checkpoint["ema_state_dict"]["ema_state_dict"]
    else:
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=strict)
    return checkpoint


def resume_training_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    checkpoint_path: Path | str,
    map_location: str = "cpu",
) -> Tuple[int, float, Dict[str, Any]]:
    """Resume model/optimizer/scheduler states and return metadata."""
    checkpoint = load_checkpoint(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_metric = float(checkpoint.get("best_val_acc", 0.0))
    return start_epoch, best_metric, checkpoint
