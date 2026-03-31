"""Loss definitions and builder."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class BalancedSoftmaxLoss(nn.Module):
    """Balanced Softmax for single-label multi-class classification."""

    def __init__(
        self,
        class_counts: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if class_counts.ndim != 1:
            raise ValueError("class_counts must be a 1D tensor.")
        if torch.any(class_counts <= 0):
            raise ValueError("Balanced Softmax requires strictly positive class_counts for every class.")
        self.register_buffer("log_class_counts", class_counts.float().log())
        self.weight = weight
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adjusted_logits = logits + self.log_class_counts.unsqueeze(0)
        return F.cross_entropy(
            adjusted_logits,
            targets,
            weight=self.weight,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )


class LogitAdjustedCrossEntropyLoss(nn.Module):
    """Cross-entropy with train-prior logit adjustment."""

    def __init__(
        self,
        class_counts: torch.Tensor,
        tau: float = 1.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if class_counts.ndim != 1:
            raise ValueError("class_counts must be a 1D tensor.")
        if torch.any(class_counts <= 0):
            raise ValueError("Logit-adjusted CE requires strictly positive class_counts for every class.")
        priors = class_counts.float() / class_counts.sum()
        self.register_buffer("log_priors", priors.log())
        self.tau = float(tau)
        self.weight = weight
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adjusted_logits = logits + (self.tau * self.log_priors.unsqueeze(0))
        return F.cross_entropy(
            adjusted_logits,
            targets,
            weight=self.weight,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin loss with optional deferred reweighting."""

    def __init__(
        self,
        class_counts: torch.Tensor,
        max_margin: float = 0.5,
        scale: float = 30.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        deferred_reweighting: Optional[Dict[str, Any]] = None,
        total_epochs: Optional[int] = None,
        current_epoch: int = 0,
    ) -> None:
        super().__init__()
        if class_counts.ndim != 1:
            raise ValueError("class_counts must be a 1D tensor.")
        if torch.any(class_counts <= 0):
            raise ValueError("LDAM requires strictly positive class_counts for every class.")
        if max_margin <= 0:
            raise ValueError("loss.ldam_max_margin must be positive.")
        if scale <= 0:
            raise ValueError("loss.ldam_scale must be positive.")

        margins = 1.0 / torch.sqrt(torch.sqrt(class_counts.float()))
        margins = margins * (float(max_margin) / margins.max())
        self.register_buffer("margins", margins)

        self.max_margin = float(max_margin)
        self.scale = float(scale)
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction
        self.total_epochs = int(total_epochs) if total_epochs is not None else None

        if weight is not None and deferred_reweighting and deferred_reweighting.get("enabled", False):
            raise ValueError(
                "loss.class_weights cannot be combined with loss.deferred_reweighting.enabled for LDAM."
            )

        if weight is None:
            self.register_buffer("static_class_weights", torch.empty(0, dtype=torch.float))
        else:
            self.register_buffer("static_class_weights", weight.detach().float().clone())

        if deferred_reweighting and deferred_reweighting.get("enabled", False):
            drw_weights = _compute_deferred_reweighting_weights(
                class_counts=class_counts,
                power=float(deferred_reweighting["power"]),
                normalize=str(deferred_reweighting["normalize"]),
            )
        else:
            drw_weights = torch.empty(0, dtype=torch.float)
        self.register_buffer("drw_class_weights", drw_weights)

        self.drw_enabled = bool(deferred_reweighting and deferred_reweighting.get("enabled", False))
        self.drw_start_epoch = (
            int(deferred_reweighting["start_epoch"])
            if deferred_reweighting and deferred_reweighting.get("enabled", False)
            else None
        )
        self.drw_power = (
            float(deferred_reweighting["power"])
            if deferred_reweighting and deferred_reweighting.get("enabled", False)
            else None
        )
        self.drw_normalize = (
            str(deferred_reweighting["normalize"])
            if deferred_reweighting and deferred_reweighting.get("enabled", False)
            else None
        )
        self.current_epoch = 1
        self.drw_active = False
        self.set_epoch(current_epoch)

    def set_epoch(self, epoch_index: int) -> None:
        """Update the current epoch so deferred reweighting can switch on at the boundary."""
        epoch_index = int(epoch_index)
        if epoch_index < 0:
            raise ValueError("current_epoch must be non-negative.")
        if self.total_epochs is not None and epoch_index >= self.total_epochs:
            raise ValueError(
                f"current_epoch {epoch_index} is out of range for total_epochs={self.total_epochs}."
            )

        self.current_epoch = epoch_index + 1
        self.drw_active = bool(self.drw_enabled and self.current_epoch >= int(self.drw_start_epoch))

    def _active_class_weights(self) -> Optional[torch.Tensor]:
        if self.drw_active and self.drw_class_weights.numel() > 0:
            return self.drw_class_weights
        if self.static_class_weights.numel() > 0:
            return self.static_class_weights
        return None

    def get_runtime_metadata(self) -> Dict[str, Any]:
        """Return structured loss metadata for summaries and run artifacts."""
        active_weights = self._active_class_weights()
        return {
            "loss_name": "ldam",
            "loss_family": "ldam",
            "ldam_max_margin": self.max_margin,
            "ldam_scale": self.scale,
            "label_smoothing": self.label_smoothing,
            "class_weight_source": (
                "deferred_reweighting"
                if self.drw_active and self.drw_class_weights.numel() > 0
                else "static"
                if self.static_class_weights.numel() > 0
                else "none"
            ),
            "active_class_weight_min": (
                float(active_weights.min().item()) if active_weights is not None else None
            ),
            "active_class_weight_max": (
                float(active_weights.max().item()) if active_weights is not None else None
            ),
            "deferred_reweighting": {
                "enabled": self.drw_enabled,
                "active": self.drw_active,
                "start_epoch": self.drw_start_epoch,
                "activation_epoch": self.drw_start_epoch,
                "power": self.drw_power,
                "normalize": self.drw_normalize,
            },
        }

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim != 1:
            raise ValueError("LDAM expects hard 1D targets.")
        if logits.ndim != 2:
            raise ValueError("LDAM expects logits shaped [batch_size, num_classes].")

        margins = self.margins[targets].unsqueeze(1)
        target_mask = torch.zeros_like(logits, dtype=torch.bool)
        target_mask.scatter_(1, targets.view(-1, 1), True)
        adjusted_logits = torch.where(target_mask, logits - margins, logits) * self.scale
        return F.cross_entropy(
            adjusted_logits,
            targets,
            weight=self._active_class_weights(),
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )


class SoftTargetCrossEntropy(nn.Module):
    """Cross-entropy for soft labels (e.g., mixup/cutmix)."""

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        return torch.sum(-targets * log_probs, dim=-1).mean()


def _validate_class_counts(
    class_counts: Optional[torch.Tensor],
    num_classes: int,
    loss_name: str,
) -> torch.Tensor:
    if class_counts is None:
        raise ValueError(f"Loss '{loss_name}' requires class_counts but none were provided.")
    if class_counts.numel() != num_classes:
        raise ValueError(
            f"class_counts length {class_counts.numel()} does not match num_classes {num_classes}."
        )
    if class_counts.ndim != 1:
        raise ValueError("class_counts must be a 1D tensor.")
    if torch.any(class_counts <= 0):
        raise ValueError(f"Loss '{loss_name}' requires strictly positive class counts for all classes.")
    return class_counts


def _validate_deferred_reweighting(
    loss_cfg: Dict[str, Any],
    loss_name: str,
    total_epochs: Optional[int],
) -> Dict[str, Any]:
    drw_cfg = loss_cfg.get("deferred_reweighting", {})
    if drw_cfg is None:
        drw_cfg = {}
    if not isinstance(drw_cfg, dict):
        raise ValueError("loss.deferred_reweighting must be a mapping when provided.")

    enabled = bool(drw_cfg.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "start_epoch": None,
            "power": None,
            "normalize": None,
        }

    if loss_name != "ldam":
        raise ValueError("loss.deferred_reweighting is only supported when loss.name is 'ldam'.")
    if total_epochs is None:
        raise ValueError("LDAM-DRW requires total_epochs so the late-stage boundary can be validated.")

    total_epochs = int(total_epochs)
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive when deferred reweighting is enabled.")

    start_epoch = drw_cfg.get("start_epoch")
    if start_epoch is None:
        raise ValueError(
            "loss.deferred_reweighting.start_epoch is required when deferred reweighting is enabled."
        )
    start_epoch = int(start_epoch)
    if start_epoch <= 1 or start_epoch > total_epochs:
        raise ValueError(
            "loss.deferred_reweighting.start_epoch must fall in [2, total_epochs] "
            f"for LDAM-DRW; got start_epoch={start_epoch}, total_epochs={total_epochs}."
        )

    power = float(drw_cfg.get("power", 1.0))
    if power < 0:
        raise ValueError("loss.deferred_reweighting.power must be non-negative.")

    normalize = str(drw_cfg.get("normalize", "mean_one")).strip().lower()
    if normalize not in {"none", "mean_one"}:
        raise ValueError("loss.deferred_reweighting.normalize must be one of: none, mean_one.")

    return {
        "enabled": True,
        "start_epoch": start_epoch,
        "power": power,
        "normalize": normalize,
    }


def _compute_deferred_reweighting_weights(
    class_counts: torch.Tensor,
    power: float,
    normalize: str,
) -> torch.Tensor:
    weights = class_counts.float().pow(-float(power))
    if normalize == "mean_one":
        weights = weights / weights.mean()
    return weights


def get_loss_runtime_metadata(loss_cfg: Dict[str, Any], criterion: Optional[nn.Module] = None) -> Dict[str, Any]:
    """Return structured loss metadata for summaries and run artifacts."""
    if criterion is not None and hasattr(criterion, "get_runtime_metadata"):
        return criterion.get_runtime_metadata()

    name = str(loss_cfg.get("name", "cross_entropy")).lower()
    drw_cfg = loss_cfg.get("deferred_reweighting", {})
    if drw_cfg is None:
        drw_cfg = {}
    if not isinstance(drw_cfg, dict):
        drw_cfg = {}

    enabled = bool(drw_cfg.get("enabled", False))
    return {
        "loss_name": name,
        "loss_family": "ldam" if name == "ldam" else name,
        "class_weight_source": "static" if loss_cfg.get("class_weights") is not None else "none",
        "active_class_weight_min": None,
        "active_class_weight_max": None,
        "deferred_reweighting": {
            "enabled": enabled,
            "active": False,
            "start_epoch": int(drw_cfg["start_epoch"]) if enabled and drw_cfg.get("start_epoch") is not None else None,
            "activation_epoch": int(drw_cfg["start_epoch"]) if enabled and drw_cfg.get("start_epoch") is not None else None,
            "power": float(drw_cfg["power"]) if enabled and drw_cfg.get("power") is not None else None,
            "normalize": str(drw_cfg["normalize"]) if enabled and drw_cfg.get("normalize") is not None else None,
        },
    }


def build_loss(
    loss_cfg: Dict,
    num_classes: int,
    class_weights: Optional[torch.Tensor] = None,
    class_counts: Optional[torch.Tensor] = None,
    total_epochs: Optional[int] = None,
    current_epoch: Optional[int] = None,
) -> nn.Module:
    """Build criterion from configuration."""
    name = loss_cfg.get("name", "cross_entropy").lower()
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
    drw_cfg = _validate_deferred_reweighting(loss_cfg, loss_name=name, total_epochs=total_epochs)

    if class_weights is not None and class_weights.numel() != num_classes:
        raise ValueError(
            f"class_weights length {class_weights.numel()} does not match num_classes {num_classes}."
        )

    if name == "cross_entropy":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
    if name == "focal":
        return FocalLoss(
            gamma=float(loss_cfg.get("focal_gamma", 2.0)),
            weight=class_weights,
            reduction="mean",
        )
    if name == "balanced_softmax":
        return BalancedSoftmaxLoss(
            class_counts=_validate_class_counts(class_counts, num_classes=num_classes, loss_name=name),
            weight=class_weights,
            label_smoothing=label_smoothing,
            reduction="mean",
        )
    if name == "logit_adjusted_ce":
        return LogitAdjustedCrossEntropyLoss(
            class_counts=_validate_class_counts(class_counts, num_classes=num_classes, loss_name=name),
            tau=float(loss_cfg.get("logit_adjusted_tau", 1.0)),
            weight=class_weights,
            label_smoothing=label_smoothing,
            reduction="mean",
        )
    if name == "ldam":
        return LDAMLoss(
            class_counts=_validate_class_counts(class_counts, num_classes=num_classes, loss_name=name),
            max_margin=float(loss_cfg.get("ldam_max_margin", 0.5)),
            scale=float(loss_cfg.get("ldam_scale", 30.0)),
            weight=class_weights,
            label_smoothing=label_smoothing,
            reduction="mean",
            deferred_reweighting=drw_cfg,
            total_epochs=total_epochs,
            current_epoch=int(current_epoch or 0),
        )
    raise ValueError(f"Unsupported loss name: {name}")
