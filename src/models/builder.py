"""Model builder, parameter-limit checks, and trainable-scope helpers."""

from __future__ import annotations

from typing import Dict, Tuple

import torch.nn as nn

from src.models.resnet_variants import build_resnet_variant, model_metadata


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Return model parameter count."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def build_model(model_cfg: Dict, num_classes: int) -> Tuple[nn.Module, Dict]:
    """Build model from config and enforce parameter limit."""
    model = build_resnet_variant(
        model_name=model_cfg["name"],
        num_classes=num_classes,
        pretrained=bool(model_cfg.get("pretrained", True)),
        custom_pretrained_init=model_cfg.get("custom_pretrained_init"),
        resnetd=bool(model_cfg.get("resnetd", False)),
        attention=str(model_cfg.get("attention", "none")),
        dropout=float(model_cfg.get("dropout", 0.0)),
        se_mode=str(model_cfg.get("se_mode", "none")),
        se_reduction=int(model_cfg.get("se_reduction", 16)),
        drop_path_rate=float(model_cfg.get("drop_path_rate", 0.0)),
    )

    metadata = model_metadata(model)
    total_params_m = metadata["total_params"] / 1_000_000
    limit_m = float(model_cfg.get("param_limit_million", 100.0))
    if total_params_m >= limit_m:
        raise ValueError(
            f"Model has {total_params_m:.2f}M params, which violates limit {limit_m:.2f}M."
        )
    metadata["total_params_million"] = total_params_m
    metadata["param_limit_million"] = limit_m
    return model, metadata


def configure_trainable_scope(model: nn.Module, trainable_scope: str) -> Dict:
    """Apply a staged trainable scope and return a compact summary."""
    normalized_scope = str(trainable_scope).strip().lower()

    if normalized_scope == "full_model":
        for param in model.parameters():
            param.requires_grad = True
    elif normalized_scope == "classifier_only":
        if not hasattr(model, "classifier") or not isinstance(
            getattr(model, "classifier"), nn.Module
        ):
            raise ValueError("classifier_only staged training requires model.classifier to exist.")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unsupported staged trainable scope: {trainable_scope}")

    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen_params = sum(param.numel() for param in model.parameters() if not param.requires_grad)
    return {
        "trainable_scope": normalized_scope,
        "trainable_parameter_names": trainable_names,
        "trainable_params": int(trainable_params),
        "frozen_params": int(frozen_params),
    }
