"""ResNet variants and lightweight ResNet-based modifications."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import resnet as tv_resnet

from src.models.modules import CBAMBlock, DropPath, SEBlock

try:
    import timm
except ImportError:  # pragma: no cover - dependency is optional until a timm-backed model is requested
    timm = None


def _build_attention(attention: str, channels: int) -> nn.Module:
    attention = attention.lower()
    if attention == "none":
        return nn.Identity()
    if attention == "se":
        return SEBlock(channels=channels)
    if attention == "cbam":
        return CBAMBlock(channels=channels)
    raise ValueError(f"Unsupported attention type: {attention}")


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def apply_resnetd_stem(model: nn.Module) -> None:
    """Replace 7x7 stem with a pretrained-friendly 3x3 stack (ResNet-D style)."""
    if not isinstance(model.conv1, nn.Conv2d):
        raise TypeError("Expected model.conv1 to be Conv2d before applying ResNet-D stem.")

    old_conv1 = model.conv1
    old_bn1 = model.bn1 if isinstance(model.bn1, nn.BatchNorm2d) else None

    stem = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
    )
    stem.apply(_init_weights)

    # Warm-start conv1 from pretrained 7x7 center weights, and keep the next 3x3 layers identity-like.
    with torch.no_grad():
        center = old_conv1.weight.data[:, :, 2:5, 2:5]
        stem[0].weight.copy_(center)
        nn.init.dirac_(stem[3].weight)
        nn.init.dirac_(stem[6].weight)

    new_bn1 = nn.BatchNorm2d(64)
    if old_bn1 is not None:
        new_bn1.load_state_dict(old_bn1.state_dict())
    else:
        new_bn1.apply(_init_weights)

    model.conv1 = stem
    model.bn1 = new_bn1
    model.relu = nn.ReLU(inplace=True)


def apply_resnetd_downsample(model: nn.Module) -> None:
    """Replace stride-2 projection shortcut with avgpool + 1x1 conv stride-1."""
    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        layer = getattr(model, layer_name)
        for block in layer:
            if not hasattr(block, "downsample") or block.downsample is None:
                continue
            downsample = block.downsample
            if not isinstance(downsample, nn.Sequential) or len(downsample) < 2:
                continue
            conv = downsample[0]
            bn = downsample[1]
            if not isinstance(conv, nn.Conv2d) or conv.stride != (2, 2):
                continue
            new_conv = nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            )
            new_conv.weight.data.copy_(conv.weight.data)
            new_bn = nn.BatchNorm2d(conv.out_channels)
            new_bn.load_state_dict(bn.state_dict())
            block.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False),
                new_conv,
                new_bn,
            )


def _normalize_se_mode(se_mode: str) -> str:
    mode = str(se_mode).lower()
    if mode in {"none", ""}:
        return "none"
    if mode in {"bottleneck", "block", "bottleneck_se", "se_bottleneck"}:
        return "bottleneck"
    raise ValueError(f"Unsupported se_mode: {se_mode}")


class BottleneckWithSEAndDropPath(nn.Module):
    """Wrap a torchvision bottleneck with optional SE and stochastic depth."""

    def __init__(
        self,
        block: tv_resnet.Bottleneck,
        se_reduction: int = 16,
        drop_prob: float = 0.0,
        enable_se: bool = True,
    ) -> None:
        super().__init__()
        self.block = block
        channels = block.conv3.out_channels
        self.se = SEBlock(channels=channels, reduction=se_reduction) if enable_se else nn.Identity()
        self.drop_path = DropPath(drop_prob=drop_prob) if drop_prob > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.block.conv1(x)
        out = self.block.bn1(out)
        out = self.block.relu(out)

        out = self.block.conv2(out)
        out = self.block.bn2(out)
        out = self.block.relu(out)

        out = self.block.conv3(out)
        out = self.block.bn3(out)
        out = self.se(out)
        out = self.drop_path(out)

        if self.block.downsample is not None:
            identity = self.block.downsample(x)

        out += identity
        out = self.block.relu(out)
        return out


def apply_bottleneck_enhancements(
    model: nn.Module,
    se_mode: str = "none",
    se_reduction: int = 16,
    drop_path_rate: float = 0.0,
) -> None:
    """Apply block-level SE and stochastic depth to bottleneck blocks."""
    se_mode_norm = _normalize_se_mode(se_mode)
    enable_se = se_mode_norm == "bottleneck"
    drop_path_rate = max(float(drop_path_rate), 0.0)

    refs = []
    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        layer = getattr(model, layer_name)
        for idx, block in enumerate(layer):
            if isinstance(block, tv_resnet.Bottleneck):
                refs.append((layer, idx, block))

    if not refs or (not enable_se and drop_path_rate <= 0.0):
        return

    total = len(refs)
    for i, (layer, idx, block) in enumerate(refs):
        drop_prob = drop_path_rate * i / max(total - 1, 1)
        layer[idx] = BottleneckWithSEAndDropPath(
            block=block,
            se_reduction=se_reduction,
            drop_prob=drop_prob,
            enable_se=enable_se,
        )


class ResNetClassifier(nn.Module):
    """ResNet wrapper with optional stage attentions and custom classifier head."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        attention: str = "none",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        in_features = backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(in_features, num_classes)
        self.attn1 = _build_attention(attention, 256)
        self.attn2 = _build_attention(attention, 512)
        self.attn3 = _build_attention(attention, 1024)
        self.attn4 = _build_attention(attention, 2048)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.attn1(x)
        x = self.backbone.layer2(x)
        x = self.attn2(x)
        x = self.backbone.layer3(x)
        x = self.attn3(x)
        x = self.backbone.layer4(x)
        x = self.attn4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.dropout(x)
        return self.classifier(x)


class TimmFeatureClassifier(nn.Module):
    """Generic classifier wrapper for timm-backed residual-family feature extractors."""

    def __init__(self, backbone: nn.Module, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.backbone = backbone
        in_features = getattr(backbone, "num_features", None)
        if not isinstance(in_features, int) or in_features <= 0:
            raise ValueError("timm-backed backbone must expose a positive integer num_features.")
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        if isinstance(x, (tuple, list)):
            if not x:
                raise ValueError("timm-backed backbone returned an empty feature container.")
            x = x[-1]
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected timm-backed backbone to return a tensor, got {type(x)!r}.")
        if x.ndim > 2:
            x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.dropout(x)
        return self.classifier(x)


def _get_resnet_weights(model_name: str):
    mapping = {
        "resnet50": tv_models.ResNet50_Weights.DEFAULT,
        "resnet101": tv_models.ResNet101_Weights.DEFAULT,
        "resnet152": tv_models.ResNet152_Weights.DEFAULT,
        "resnext50_32x4d": tv_models.ResNeXt50_32X4D_Weights.DEFAULT,
        "wide_resnet50_2": tv_models.Wide_ResNet50_2_Weights.DEFAULT,
        "resnext101_64x4d": tv_models.ResNeXt101_64X4D_Weights.DEFAULT,
        "resnext101_32x8d": tv_models.ResNeXt101_32X8D_Weights.DEFAULT,
    }
    return mapping[model_name]


def _build_resnext50_32x3d() -> tv_resnet.ResNet:
    """Construct a custom ResNeXt-50 32x3d backbone."""
    return tv_resnet.ResNet(
        block=tv_resnet.Bottleneck,
        layers=[3, 4, 6, 3],
        groups=32,
        width_per_group=3,
    )


def _slice_grouped_axis0(weight: torch.Tensor, target_channels: int, groups: int) -> torch.Tensor:
    """Slice grouped channels along axis 0 while preserving every group."""
    source_channels = weight.shape[0]
    if source_channels == target_channels:
        return weight.clone()
    if source_channels < target_channels or source_channels % groups != 0 or target_channels % groups != 0:
        raise ValueError(
            f"Cannot slice axis0 grouped tensor from {tuple(weight.shape)} to target channels {target_channels}."
        )
    source_per_group = source_channels // groups
    target_per_group = target_channels // groups
    parts = []
    for group_idx in range(groups):
        start = group_idx * source_per_group
        stop = start + target_per_group
        parts.append(weight[start:stop].clone())
    return torch.cat(parts, dim=0)


def _slice_grouped_axis1(weight: torch.Tensor, target_channels: int, groups: int) -> torch.Tensor:
    """Slice grouped channels along axis 1 while preserving every group."""
    source_channels = weight.shape[1]
    if source_channels == target_channels:
        return weight.clone()
    if source_channels < target_channels or source_channels % groups != 0 or target_channels % groups != 0:
        raise ValueError(
            f"Cannot slice axis1 grouped tensor from {tuple(weight.shape)} to target channels {target_channels}."
        )
    source_per_group = source_channels // groups
    target_per_group = target_channels // groups
    parts = []
    for group_idx in range(groups):
        start = group_idx * source_per_group
        stop = start + target_per_group
        parts.append(weight[:, start:stop].clone())
    return torch.cat(parts, dim=1)


def _copy_bn_stats(target_bn: nn.BatchNorm2d, source_bn: nn.BatchNorm2d, groups: int | None = None) -> None:
    """Copy batchnorm statistics, using grouped slicing when width shrinks."""
    with torch.no_grad():
        if groups is None:
            target_bn.weight.copy_(source_bn.weight[: target_bn.weight.shape[0]])
            target_bn.bias.copy_(source_bn.bias[: target_bn.bias.shape[0]])
            target_bn.running_mean.copy_(source_bn.running_mean[: target_bn.running_mean.shape[0]])
            target_bn.running_var.copy_(source_bn.running_var[: target_bn.running_var.shape[0]])
        else:
            target_bn.weight.copy_(_slice_grouped_axis0(source_bn.weight, target_bn.weight.shape[0], groups))
            target_bn.bias.copy_(_slice_grouped_axis0(source_bn.bias, target_bn.bias.shape[0], groups))
            target_bn.running_mean.copy_(
                _slice_grouped_axis0(source_bn.running_mean, target_bn.running_mean.shape[0], groups)
            )
            target_bn.running_var.copy_(
                _slice_grouped_axis0(source_bn.running_var, target_bn.running_var.shape[0], groups)
            )
        target_bn.num_batches_tracked.copy_(source_bn.num_batches_tracked)


def _copy_resnext50_32x_width_reduced_weights(
    target_model: tv_resnet.ResNet,
    source_model: tv_resnet.ResNet,
) -> None:
    """Copy ResNeXt-50 weights from a wider grouped bottleneck source model."""
    with torch.no_grad():
        target_model.conv1.weight.copy_(source_model.conv1.weight)
        _copy_bn_stats(target_model.bn1, source_model.bn1)

        for layer_name in ("layer1", "layer2", "layer3", "layer4"):
            target_layer = getattr(target_model, layer_name)
            source_layer = getattr(source_model, layer_name)
            for target_block, source_block in zip(target_layer, source_layer):
                groups = int(target_block.conv2.groups)

                target_block.conv1.weight.copy_(
                    _slice_grouped_axis0(source_block.conv1.weight, target_block.conv1.out_channels, groups)
                )
                _copy_bn_stats(target_block.bn1, source_block.bn1, groups=groups)

                conv2_weight = _slice_grouped_axis0(
                    source_block.conv2.weight,
                    target_block.conv2.out_channels,
                    groups,
                )
                conv2_weight = conv2_weight[:, : target_block.conv2.in_channels // groups].clone()
                target_block.conv2.weight.copy_(conv2_weight)
                _copy_bn_stats(target_block.bn2, source_block.bn2, groups=groups)

                conv3_weight = _slice_grouped_axis1(
                    source_block.conv3.weight,
                    target_block.conv3.in_channels,
                    groups,
                )
                target_block.conv3.weight.copy_(conv3_weight)
                _copy_bn_stats(target_block.bn3, source_block.bn3)

                if target_block.downsample is not None and source_block.downsample is not None:
                    target_downsample_conv = target_block.downsample[0]
                    source_downsample_conv = source_block.downsample[0]
                    target_downsample_conv.weight.copy_(source_downsample_conv.weight)
                    _copy_bn_stats(target_block.downsample[1], source_block.downsample[1])


def _warm_start_resnext50_32x3d_from_32x4d(target_model: tv_resnet.ResNet) -> Dict[str, Any]:
    """Warm-start custom ResNeXt-50 32x3d from torchvision ResNeXt-50 32x4d weights."""
    try:
        source_model = tv_models.resnext50_32x4d(weights=_get_resnet_weights("resnext50_32x4d"))
    except Exception as exc:  # pragma: no cover - environment-dependent I/O
        raise RuntimeError(
            "Failed to load torchvision ResNeXt-50 32x4d weights for custom "
            "resnext50_32x3d warm-start. Ensure the weights are cached or "
            "network access is available, or set model.pretrained to false."
        ) from exc

    _copy_resnext50_32x_width_reduced_weights(target_model, source_model)

    return {
        "mode": "resnext50_32x4d_slice",
        "source_model": "resnext50_32x4d",
        "source_weights": "torchvision_default",
        "status": "applied",
    }


def _build_timm_resnet_family_backbone(model_name: str, pretrained: bool) -> tuple[nn.Module, Dict[str, Any]]:
    """Construct a timm residual-family backbone with pooled features only."""
    if timm is None:
        raise ImportError(
            f"Model '{model_name}' requires the optional 'timm' dependency. "
            "Install timm in the active environment before using this backbone."
        )

    try:
        backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
    except Exception as exc:  # pragma: no cover - model registry and weight download are environment-dependent
        raise RuntimeError(
            f"Failed to build timm-backed model '{model_name}'. "
            "Ensure the model name is valid and pretrained weights are available in the active environment."
        ) from exc

    initialization = {
        "requested_pretrained": bool(pretrained),
        "custom_pretrained_init": None,
        "source": "timm_pretrained" if pretrained else "random",
        "source_model": model_name if pretrained else None,
        "status": "loaded" if pretrained else "not_requested",
        "backend": "timm",
    }
    return backbone, initialization


def build_resnet_variant(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    custom_pretrained_init: str | None = None,
    resnetd: bool = False,
    attention: str = "none",
    dropout: float = 0.0,
    se_mode: str = "none",
    se_reduction: int = 16,
    drop_path_rate: float = 0.0,
) -> nn.Module:
    """Build legal ResNet variants for homework."""
    model_name = model_name.lower()
    profile_defaults = {
        "resnet101d_bse_sd": {
            "base_model_name": "resnet101",
            "resnetd": True,
            "se_mode": "bottleneck",
            "drop_path_rate": 0.10,
        },
        "resnext101_32x8d_d_bse_sd": {
            "base_model_name": "resnext101_32x8d",
            "resnetd": True,
            "se_mode": "bottleneck",
            "drop_path_rate": 0.10,
        },
    }
    supported_timm_base = {
        "resnetv2_101x1_bit.goog_in21k_ft_in1k",
    }

    supported_base = {
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext50_32x3d",
        "resnext50_32x4d",
        "wide_resnet50_2",
        "resnext101_64x4d",
        "resnext101_32x8d",
    }
    if model_name in supported_base:
        base_model_name = model_name
        use_timm_backbone = False
    elif model_name in supported_timm_base:
        base_model_name = model_name
        use_timm_backbone = True
    elif model_name in profile_defaults:
        base_model_name = profile_defaults[model_name]["base_model_name"]
        use_timm_backbone = False
        if not resnetd:
            resnetd = bool(profile_defaults[model_name]["resnetd"])
        if str(se_mode).lower() in {"none", ""}:
            se_mode = str(profile_defaults[model_name]["se_mode"])
        if float(drop_path_rate) <= 0.0:
            drop_path_rate = float(profile_defaults[model_name]["drop_path_rate"])
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    initialization = {
        "requested_pretrained": bool(pretrained),
        "custom_pretrained_init": custom_pretrained_init,
        "source": "random",
        "status": "not_requested" if not pretrained else "requested",
    }

    if use_timm_backbone:
        if custom_pretrained_init not in {None, ""}:
            raise ValueError(
                f"model.custom_pretrained_init is not supported for timm-backed model '{base_model_name}'."
            )
        if resnetd:
            raise ValueError(f"model.resnetd is not supported for timm-backed model '{base_model_name}'.")
        if str(attention).lower() != "none":
            raise ValueError(f"model.attention is not supported for timm-backed model '{base_model_name}'.")
        if str(se_mode).lower() not in {"none", ""}:
            raise ValueError(f"model.se_mode is not supported for timm-backed model '{base_model_name}'.")
        if float(drop_path_rate) > 0.0:
            raise ValueError(f"model.drop_path_rate is not supported for timm-backed model '{base_model_name}'.")
        base_model, initialization = _build_timm_resnet_family_backbone(
            model_name=base_model_name,
            pretrained=pretrained,
        )
    elif base_model_name == "resnext50_32x3d":
        base_model = _build_resnext50_32x3d()
        if pretrained:
            init_mode = str(custom_pretrained_init or "").strip().lower()
            if not init_mode:
                raise ValueError(
                    "model.custom_pretrained_init must be set explicitly for resnext50_32x3d "
                    "when model.pretrained is true. Supported value: resnext50_32x4d_slice."
                )
            if init_mode != "resnext50_32x4d_slice":
                raise ValueError(
                    f"Unsupported custom_pretrained_init for resnext50_32x3d: {custom_pretrained_init}"
                )
            initialization = _warm_start_resnext50_32x3d_from_32x4d(base_model)
            initialization["requested_pretrained"] = True
            initialization["custom_pretrained_init"] = init_mode
        else:
            initialization["custom_pretrained_init"] = None
    else:
        weights = _get_resnet_weights(base_model_name) if pretrained else None
        base_model_fn = getattr(tv_models, base_model_name)
        base_model = base_model_fn(weights=weights)
        initialization = {
            "requested_pretrained": bool(pretrained),
            "custom_pretrained_init": custom_pretrained_init,
            "source": "torchvision_default" if pretrained else "random",
            "source_model": base_model_name if pretrained else None,
            "status": "loaded" if pretrained else "not_requested",
        }

    if resnetd:
        apply_resnetd_stem(base_model)
        apply_resnetd_downsample(base_model)

    if not use_timm_backbone:
        apply_bottleneck_enhancements(
            model=base_model,
            se_mode=se_mode,
            se_reduction=int(se_reduction),
            drop_path_rate=float(drop_path_rate),
        )

    if use_timm_backbone:
        model = TimmFeatureClassifier(
            backbone=base_model,
            num_classes=num_classes,
            dropout=dropout,
        )
    else:
        model = ResNetClassifier(
            backbone=base_model,
            num_classes=num_classes,
            attention=attention,
            dropout=dropout,
        )
    model._build_info = {  # type: ignore[attr-defined]
        "base_model_name": base_model_name,
        "initialization": initialization,
    }
    return model


def model_metadata(model: nn.Module) -> Dict[str, Any]:
    """Return parameter count stats."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metadata = {
        "total_params": total_params,
        "trainable_params": trainable_params,
    }
    build_info = getattr(model, "_build_info", None)
    if isinstance(build_info, dict):
        metadata["build_info"] = build_info
    return metadata
