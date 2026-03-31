"""Exponential moving average (EMA) for model weights."""

from __future__ import annotations

import copy

import torch
from torch import nn


class ModelEMA:
    """Maintain an EMA copy of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9998) -> None:
        """Clone the model and freeze the EMA weights."""
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend the live model weights into the EMA copy."""
        msd = model.state_dict()
        for k, ema_v in self.ema_model.state_dict().items():
            model_v = msd[k].detach()
            if not torch.is_floating_point(ema_v):
                ema_v.copy_(model_v)
            else:
                ema_v.copy_(ema_v * self.decay + (1.0 - self.decay) * model_v)

    def state_dict(self) -> dict:
        """Return the serializable EMA state."""
        return {
            "decay": self.decay,
            "ema_state_dict": self.ema_model.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore EMA weights from a serialized state."""
        self.decay = state_dict["decay"]
        self.ema_model.load_state_dict(state_dict["ema_state_dict"], strict=True)
