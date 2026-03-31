"""Inference utilities (single model + optional TTA)."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def predict_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool = True,
    tta_cfg: Dict | None = None,
    return_logits: bool = False,
    desc: str = "Inference",
) -> Tuple[List[str], np.ndarray] | Tuple[List[str], np.ndarray, np.ndarray]:
    """Predict probabilities for test set with optional horizontal-flip TTA."""
    model.eval()
    tta_cfg = tta_cfg or {}
    use_tta = bool(tta_cfg.get("enabled", False))
    use_hflip = bool(tta_cfg.get("horizontal_flip", True))

    all_ids: List[str] = []
    all_probs: List[np.ndarray] = []
    all_logits: List[np.ndarray] = []

    pbar = tqdm(loader, desc=desc, leave=False)
    for images, sample_ids in pbar:
        images = images.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp and device.type == "cuda",
        ):
            logits = model(images)
            views = 1
            if use_tta and use_hflip:
                flipped = torch.flip(images, dims=[3])
                logits = logits + model(flipped)
                views += 1
            logits = logits / views
            probs = torch.softmax(logits, dim=1)

        all_ids.extend(sample_ids)
        all_probs.append(probs.detach().cpu().numpy())
        if return_logits:
            all_logits.append(logits.detach().cpu().numpy())

    probs_arr = np.concatenate(all_probs, axis=0)
    if return_logits:
        logits_arr = np.concatenate(all_logits, axis=0)
        return all_ids, probs_arr, logits_arr
    return all_ids, probs_arr
