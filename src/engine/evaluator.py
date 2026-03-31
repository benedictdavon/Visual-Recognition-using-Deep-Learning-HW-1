"""Evaluation utilities."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.metrics import (
    AverageMeter,
    compute_confusion_matrix,
    expected_calibration_error,
    macro_recall_from_confusion_matrix,
    per_class_accuracy,
    topk_accuracy,
)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[torch.nn.Module] = None,
    amp: bool = True,
    num_classes: Optional[int] = None,
    desc: str = "Validation",
    return_predictions: bool = False,
) -> Dict:
    """Evaluate a classification model on a labeled dataloader."""
    model.eval()
    loss_meter = AverageMeter()
    nll_meter = AverageMeter()
    acc_meter = AverageMeter()
    acc5_meter = AverageMeter()

    all_targets = []
    all_preds = []
    all_probs = []

    pbar = tqdm(loader, desc=desc, leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp and device.type == "cuda",
        ):
            logits = model(images)
            loss = criterion(logits, targets) if criterion is not None else None

        if loss is not None:
            loss_meter.update(loss.item(), n=images.size(0))
        nll = F.cross_entropy(logits, targets)
        nll_meter.update(nll.item(), n=images.size(0))

        topk_list = (1, 5) if logits.size(1) >= 5 else (1, logits.size(1))
        acc_values = topk_accuracy(logits, targets, topk=topk_list)
        acc1 = acc_values[0]
        acc_meter.update(acc1, n=images.size(0))
        if len(acc_values) > 1:
            acc5_meter.update(acc_values[1], n=images.size(0))
        else:
            acc5_meter.update(acc1, n=images.size(0))

        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())
        all_probs.append(probs.detach().cpu().numpy())

        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc1": f"{acc_meter.avg:.2f}"})

    targets_arr = np.array(all_targets, dtype=np.int64)
    preds_arr = np.array(all_preds, dtype=np.int64)
    probs_arr = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, 0), dtype=np.float32)
    metrics = {
        "loss": float(loss_meter.avg),
        "nll": float(nll_meter.avg),
        "acc1": float(acc_meter.avg),
        "acc5": float(acc5_meter.avg),
        "num_samples": len(all_targets),
        "ece": expected_calibration_error(probs_arr, targets_arr),
    }

    if num_classes is not None and num_classes > 0:
        conf_mat = compute_confusion_matrix(all_targets, all_preds, num_classes=num_classes)
        class_acc = per_class_accuracy(conf_mat)
        metrics["confusion_matrix"] = conf_mat
        metrics["per_class_accuracy"] = class_acc
        metrics["macro_per_class_acc"] = float(np.mean(list(class_acc.values())))
        metrics["macro_recall"] = macro_recall_from_confusion_matrix(conf_mat, as_percentage=True)

    if return_predictions:
        metrics["targets"] = targets_arr
        metrics["preds"] = preds_arr
        metrics["probs"] = probs_arr

    return metrics
