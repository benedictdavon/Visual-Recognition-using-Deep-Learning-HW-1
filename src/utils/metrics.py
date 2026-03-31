"""Metric helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import torch
from sklearn.metrics import confusion_matrix


@dataclass
class AverageMeter:
    """Track streaming averages."""

    value: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def topk_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: Iterable[int] = (1,),
) -> List[float]:
    """Compute top-k accuracy values in percentage."""
    with torch.no_grad():
        max_k = max(topk)
        batch_size = targets.size(0)
        _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        output = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            output.append(correct_k.mul_(100.0 / batch_size).item())
        return output


def compute_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    num_classes: int,
) -> np.ndarray:
    """Compute confusion matrix as numpy array."""
    labels = list(range(num_classes))
    return confusion_matrix(y_true, y_pred, labels=labels)


def per_class_accuracy(conf_mat: np.ndarray) -> Dict[int, float]:
    """Compute per-class accuracy from confusion matrix."""
    result = {}
    for class_idx in range(conf_mat.shape[0]):
        denom = conf_mat[class_idx].sum()
        result[class_idx] = float(conf_mat[class_idx, class_idx] / denom) if denom > 0 else 0.0
    return result


def macro_recall_from_confusion_matrix(
    conf_mat: np.ndarray,
    as_percentage: bool = False,
) -> float:
    """Compute macro recall from a confusion matrix."""
    class_recalls = per_class_accuracy(conf_mat)
    value = float(np.mean(list(class_recalls.values()))) if class_recalls else 0.0
    return value * 100.0 if as_percentage else value


def expected_calibration_error(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute top-1 expected calibration error from probabilities."""
    if probs.size == 0 or targets.size == 0:
        return 0.0
    if probs.shape[0] != targets.shape[0]:
        raise ValueError("probs and targets must contain the same number of samples.")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == targets).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, num=n_bins + 1)

    ece = 0.0
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        if upper >= 1.0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not np.any(mask):
            continue
        bin_accuracy = accuracies[mask].mean()
        bin_confidence = confidences[mask].mean()
        ece += np.abs(bin_accuracy - bin_confidence) * mask.mean()
    return float(ece)
