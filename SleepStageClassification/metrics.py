"""Metrics and class weighting."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score


CLASS_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }

    per_class_recall = recall_score(y_true, y_pred, labels=np.arange(5), average=None, zero_division=0)
    for i in range(5):
        result[f"recall_class_{i}"] = per_class_recall[i]

    return result


def smooth_class_weights(
    y_train: np.ndarray,
    device: torch.device | str,
    priority_scale: bool = True,
    scale: str = "none",
    min_clip: float = 0.8,
    max_clip: float = 3.0,
) -> torch.Tensor:
    classes = np.arange(5)
    class_counts = np.bincount(y_train, minlength=len(classes)).astype(np.float32)
    total = class_counts.sum()
    inv_freq = total / (class_counts + 1e-6)
    weights = inv_freq / np.mean(inv_freq)

    if priority_scale:
        priority_multipliers = np.array([1.2, 1.8, 1.1, 1.0, 1.5])
        weights *= priority_multipliers

    if scale == "log":
        weights = np.log1p(weights)
    elif scale == "sqrt":
        weights = np.sqrt(weights)
    elif scale != "none":
        raise ValueError("scale must be one of: none, log, sqrt")

    weights = np.clip(weights, min_clip, max_clip)
    return torch.tensor(weights, dtype=torch.float32, device=device)
