"""Plotting helpers."""

from __future__ import annotations

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, classes_mapping, normalize=True):
    conf_mat = confusion_matrix(y_true, y_pred)
    if normalize:
        with np.errstate(all="ignore"):
            conf_mat_percent = conf_mat.astype(float) / conf_mat.sum(axis=1, keepdims=True) * 100
            conf_mat_percent = np.nan_to_num(conf_mat_percent)
    else:
        conf_mat_percent = conf_mat

    tick_labels = list(classes_mapping.values())
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = sns.color_palette("Blues", as_cmap=True)
    annot = np.empty_like(conf_mat, dtype=object)

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            count = conf_mat[i, j]
            percent = conf_mat_percent[i, j]
            annot[i, j] = f"{count}\n({percent:.1f}%)" if normalize else str(count)

    sns.heatmap(
        conf_mat_percent,
        annot=annot,
        fmt="",
        cmap=cmap,
        cbar=True,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        linewidths=0.5,
        square=True,
        cbar_kws={"label": "Percentage (%)" if normalize else "Count"},
        ax=ax,
    )
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")
    return fig, ax
