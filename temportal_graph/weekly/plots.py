from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_history(history: list[dict[str, float]], path: Path) -> None:
    if not history:
        return
    epochs = [int(row["epoch"]) for row in history]
    losses = [float(row["loss"]) for row in history]
    count_losses = [float(row["count_bucket_loss"]) for row in history]
    type_losses = [float(row["type_multilabel_loss"]) for row in history]
    val_f1 = [float(row["val_count_macro_f1"]) for row in history]

    fig, loss_axis = plt.subplots(figsize=(9, 5))
    f1_axis = loss_axis.twinx()
    lines = []
    lines += loss_axis.plot(epochs, losses, marker="o", color="#1f77b4", label="total loss")
    lines += loss_axis.plot(epochs, count_losses, marker=".", color="#ff7f0e", label="count bucket loss")
    lines += loss_axis.plot(epochs, type_losses, marker=".", color="#9467bd", label="type multilabel loss")
    lines += f1_axis.plot(epochs, val_f1, marker="s", color="#2ca02c", label="val count macro F1")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    f1_axis.set_ylabel("F1")
    loss_axis.grid(True, alpha=0.25)
    loss_axis.legend(lines, [line.get_label() for line in lines], loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_labeled_confusion(metrics: dict[str, object], title: str, path: Path) -> None:
    matrix = np.array(metrics.get("confusion_matrix") or [], dtype=np.int64)
    labels = list(metrics.get("confusion_labels") or [])
    if matrix.size == 0 or not labels:
        return
    fig, axis = plt.subplots(figsize=(7.2, 6.0))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=25, ha="right")
    axis.set_yticklabels(labels)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center", color="#111827")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="count")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_type_metrics(metrics: dict[str, object], path: Path) -> None:
    per_label = metrics.get("per_label") or {}
    rows = [
        (
            str(label),
            int(values.get("support", 0)),
            float(values.get("f1", 0.0) or 0.0),
            float(values.get("average_precision", 0.0) or 0.0),
        )
        for label, values in per_label.items()
        if isinstance(values, dict)
    ]
    if not rows:
        return
    rows.sort(key=lambda item: item[1])
    labels = [f"{row[0]} (n={row[1]})" for row in rows]
    f1_values = [row[2] for row in rows]
    ap_values = [row[3] for row in rows]
    fig_height = max(5.5, 0.42 * len(rows))
    fig, axis = plt.subplots(figsize=(9.0, fig_height))
    y_pos = np.arange(len(rows))
    axis.barh(y_pos - 0.18, f1_values, height=0.34, label="F1", color="#1f77b4")
    axis.barh(y_pos + 0.18, ap_values, height=0.34, label="average precision", color="#ff7f0e")
    axis.set_yticks(y_pos)
    axis.set_yticklabels(labels)
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("score")
    axis.set_title("Test threat type metrics")
    axis.grid(True, axis="x", alpha=0.25)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_plots(
    metrics: dict[str, object],
    masks: dict[str, object],
    plots_dir: Path,
) -> dict[str, str]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "history": plots_dir / "weekly_training_history.png",
        "val_count_confusion_matrix": plots_dir / "weekly_val_count_confusion_matrix.png",
        "test_count_confusion_matrix": plots_dir / "weekly_test_count_confusion_matrix.png",
        "test_type_metrics": plots_dir / "weekly_test_type_metrics.png",
    }
    plot_history(metrics["history"], paths["history"])  # type: ignore[arg-type]
    count_metrics_by_split = metrics["count_bucket"]  # type: ignore[assignment]
    plot_labeled_confusion(
        count_metrics_by_split["val"], "Validation event count bucket confusion matrix", paths["val_count_confusion_matrix"]  # type: ignore[index,arg-type]
    )
    plot_labeled_confusion(
        count_metrics_by_split["test"], "Test event count bucket confusion matrix", paths["test_count_confusion_matrix"]  # type: ignore[index,arg-type]
    )
    type_metrics_by_split = metrics["threat_type_multilabel"]  # type: ignore[assignment]
    plot_type_metrics(type_metrics_by_split["test"], paths["test_type_metrics"])  # type: ignore[index,arg-type]
    return {name: str(path) for name, path in paths.items() if path.exists()}
