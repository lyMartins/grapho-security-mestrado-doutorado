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


def plot_count_bucket_summary(metrics_by_split: dict[str, object], path: Path) -> None:
    splits = ["train", "val", "test"]
    metric_names = ["accuracy", "macro_f1", "weighted_f1"]
    values = np.array(
        [
            [
                float(metrics_by_split.get(split, {}).get(metric_name, 0.0) or 0.0)  # type: ignore[union-attr]
                for metric_name in metric_names
            ]
            for split in splits
        ],
        dtype=np.float32,
    )
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    x_pos = np.arange(len(splits))
    width = 0.24
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    labels = ["Accuracy", "Macro F1", "Weighted F1"]
    for idx, label in enumerate(labels):
        axis.bar(x_pos + (idx - 1) * width, values[:, idx], width=width, label=label, color=colors[idx])
    axis.set_title("Event volume bucket classification")
    axis.set_ylabel("score")
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(x_pos)
    axis.set_xticklabels(["Train", "Validation", "Test"])
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_multilabel_summary(metrics_by_split: dict[str, object], path: Path) -> None:
    splits = ["train", "val", "test"]
    metric_names = ["micro_f1", "macro_f1"]
    values = np.array(
        [
            [
                float(metrics_by_split.get(split, {}).get(metric_name, 0.0) or 0.0)  # type: ignore[union-attr]
                for metric_name in metric_names
            ]
            for split in splits
        ],
        dtype=np.float32,
    )
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    x_pos = np.arange(len(splits))
    width = 0.30
    axis.bar(x_pos - width / 2, values[:, 0], width=width, label="Micro F1", color="#1f77b4")
    axis.bar(x_pos + width / 2, values[:, 1], width=width, label="Macro F1", color="#ff7f0e")
    axis.set_title("Threat type presence multilabel")
    axis.set_ylabel("score")
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(x_pos)
    axis.set_xticklabels(["Train", "Validation", "Test"])
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_count_regression_summary(metrics_by_split: dict[str, object], path: Path) -> None:
    splits = ["train", "val", "test"]
    mae_values = [
        float(metrics_by_split.get(split, {}).get("mae", 0.0) or 0.0)  # type: ignore[union-attr]
        for split in splits
    ]
    rmse_values = [
        float(metrics_by_split.get(split, {}).get("rmse", 0.0) or 0.0)  # type: ignore[union-attr]
        for split in splits
    ]
    true_means = [
        float(metrics_by_split.get(split, {}).get("true_mean", 0.0) or 0.0)  # type: ignore[union-attr]
        for split in splits
    ]
    pred_means = [
        float(metrics_by_split.get(split, {}).get("pred_mean", 0.0) or 0.0)  # type: ignore[union-attr]
        for split in splits
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    x_pos = np.arange(len(splits))
    width = 0.34
    axes[0].bar(x_pos - width / 2, mae_values, width=width, label="MAE", color="#d62728")
    axes[0].bar(x_pos + width / 2, rmse_values, width=width, label="RMSE", color="#9467bd")
    axes[0].set_title("Count regression error")
    axes[0].set_ylabel("events")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(["Train", "Validation", "Test"])
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].bar(x_pos - width / 2, true_means, width=width, label="True mean", color="#1f77b4")
    axes[1].bar(x_pos + width / 2, pred_means, width=width, label="Pred mean", color="#ff7f0e")
    axes[1].set_title("Mean event count")
    axes[1].set_ylabel("events")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(["Train", "Validation", "Test"])
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(loc="upper right")

    fig.suptitle("Threat type count regression")
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
        "count_bucket_summary": plots_dir / "weekly_count_bucket_metrics.png",
        "type_multilabel_summary": plots_dir / "weekly_type_multilabel_metrics.png",
        "type_count_regression_summary": plots_dir / "weekly_type_count_regression_metrics.png",
    }
    plot_history(metrics["history"], paths["history"])  # type: ignore[arg-type]
    count_metrics_by_split = metrics["count_bucket"]  # type: ignore[assignment]
    plot_count_bucket_summary(count_metrics_by_split, paths["count_bucket_summary"])  # type: ignore[arg-type]
    plot_labeled_confusion(
        count_metrics_by_split["val"], "Validation event count bucket confusion matrix", paths["val_count_confusion_matrix"]  # type: ignore[index,arg-type]
    )
    plot_labeled_confusion(
        count_metrics_by_split["test"], "Test event count bucket confusion matrix", paths["test_count_confusion_matrix"]  # type: ignore[index,arg-type]
    )
    type_metrics_by_split = metrics["threat_type_multilabel"]  # type: ignore[assignment]
    plot_multilabel_summary(type_metrics_by_split, paths["type_multilabel_summary"])  # type: ignore[arg-type]
    plot_type_metrics(type_metrics_by_split["test"], paths["test_type_metrics"])  # type: ignore[index,arg-type]
    count_regression_by_split = metrics["event_count_regression"]  # type: ignore[assignment]
    plot_count_regression_summary(count_regression_by_split, paths["type_count_regression_summary"])  # type: ignore[arg-type]
    return {name: str(path) for name, path in paths.items() if path.exists()}
