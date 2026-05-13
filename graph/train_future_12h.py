#!/usr/bin/env python3
"""Train a day-level 12-hour future threat forecaster."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from graph.model import HGTFutureThreatForecaster
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    from model import HGTFutureThreatForecaster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("graph/output/future_12h_dataset.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--multilabel-weight", type=float, default=0.25)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-output", type=Path, default=Path("graph/output/future_12h_metrics.json"))
    parser.add_argument("--plots-dir", type=Path, default=Path("graph/output"))
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def chronological_masks(num_nodes: int, val_size: float, test_size: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    test_count = max(1, int(num_nodes * test_size)) if num_nodes >= 3 else 0
    val_count = max(1, int(num_nodes * val_size)) if num_nodes >= 3 else 0
    train_count = max(0, num_nodes - val_count - test_count)
    train = torch.zeros(num_nodes, dtype=torch.bool)
    val = torch.zeros(num_nodes, dtype=torch.bool)
    test = torch.zeros(num_nodes, dtype=torch.bool)
    train[:train_count] = True
    val[train_count : train_count + val_count] = True
    test[train_count + val_count :] = True
    if not bool(train.any()) and num_nodes:
        train[0] = True
    return train, val, test


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, threshold: float) -> dict[str, object]:
    if int(mask.sum().item()) == 0:
        return {"support": 0}
    y_true = labels[mask].detach().cpu().numpy().astype(np.int64)
    scores = torch.sigmoid(logits[mask]).detach().cpu().numpy()
    y_pred = (scores >= threshold).astype(np.int64)
    metrics: dict[str, object] = {
        "support": int(len(y_true)),
        "positive_support": int(y_true.sum()),
        "negative_support": int(len(y_true) - y_true.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "confusion_labels": ["no_future_hackmageddon_event", "future_hackmageddon_event"],
    }
    if len(set(y_true.tolist())) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        metrics["pr_auc"] = float(average_precision_score(y_true, scores))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics


def _masked_numpy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    y_true = labels[mask].detach().cpu().numpy().astype(np.int64)
    scores = torch.sigmoid(logits[mask]).detach().cpu().numpy()
    return y_true, scores


def plot_history(history: list[dict[str, float]], path: Path) -> None:
    if not history:
        return
    epochs = [int(row["epoch"]) for row in history]
    losses = [float(row["loss"]) for row in history]
    binary_losses = [float(row["binary_loss"]) for row in history]
    multilabel_losses = [float(row["multilabel_loss"]) for row in history]
    val_f1 = [float(row["val_f1"]) for row in history]

    fig, loss_axis = plt.subplots(figsize=(9, 5))
    f1_axis = loss_axis.twinx()
    lines = []
    lines += loss_axis.plot(epochs, losses, marker="o", color="#1f77b4", label="total loss")
    lines += loss_axis.plot(epochs, binary_losses, marker=".", color="#ff7f0e", label="binary loss")
    lines += loss_axis.plot(epochs, multilabel_losses, marker=".", color="#9467bd", label="multilabel loss")
    lines += f1_axis.plot(epochs, val_f1, marker="s", color="#2ca02c", label="val F1")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    f1_axis.set_ylabel("F1")
    loss_axis.grid(True, alpha=0.25)
    loss_axis.legend(lines, [line.get_label() for line in lines], loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_binary_confusion(metrics: dict[str, object], title: str, path: Path) -> None:
    matrix = np.array(metrics.get("confusion_matrix") or [], dtype=np.int64)
    labels = list(metrics.get("confusion_labels") or ["negative", "positive"])
    if matrix.size == 0:
        return

    fig, axis = plt.subplots(figsize=(6.5, 5.5))
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


def plot_score_distribution(
    logits: torch.Tensor,
    labels: torch.Tensor,
    masks: dict[str, torch.Tensor],
    threshold: float,
    path: Path,
) -> None:
    fig, axes = plt.subplots(len(masks), 1, figsize=(8, 3.0 * len(masks)), sharex=True)
    if len(masks) == 1:
        axes = [axes]
    for axis, (name, mask) in zip(axes, masks.items(), strict=True):
        y_true, scores = _masked_numpy(logits, labels, mask)
        negative = scores[y_true == 0]
        positive = scores[y_true == 1]
        bins = np.linspace(0.0, 1.0, 31)
        axis.hist(negative, bins=bins, alpha=0.65, label="negative", color="#1f77b4")
        axis.hist(positive, bins=bins, alpha=0.65, label="positive", color="#d62728")
        axis.axvline(threshold, color="#111827", linestyle="--", linewidth=1.2, label=f"threshold={threshold:.2f}")
        axis.set_title(f"{name} predicted probabilities")
        axis.set_ylabel("snapshots")
        axis.grid(True, alpha=0.2)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("P(future Hackmageddon event)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_precision_recall(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, path: Path) -> None:
    y_true, scores = _masked_numpy(logits, labels, mask)
    if len(set(y_true.tolist())) < 2:
        return
    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)
    fig, axis = plt.subplots(figsize=(6.5, 5.5))
    axis.plot(recall, precision, color="#2ca02c", linewidth=2, label=f"AP={pr_auc:.3f}")
    axis.set_title("Test precision-recall curve")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_plots(
    metrics: dict[str, object],
    binary_logits: torch.Tensor,
    y_binary: torch.Tensor,
    masks: dict[str, torch.Tensor],
    plots_dir: Path,
    threshold: float,
) -> dict[str, str]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "history": plots_dir / "future_12h_training_history.png",
        "val_confusion_matrix": plots_dir / "future_12h_val_confusion_matrix.png",
        "test_confusion_matrix": plots_dir / "future_12h_test_confusion_matrix.png",
        "score_distribution": plots_dir / "future_12h_score_distribution.png",
        "test_precision_recall": plots_dir / "future_12h_test_precision_recall.png",
    }
    plot_history(metrics["history"], paths["history"])  # type: ignore[arg-type]
    plot_binary_confusion(metrics["val"], "Validation confusion matrix", paths["val_confusion_matrix"])  # type: ignore[arg-type]
    plot_binary_confusion(metrics["test"], "Test confusion matrix", paths["test_confusion_matrix"])  # type: ignore[arg-type]
    plot_score_distribution(binary_logits, y_binary, masks, threshold, paths["score_distribution"])
    plot_precision_recall(binary_logits, y_binary, masks["test"], paths["test_precision_recall"])
    return {name: str(path) for name, path in paths.items() if path.exists()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data = torch.load(args.data, weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    num_days = int(data["day"].num_nodes)
    train_mask, val_mask, test_mask = chronological_masks(num_days, args.val_size, args.test_size)
    train_mask = train_mask.to(device) & data["day"].train_mask
    val_mask = val_mask.to(device) & data["day"].train_mask
    test_mask = test_mask.to(device) & data["day"].train_mask

    y_binary = data["day"].y_future_binary
    y_multilabel = data["day"].y_future_multilabel
    positive_count = y_binary[train_mask].sum().clamp_min(1.0)
    negative_count = (train_mask.sum() - y_binary[train_mask].sum()).clamp_min(1.0)
    pos_weight = (negative_count / positive_count).to(device)

    model = HGTFutureThreatForecaster(
        metadata=data.metadata(),
        hidden_dim=args.hidden_dim,
        num_labels=len(data.label_to_id),
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        binary_logits, multilabel_logits = model(data.x_dict, data.edge_index_dict)
        binary_loss = F.binary_cross_entropy_with_logits(
            binary_logits[train_mask],
            y_binary[train_mask],
            pos_weight=pos_weight,
        )
        multilabel_loss = F.binary_cross_entropy_with_logits(
            multilabel_logits[train_mask],
            y_multilabel[train_mask],
        )
        loss = binary_loss + args.multilabel_weight * multilabel_loss
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch == args.epochs or epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                binary_logits, _ = model(data.x_dict, data.edge_index_dict)
                val_metrics = binary_metrics(binary_logits, y_binary, val_mask, args.threshold)
            row = {
                "epoch": epoch,
                "loss": float(loss.item()),
                "binary_loss": float(binary_loss.item()),
                "multilabel_loss": float(multilabel_loss.item()),
                "val_f1": float(val_metrics.get("f1", 0.0) or 0.0),
            }
            history.append(row)
            print(
                f"epoch={epoch} loss={row['loss']:.4f} "
                f"val_f1={row['val_f1']:.4f}"
            )

    model.eval()
    with torch.no_grad():
        binary_logits, multilabel_logits = model(data.x_dict, data.edge_index_dict)

    metrics = {
        "device": str(device),
        "epochs": args.epochs,
        "target": (
            "For each day snapshot at 12:00 UTC, predict whether at least one "
            "Hackmageddon event is reported in the future target window."
        ),
        "threshold": args.threshold,
        "train": binary_metrics(binary_logits, y_binary, train_mask, args.threshold),
        "val": binary_metrics(binary_logits, y_binary, val_mask, args.threshold),
        "test": binary_metrics(binary_logits, y_binary, test_mask, args.threshold),
        "history": history,
        "parameters": {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "heads": args.heads,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "multilabel_weight": args.multilabel_weight,
        },
    }
    plot_paths = write_plots(
        metrics,
        binary_logits,
        y_binary,
        {"train": train_mask, "val": val_mask, "test": test_mask},
        args.plots_dir,
        args.threshold,
    )
    metrics["plots"] = plot_paths
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved metrics: {args.metrics_output}")
    for name, path in plot_paths.items():
        print(f"saved plot {name}: {path}")
    print(
        "summary: "
        f"test_f1={float(metrics['test'].get('f1', 0.0) or 0.0):.4f} "
        f"test_pr_auc={metrics['test'].get('pr_auc')}"
    )


if __name__ == "__main__":
    main()
