#!/usr/bin/env python3
"""Train an HGT classifier over the Option A HeteroData graph."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

try:
    from graph.model import HGTMessageClassifier
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    from model import HGTMessageClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("graph/output/heterodata_option_a.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--gold-weight", type=float, default=3.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-output", type=Path, default=Path("graph/output/train_metrics.json"))
    parser.add_argument("--plots-dir", type=Path, default=Path("graph/output"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_masks(num_nodes: int, seed: int, val_size: float, test_size: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_nodes, generator=generator)
    test_count = int(num_nodes * test_size)
    val_count = int(num_nodes * val_size)
    test_indices = permutation[:test_count]
    val_indices = permutation[test_count : test_count + val_count]
    train_indices = permutation[test_count + val_count :]
    masks = []
    for indices in (train_indices, val_indices, test_indices):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[indices] = True
        masks.append(mask)
    return masks[0], masks[1], masks[2]


def weighted_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if int(mask.sum().item()) == 0:
        return logits.sum() * 0.0
    losses = F.cross_entropy(logits[mask], labels[mask], reduction="none")
    selected_weights = weights[mask].to(losses.device)
    return (losses * selected_weights).sum() / selected_weights.sum().clamp_min(1.0)


def evaluate(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    id_to_label: dict[int, str],
) -> dict[str, object]:
    if int(mask.sum().item()) == 0:
        return {"support": 0, "accuracy": 0.0, "macro_f1": 0.0, "confusion_matrix": []}
    y_true = labels[mask].detach().cpu().numpy()
    y_pred = logits[mask].argmax(dim=-1).detach().cpu().numpy()
    label_ids = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    matrix = confusion_matrix(y_true, y_pred, labels=label_ids)
    return {
        "support": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)),
        "labels": [id_to_label.get(int(label_id), str(label_id)) for label_id in label_ids],
        "confusion_matrix": matrix.tolist(),
    }


def plot_history(history: list[dict[str, float]], path: Path) -> None:
    if not history:
        return
    epochs = [int(row["epoch"]) for row in history]
    losses = [float(row["loss"]) for row in history]
    val_f1 = [float(row["val_macro_f1"]) for row in history]

    fig, loss_axis = plt.subplots(figsize=(9, 5))
    f1_axis = loss_axis.twinx()
    loss_line = loss_axis.plot(epochs, losses, marker="o", color="#1f77b4", label="train loss")
    f1_line = f1_axis.plot(epochs, val_f1, marker="s", color="#2ca02c", label="val macro-F1")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    f1_axis.set_ylabel("Macro-F1")
    loss_axis.grid(True, alpha=0.25)
    lines = loss_line + f1_line
    loss_axis.legend(lines, [line.get_label() for line in lines], loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(metrics: dict[str, object], title: str, path: Path) -> None:
    matrix = np.array(metrics.get("confusion_matrix") or [], dtype=np.float32)
    labels = list(metrics.get("labels") or [])
    if matrix.size == 0 or not labels:
        return

    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

    size = max(7.0, min(14.0, 0.52 * len(labels)))
    fig, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    axis.set_title(title)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axis.set_yticklabels(labels, fontsize=8)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Row-normalized count")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_plots(metrics: dict[str, object], plots_dir: Path) -> dict[str, str]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "history": plots_dir / "hgt_training_history.png",
        "test_confusion_matrix": plots_dir / "hgt_test_confusion_matrix.png",
        "gold_confusion_matrix": plots_dir / "hgt_gold_confusion_matrix.png",
    }
    plot_history(metrics["history"], paths["history"])  # type: ignore[arg-type]
    plot_confusion_matrix(metrics["test"], "Test confusion matrix", paths["test_confusion_matrix"])  # type: ignore[arg-type]
    plot_confusion_matrix(metrics["gold"], "Hackmageddon gold confusion matrix", paths["gold_confusion_matrix"])  # type: ignore[arg-type]
    return {name: str(path) for name, path in paths.items() if path.exists()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data = torch.load(args.data, weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    num_messages = int(data["message"].num_nodes)
    train_mask, val_mask, test_mask = split_masks(num_messages, args.seed, args.val_size, args.test_size)
    train_mask = train_mask.to(device) & data["message"].train_mask
    val_mask = val_mask.to(device) & data["message"].train_mask
    test_mask = test_mask.to(device) & data["message"].train_mask

    labels = data["message"].y_weak
    weights = torch.ones(num_messages, dtype=torch.float32, device=device)
    weights[data["message"].gold_mask] = args.gold_weight

    num_classes = len(data.label_to_id)
    model = HGTMessageClassifier(
        metadata=data.metadata(),
        hidden_dim=args.hidden_dim,
        out_channels=num_classes,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x_dict, data.edge_index_dict)
        loss = weighted_loss(logits, labels, train_mask, weights)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch == args.epochs or epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(data.x_dict, data.edge_index_dict)
                val_metrics = evaluate(logits, labels, val_mask, data.id_to_label)
            row = {"epoch": epoch, "loss": float(loss.item()), "val_macro_f1": val_metrics["macro_f1"]}
            history.append(row)
            print(
                f"epoch={epoch} loss={row['loss']:.4f} "
                f"val_macro_f1={float(row['val_macro_f1']):.4f}"
            )

    model.eval()
    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict)
    gold_mask = data["message"].gold_mask
    metrics = {
        "device": str(device),
        "epochs": args.epochs,
        "train": evaluate(logits, labels, train_mask, data.id_to_label),
        "val": evaluate(logits, labels, val_mask, data.id_to_label),
        "test": evaluate(logits, labels, test_mask, data.id_to_label),
        "gold": evaluate(logits, data["message"].y_gold.clamp_min(0), gold_mask, data.id_to_label),
        "history": history,
    }
    plot_paths = write_plots(metrics, args.plots_dir)
    metrics["plots"] = plot_paths
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved metrics: {args.metrics_output}")
    for name, path in plot_paths.items():
        print(f"saved plot {name}: {path}")
    print(
        "summary: "
        f"test_acc={metrics['test']['accuracy']:.4f} "
        f"test_macro_f1={metrics['test']['macro_f1']:.4f} "
        f"gold_acc={metrics['gold']['accuracy']:.4f} "
        f"gold_macro_f1={metrics['gold']['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
