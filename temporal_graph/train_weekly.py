#!/usr/bin/env python3
"""Train a day-level next-day threat forecaster (7-day lookback window)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from weekly.metrics import (
    best_multilabel_f1_thresholds,
    count_bucket_class_weights,
    count_bucket_metrics,
    count_regression_metrics,
    count_bucket_baselines,
    multilabel_metrics,
    multilabel_pos_weights,
)
from weekly.model import WeeklyMultiTaskForecaster
from weekly.plots import write_plots
from weekly.splits import balanced_chronological_masks, chronological_masks, set_seed

_ROOT = Path(__file__).parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=_ROOT / "temporal_graph/output/weekly_dataset.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--count-bucket-weight", type=float, default=1.0)
    parser.add_argument("--multilabel-weight", type=float, default=0.25)
    parser.add_argument("--type-count-weight", type=float, default=0.10)
    parser.add_argument(
        "--count-class-weight-policy",
        choices=["none", "inverse", "inverse_sqrt"],
        default="inverse_sqrt",
    )
    parser.add_argument("--type-pos-weight-max", type=float, default=20.0)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-output", type=Path, default=_ROOT / "temporal_graph/output/weekly_metrics.json")
    parser.add_argument("--plots-dir", type=Path, default=_ROOT / "temporal_graph/output")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold-policy", choices=["fixed", "val_f1"], default="val_f1")
    parser.add_argument("--split-policy", choices=["chronological", "balanced_chronological"], default="balanced_chronological")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data = torch.load(args.data, weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    use_amp = device.type == "cuda"
    if use_amp:
        for node_type in data.node_types:
            x = data[node_type].x
            if x is not None and x.is_floating_point():
                data[node_type].x = x.half()

    y_count = data["day"].y_future_count
    y_count_bucket = data["day"].y_future_count_bucket
    y_type_multilabel = data["day"].y_future_type_multilabel
    y_type_counts = data["day"].y_future_type_counts
    threat_type_labels = [data.id_to_threat_type[idx] for idx in sorted(data.id_to_threat_type)]
    count_bucket_labels = [data.id_to_count_bucket[idx] for idx in sorted(data.id_to_count_bucket)]
    eligible_mask = data["day"].train_mask.detach().cpu()
    if args.split_policy == "balanced_chronological":
        train_mask, val_mask, test_mask = balanced_chronological_masks(
            y_count_bucket.detach().cpu(), eligible_mask, args.val_size, args.test_size
        )
    else:
        train_mask, val_mask, test_mask = chronological_masks(
            y_count_bucket.detach().cpu(), eligible_mask, args.val_size, args.test_size
        )
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    count_class_weights = count_bucket_class_weights(
        y_count_bucket, train_mask, len(count_bucket_labels), args.count_class_weight_policy
    )
    if count_class_weights is not None:
        count_class_weights = count_class_weights.to(device)
    type_pos_weights = multilabel_pos_weights(
        y_type_multilabel, train_mask, args.type_pos_weight_max
    ).to(device)

    model = WeeklyMultiTaskForecaster(
        metadata=data.metadata(),
        hidden_dim=args.hidden_dim,
        num_threat_types=len(threat_type_labels),
        num_count_buckets=len(count_bucket_labels),
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(data.x_dict, data.edge_index_dict)
            count_bucket_loss = F.cross_entropy(
                outputs["count_bucket"][train_mask],
                y_count_bucket[train_mask],
                weight=count_class_weights,
            )
            type_multilabel_loss = F.binary_cross_entropy_with_logits(
                outputs["type_multilabel"][train_mask],
                y_type_multilabel[train_mask],
                pos_weight=type_pos_weights,
            )
            type_count_loss = F.smooth_l1_loss(
                outputs["type_counts"][train_mask],
                torch.log1p(y_type_counts[train_mask]),
            )
            loss = (
                args.count_bucket_weight * count_bucket_loss
                + args.multilabel_weight * type_multilabel_loss
                + args.type_count_weight * type_count_loss
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if epoch == 1 or epoch == args.epochs or epoch % 10 == 0:
            model.eval()
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(data.x_dict, data.edge_index_dict)
                val_count_metrics = count_bucket_metrics(
                    outputs["count_bucket"], y_count_bucket, val_mask, count_bucket_labels
                )
            row = {
                "epoch": epoch,
                "loss": float(loss.item()),
                "count_bucket_loss": float(count_bucket_loss.item()),
                "type_multilabel_loss": float(type_multilabel_loss.item()),
                "type_count_loss": float(type_count_loss.item()),
                "val_count_macro_f1": float(val_count_metrics.get("macro_f1", 0.0) or 0.0),
                "lr": scheduler.get_last_lr()[0],
            }
            history.append(row)
            print(
                f"epoch={epoch} loss={row['loss']:.4f} "
                f"val_count_macro_f1={row['val_count_macro_f1']:.4f} "
                f"lr={row['lr']:.2e}"
            )

    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        outputs = model(data.x_dict, data.edge_index_dict)
    type_thresholds = (
        best_multilabel_f1_thresholds(outputs["type_multilabel"], y_type_multilabel, val_mask)
        if args.threshold_policy == "val_f1"
        else np.full(len(threat_type_labels), args.threshold, dtype=np.float32)
    )
    masks = {"train": train_mask, "val": val_mask, "test": test_mask}
    observed_counts = data["day"].observed_message_count
    split_summary = {
        name: {
            "support": int(mask.sum().item()),
            "mean_event_count": float(y_count[mask].mean().item()) if bool(mask.any()) else None,
            "count_bucket_counts": {
                count_bucket_labels[idx]: int((y_count_bucket[mask] == idx).sum().item())
                for idx in range(len(count_bucket_labels))
            },
            "start_date": data["day"].date[int(torch.nonzero(mask.detach().cpu(), as_tuple=False).flatten()[0])]
            if bool(mask.any())
            else None,
            "end_date": data["day"].date[int(torch.nonzero(mask.detach().cpu(), as_tuple=False).flatten()[-1])]
            if bool(mask.any())
            else None,
        }
        for name, mask in masks.items()
    }

    metrics: dict[str, object] = {
        "device": str(device),
        "epochs": args.epochs,
        "target": (
            "For each day snapshot D, predict Hackmageddon event volume and threat types "
            "reported on day D+1, using messages from the 7-day window [D-6 .. D]."
        ),
        "type_thresholds": {
            label: float(type_thresholds[idx]) for idx, label in enumerate(threat_type_labels)
        },
        "threshold_policy": args.threshold_policy,
        "split_policy": args.split_policy,
        "split_summary": split_summary,
        "count_bucket_labels": count_bucket_labels,
        "threat_type_labels": threat_type_labels,
        "count_bucket": {
            "train": count_bucket_metrics(outputs["count_bucket"], y_count_bucket, train_mask, count_bucket_labels),
            "val": count_bucket_metrics(outputs["count_bucket"], y_count_bucket, val_mask, count_bucket_labels),
            "test": count_bucket_metrics(outputs["count_bucket"], y_count_bucket, test_mask, count_bucket_labels),
        },
        "event_count_regression": {
            "train": count_regression_metrics(outputs["type_counts"], y_count, train_mask),
            "val": count_regression_metrics(outputs["type_counts"], y_count, val_mask),
            "test": count_regression_metrics(outputs["type_counts"], y_count, test_mask),
        },
        "threat_type_multilabel": {
            "train": multilabel_metrics(
                outputs["type_multilabel"], y_type_multilabel, train_mask, threat_type_labels, type_thresholds
            ),
            "val": multilabel_metrics(
                outputs["type_multilabel"], y_type_multilabel, val_mask, threat_type_labels, type_thresholds
            ),
            "test": multilabel_metrics(
                outputs["type_multilabel"], y_type_multilabel, test_mask, threat_type_labels, type_thresholds
            ),
        },
        "baselines": {
            "count_bucket": count_bucket_baselines(
                y_count_bucket.detach().cpu(),
                {k: v.detach().cpu() for k, v in masks.items()},
                count_bucket_labels,
            ),
        },
        "history": history,
        "parameters": {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "heads": args.heads,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "count_bucket_weight": args.count_bucket_weight,
            "multilabel_weight": args.multilabel_weight,
            "type_count_weight": args.type_count_weight,
            "count_class_weight_policy": args.count_class_weight_policy,
            "count_class_weights": None
            if count_class_weights is None
            else {
                count_bucket_labels[idx]: float(count_class_weights.detach().cpu()[idx].item())
                for idx in range(len(count_bucket_labels))
            },
            "type_pos_weight_max": args.type_pos_weight_max,
        },
    }
    plot_paths = write_plots(metrics, masks, args.plots_dir)
    metrics["plots"] = plot_paths
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved metrics: {args.metrics_output}")
    for name, path in plot_paths.items():
        print(f"saved plot {name}: {path}")
    print(
        "summary: "
        f"test_count_macro_f1={float(metrics['count_bucket']['test'].get('macro_f1', 0.0) or 0.0):.4f} "  # type: ignore[index,union-attr]
        f"test_multilabel_macro_f1={float(metrics['threat_type_multilabel']['test'].get('macro_f1', 0.0) or 0.0):.4f}"  # type: ignore[index,union-attr]
    )


if __name__ == "__main__":
    main()
