from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _masked_numpy(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    y_true = labels[mask].detach().cpu().numpy().astype(np.int64)
    scores = torch.sigmoid(logits[mask]).detach().cpu().numpy()
    return y_true, scores


def binary_metrics(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, threshold: float
) -> dict[str, object]:
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
        "brier": float(brier_score_loss(y_true, scores)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "confusion_labels": ["no_attack_tomorrow", "attack_tomorrow"],
    }
    if len(set(y_true.tolist())) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        metrics["pr_auc"] = float(average_precision_score(y_true, scores))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics


def binary_metrics_from_scores(
    scores: np.ndarray, labels: torch.Tensor, mask: torch.Tensor, threshold: float
) -> dict[str, object]:
    if int(mask.sum().item()) == 0:
        return {"support": 0}
    y_true = labels[mask].detach().cpu().numpy().astype(np.int64)
    masked_scores = scores[mask.detach().cpu().numpy()]
    y_pred = (masked_scores >= threshold).astype(np.int64)
    metrics: dict[str, object] = {
        "support": int(len(y_true)),
        "positive_support": int(y_true.sum()),
        "negative_support": int(len(y_true) - y_true.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, masked_scores)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "confusion_labels": ["no_attack_tomorrow", "attack_tomorrow"],
    }
    if len(set(y_true.tolist())) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, masked_scores))
        metrics["pr_auc"] = float(average_precision_score(y_true, masked_scores))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics


def count_bucket_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    bucket_labels: list[str],
) -> dict[str, object]:
    if int(mask.sum().item()) == 0:
        return {"support": 0}
    y_true = labels[mask].detach().cpu().numpy().astype(np.int64)
    y_pred = logits[mask].argmax(dim=-1).detach().cpu().numpy().astype(np.int64)
    label_ids = list(range(len(bucket_labels)))
    return {
        "support": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=label_ids, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=label_ids).tolist(),
        "confusion_labels": bucket_labels,
        "true_bucket_counts": {bucket_labels[i]: int((y_true == i).sum()) for i in label_ids},
        "pred_bucket_counts": {bucket_labels[i]: int((y_pred == i).sum()) for i in label_ids},
    }


def count_regression_metrics(predicted_log_type_counts: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> dict[str, object]:
    if int(mask.sum().item()) == 0:
        return {"support": 0}
    y_true = labels[mask].detach().cpu().numpy().astype(np.float32)
    y_pred = (
        torch.expm1(predicted_log_type_counts[mask].clamp_min(0.0))
        .sum(dim=-1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    return {
        "support": int(len(y_true)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "true_mean": float(np.mean(y_true)),
        "pred_mean": float(np.mean(y_pred)),
    }


def multilabel_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    label_names: list[str],
    threshold: float | np.ndarray = 0.5,
) -> dict[str, object]:
    if int(mask.sum().item()) == 0:
        return {"support": 0}
    y_true = labels[mask].detach().cpu().numpy().astype(np.int64)
    scores = torch.sigmoid(logits[mask]).detach().cpu().numpy()
    thresholds = np.asarray(threshold, dtype=np.float32)
    y_pred = (scores >= thresholds).astype(np.int64)
    per_label: dict[str, object] = {}
    for idx, label in enumerate(label_names):
        support = int(y_true[:, idx].sum())
        predicted = int(y_pred[:, idx].sum())
        if support:
            ap = float(average_precision_score(y_true[:, idx], scores[:, idx]))
        else:
            ap = None
        per_label[label] = {
            "support": support,
            "predicted": predicted,
            "precision": float(precision_score(y_true[:, idx], y_pred[:, idx], zero_division=0)),
            "recall": float(recall_score(y_true[:, idx], y_pred[:, idx], zero_division=0)),
            "f1": float(f1_score(y_true[:, idx], y_pred[:, idx], zero_division=0)),
            "average_precision": ap,
        }
    return {
        "support": int(y_true.shape[0]),
        "micro_f1": float(f1_score(y_true.ravel(), y_pred.ravel(), zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_label": per_label,
    }


def best_f1_threshold(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum().item()) == 0:
        return 0.5
    y_true, scores = _masked_numpy(logits, labels, mask)
    if len(set(y_true.tolist())) < 2:
        return 0.5
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        score = f1_score(y_true, (scores >= threshold).astype(np.int64), zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold


def best_multilabel_f1_thresholds(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    num_labels = int(labels.shape[1])
    thresholds = np.full(num_labels, 0.5, dtype=np.float32)
    if int(mask.sum().item()) == 0:
        return thresholds
    y_true = labels[mask].detach().cpu().numpy().astype(np.int64)
    scores = torch.sigmoid(logits[mask]).detach().cpu().numpy()
    grid = np.linspace(0.01, 0.99, 99)
    for idx in range(num_labels):
        if len(set(y_true[:, idx].tolist())) < 2:
            continue
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in grid:
            score = f1_score(y_true[:, idx], (scores[:, idx] >= threshold).astype(np.int64), zero_division=0)
            if score > best_f1:
                best_f1 = float(score)
                best_threshold = float(threshold)
        thresholds[idx] = best_threshold
    return thresholds


def best_f1_threshold_from_scores(scores: np.ndarray, labels: torch.Tensor, mask: torch.Tensor) -> float:
    y_true = labels[mask].detach().cpu().numpy().astype(np.int64)
    masked_scores = scores[mask.detach().cpu().numpy()]
    if len(set(y_true.tolist())) < 2:
        return 0.5
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        score = f1_score(y_true, (masked_scores >= threshold).astype(np.int64), zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold


def count_bucket_class_weights(
    labels: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
    policy: str,
) -> torch.Tensor | None:
    if policy == "none":
        return None
    counts = torch.bincount(labels[mask].detach().cpu(), minlength=num_classes).float()
    counts = counts.clamp_min(1.0)
    if policy == "inverse":
        weights = counts.sum() / (num_classes * counts)
    else:
        weights = torch.sqrt(counts.sum() / (num_classes * counts))
    return weights / weights.mean().clamp_min(1e-8)


def multilabel_pos_weights(labels: torch.Tensor, mask: torch.Tensor, max_weight: float) -> torch.Tensor:
    train_labels = labels[mask].detach()
    positive = train_labels.sum(dim=0)
    negative = train_labels.shape[0] - positive
    weights = torch.ones_like(positive)
    has_positive = positive > 0
    weights[has_positive] = negative[has_positive] / positive[has_positive].clamp_min(1.0)
    return weights.clamp(max=max_weight)


def baseline_metrics(
    labels: torch.Tensor,
    observed_counts: torch.Tensor,
    masks: dict[str, torch.Tensor],
) -> dict[str, object]:
    y_np = labels.detach().cpu().numpy().astype(np.float32)
    counts = observed_counts.detach().cpu().numpy().astype(np.float32)
    count_min = float(counts.min()) if counts.size else 0.0
    count_span = float(counts.max() - counts.min()) if counts.size else 1.0
    volume_scores = (counts - count_min) / max(count_span, 1.0)
    previous_scores = np.roll(y_np, 1)
    previous_scores[0] = float(y_np[masks["train"].detach().cpu().numpy()].mean())
    train_prior = float(y_np[masks["train"].detach().cpu().numpy()].mean())
    score_sets = {
        "always_positive": np.ones_like(y_np),
        "always_negative": np.zeros_like(y_np),
        "train_prior": np.full_like(y_np, train_prior),
        "previous_day_label": previous_scores,
        "message_volume": volume_scores,
    }
    results: dict[str, object] = {}
    for name, scores in score_sets.items():
        threshold = best_f1_threshold_from_scores(scores, labels, masks["val"])
        results[name] = {
            "threshold": threshold,
            "train": binary_metrics_from_scores(scores, labels, masks["train"], threshold),
            "val": binary_metrics_from_scores(scores, labels, masks["val"], threshold),
            "test": binary_metrics_from_scores(scores, labels, masks["test"], threshold),
        }
    return results


def count_bucket_baselines(
    labels: torch.Tensor,
    masks: dict[str, torch.Tensor],
    bucket_labels: list[str],
) -> dict[str, object]:
    y_np = labels.detach().cpu().numpy().astype(np.int64)
    train_values = y_np[masks["train"].detach().cpu().numpy()]
    majority = int(np.bincount(train_values, minlength=len(bucket_labels)).argmax())
    previous = np.roll(y_np, 1)
    previous[0] = majority
    baseline_sets = {
        "train_majority_bucket": np.full_like(y_np, majority),
        "previous_day_bucket": previous,
    }
    results: dict[str, object] = {}
    for name, preds in baseline_sets.items():
        split_metrics: dict[str, object] = {}
        for split_name, mask in masks.items():
            mask_np = mask.detach().cpu().numpy()
            y_true = y_np[mask_np]
            y_pred = preds[mask_np]
            label_ids = list(range(len(bucket_labels)))
            split_metrics[split_name] = {
                "support": int(len(y_true)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y_true, y_pred, labels=label_ids, average="weighted", zero_division=0)),
                "confusion_matrix": confusion_matrix(y_true, y_pred, labels=label_ids).tolist(),
                "confusion_labels": bucket_labels,
            }
        results[name] = split_metrics
    return results
