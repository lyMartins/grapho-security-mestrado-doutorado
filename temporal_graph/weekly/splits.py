from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _has_multiple_classes(labels: torch.Tensor, indices: torch.Tensor) -> bool:
    if indices.numel() < 2:
        return False
    values = labels[indices].detach().cpu().to(torch.int64)
    return len(set(values.tolist())) > 1


def chronological_masks(
    labels: torch.Tensor, eligible_mask: torch.Tensor, val_size: float, test_size: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = torch.nonzero(eligible_mask.detach().cpu(), as_tuple=False).flatten()
    num_eligible = int(indices.numel())
    if num_eligible < 3:
        raise ValueError(f"Need at least 3 label-observed snapshots, found {num_eligible}.")
    test_count = max(1, int(num_eligible * test_size))
    val_count = max(1, int(num_eligible * val_size))
    train_count = max(1, num_eligible - val_count - test_count)
    if train_count + val_count + test_count > num_eligible:
        train_count = max(1, num_eligible - val_count - test_count)
    train = torch.zeros(labels.numel(), dtype=torch.bool)
    val = torch.zeros(labels.numel(), dtype=torch.bool)
    test = torch.zeros(labels.numel(), dtype=torch.bool)
    train[indices[:train_count]] = True
    val[indices[train_count : train_count + val_count]] = True
    test[indices[train_count + val_count :]] = True
    return train, val, test


def balanced_chronological_masks(
    labels: torch.Tensor, eligible_mask: torch.Tensor, val_size: float, test_size: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = torch.nonzero(eligible_mask.detach().cpu(), as_tuple=False).flatten()
    num_eligible = int(indices.numel())
    if num_eligible < 6:
        raise ValueError(f"Need at least 6 label-observed snapshots for balanced split, found {num_eligible}.")

    nominal_test = max(2, int(num_eligible * test_size))
    nominal_val = max(2, int(num_eligible * val_size))
    nominal_train = num_eligible - nominal_val - nominal_test
    candidates: list[tuple[int, int, int]] = []
    train_options = sorted(range(2, num_eligible - 3), key=lambda count: abs(count - nominal_train))
    for train_count in train_options:
        max_val = num_eligible - train_count - 2
        if max_val < 2:
            continue
        val_options = sorted(range(2, max_val + 1), key=lambda count: abs(count - nominal_val))
        for val_count in val_options:
            test_count = num_eligible - train_count - val_count
            if test_count < 2:
                continue
            train_idx = indices[:train_count]
            val_idx = indices[train_count : train_count + val_count]
            test_idx = indices[train_count + val_count :]
            if all(
                _has_multiple_classes(labels.detach().cpu(), split_idx)
                for split_idx in (train_idx, val_idx, test_idx)
            ):
                distance = abs(train_count - nominal_train) + abs(val_count - nominal_val) + abs(test_count - nominal_test)
                candidates.append((distance, train_count, val_count))
                break
    if not candidates:
        raise ValueError(
            "Could not build chronological train/val/test masks with both classes in every split. "
            "Use a different coverage threshold, split size, or inspect label coverage."
        )
    _, train_count, val_count = min(candidates, key=lambda item: item[0])
    train = torch.zeros(labels.numel(), dtype=torch.bool)
    val = torch.zeros(labels.numel(), dtype=torch.bool)
    test = torch.zeros(labels.numel(), dtype=torch.bool)
    train[indices[:train_count]] = True
    val[indices[train_count : train_count + val_count]] = True
    test[indices[train_count + val_count :]] = True
    return train, val, test
