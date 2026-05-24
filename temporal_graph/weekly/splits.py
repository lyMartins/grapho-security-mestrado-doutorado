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


def _validate_split_sizes(val_size: float, test_size: float) -> None:
    if not 0.0 < val_size < 1.0:
        raise ValueError(f"val_size must be between 0 and 1, got {val_size}.")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}.")
    if val_size + test_size >= 1.0:
        raise ValueError(
            f"val_size + test_size must be lower than 1, got {val_size + test_size}."
        )


def _target_split_counts(num_items: int, val_size: float, test_size: float) -> tuple[int, int, int]:
    test_count = max(1, int(round(num_items * test_size)))
    val_count = max(1, int(round(num_items * val_size)))
    train_count = num_items - val_count - test_count
    while train_count < 1:
        if test_count >= val_count and test_count > 1:
            test_count -= 1
        elif val_count > 1:
            val_count -= 1
        else:
            break
        train_count = num_items - val_count - test_count
    if train_count < 1:
        raise ValueError(f"Need at least 3 label-observed snapshots, found {num_items}.")
    return train_count, val_count, test_count


def _stratified_take(
    labels: torch.Tensor,
    indices: torch.Tensor,
    count: int,
    rng: np.random.Generator,
    min_remaining_per_class: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if count <= 0:
        return indices[:0], indices

    labels_cpu = labels.detach().cpu().to(torch.int64)
    groups: dict[int, np.ndarray] = {}
    for label in sorted(set(labels_cpu[indices].tolist())):
        class_indices = indices[labels_cpu[indices] == label].detach().cpu().numpy()
        rng.shuffle(class_indices)
        groups[int(label)] = class_indices

    reserve = min_remaining_per_class
    while reserve > 0 and sum(max(0, len(group) - reserve) for group in groups.values()) < count:
        reserve -= 1

    selectable = {label: max(0, len(group) - reserve) for label, group in groups.items()}
    selectable_total = sum(selectable.values())
    if selectable_total < count:
        raise ValueError(f"Cannot take {count} stratified samples from {int(indices.numel())} candidates.")

    quotas = {
        label: (count * class_count / selectable_total if selectable_total else 0.0)
        for label, class_count in selectable.items()
    }
    take_counts = {label: min(selectable[label], int(np.floor(quota))) for label, quota in quotas.items()}
    remaining = count - sum(take_counts.values())
    remainders = sorted(
        ((quotas[label] - take_counts[label], selectable[label], label) for label in groups),
        reverse=True,
    )
    while remaining > 0:
        allocated = False
        for _, _, label in remainders:
            if take_counts[label] < selectable[label]:
                take_counts[label] += 1
                remaining -= 1
                allocated = True
                if remaining == 0:
                    break
        if not allocated:
            break

    selected: list[int] = []
    kept: list[int] = []
    for label, group in groups.items():
        take_count = take_counts[label]
        selected.extend(group[:take_count].tolist())
        kept.extend(group[take_count:].tolist())

    selected_array = np.array(selected, dtype=np.int64)
    kept_array = np.array(kept, dtype=np.int64)
    rng.shuffle(selected_array)
    rng.shuffle(kept_array)
    return torch.as_tensor(selected_array, dtype=torch.long), torch.as_tensor(kept_array, dtype=torch.long)


def stratified_masks(
    labels: torch.Tensor,
    eligible_mask: torch.Tensor,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_split_sizes(val_size, test_size)
    indices = torch.nonzero(eligible_mask.detach().cpu(), as_tuple=False).flatten()
    num_eligible = int(indices.numel())
    if num_eligible < 3:
        raise ValueError(f"Need at least 3 label-observed snapshots, found {num_eligible}.")

    _, val_count, test_count = _target_split_counts(num_eligible, val_size, test_size)
    rng = np.random.default_rng(seed)
    test_idx, remaining_idx = _stratified_take(
        labels, indices, test_count, rng, min_remaining_per_class=2
    )
    val_idx, train_idx = _stratified_take(
        labels, remaining_idx, val_count, rng, min_remaining_per_class=1
    )

    train = torch.zeros(labels.numel(), dtype=torch.bool)
    val = torch.zeros(labels.numel(), dtype=torch.bool)
    test = torch.zeros(labels.numel(), dtype=torch.bool)
    train[train_idx] = True
    val[val_idx] = True
    test[test_idx] = True
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
