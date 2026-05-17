from __future__ import annotations

import math
from collections import Counter
from datetime import date

import numpy as np
from sklearn.preprocessing import normalize

from .config import DatedMessageRecord


def _pool_mean(vectors: np.ndarray, dim: int) -> np.ndarray:
    if vectors.size == 0:
        return np.zeros(dim, dtype=np.float32)
    pooled = vectors.mean(axis=0, keepdims=True).astype(np.float32)
    return normalize(pooled, norm="l2", axis=1)[0].astype(np.float32)


def _build_message_features(
    records: list[DatedMessageRecord],
    embeddings: np.ndarray,
    min_date: date,
    day_span: int,
    include_absolute_time: bool,
) -> np.ndarray:
    extras = []
    for record in records:
        values = [float(record.ioc_present), math.log1p(len(record.entities))]
        if include_absolute_time:
            temporal = (record.post_date - min_date).days / max(day_span, 1)
            values.insert(0, temporal)
        extras.append(values)
    return np.hstack([embeddings, np.array(extras, dtype=np.float32)]).astype(np.float32)


def reliable_hackmageddon_months(
    hack_by_event_date: dict[date, list[int]], min_positive_days: int
) -> set[tuple[int, int]]:
    month_positive_days: Counter[tuple[int, int]] = Counter(
        (event_date.year, event_date.month) for event_date in hack_by_event_date
    )
    return {
        month
        for month, positive_days in month_positive_days.items()
        if positive_days >= min_positive_days
    }


def target_coverage_status(target_date: date, reliable_months: set[tuple[int, int]]) -> str:
    if (target_date.year, target_date.month) in reliable_months:
        return "observed"
    return "uncovered_month"
