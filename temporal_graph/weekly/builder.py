from __future__ import annotations

import argparse
import itertools
import math
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import HeteroData

from .config import (
    ATTACK_CLASS_TO_ID,
    ATTACK_CLASSES,
    COUNT_BUCKET_LABELS,
    THREAT_TYPE_LABELS,
    THREAT_TYPE_TO_ID,
    _count_bucket,
)
from .features import (
    _build_message_features,
    _pool_mean,
    reliable_hackmageddon_months,
    target_coverage_status,
)
from .loader import load_classified_rows, load_hack_events, parse_records

try:
    from graph.build_graph import (
        add_edge_pair,
        build_edge_index,
        canonical_text,
        entity_key,
        message_similarity_edges,
        one_hot,
    )
    from graph.embeddings import encode_texts
except ModuleNotFoundError:
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).parent.parent.parent / "graph"))
    from build_graph import (  # type: ignore[no-redef]
        add_edge_pair,
        build_edge_index,
        canonical_text,
        entity_key,
        message_similarity_edges,
        one_hot,
    )
    from embeddings import encode_texts  # type: ignore[no-redef]


def build_graph(args: argparse.Namespace) -> tuple[HeteroData, dict[str, Any]]:
    if args.lookback_days < 1:
        raise ValueError("--lookback-days must be at least 1")

    rows = load_classified_rows(args.deterministic_dir)
    records = parse_records(rows)
    excluded = set(getattr(args, "exclude_groups", None) or [])
    if excluded:
        records = [r for r in records if r.group not in excluded]
    if not records:
        raise ValueError("No records with parseable dates found in the deterministic directory.")

    hack_events = load_hack_events(args.hackmageddon_csv)

    texts = [record.cleaned_text or record.text for record in records]
    hack_texts = [event.text for event in hack_events] if args.include_hackmageddon_context else []
    all_texts = texts + hack_texts

    all_embeddings, embedding_metadata = encode_texts(
        all_texts,
        backend=args.embedding_backend,
        embedding_dim=args.embedding_dim,
        max_features=args.max_features,
        random_seed=args.random_seed,
        qwen_model=args.qwen_model,
        cache_dir=args.embedding_cache,
        batch_size=args.embedding_batch_size,
    )
    message_embeddings = all_embeddings[: len(records)]
    hack_embeddings = all_embeddings[len(records) :] if args.include_hackmageddon_context else None
    embedding_dim = int(message_embeddings.shape[1])

    by_date: dict[date, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_date[record.post_date].append(index)

    hack_by_event_date: dict[date, list[int]] = defaultdict(list)
    for idx, event in enumerate(hack_events):
        hack_by_event_date[event.event_date].append(idx)
    reliable_months = reliable_hackmageddon_months(
        hack_by_event_date, args.coverage_month_min_positive_days
    )

    all_dates = sorted(by_date.keys())
    snapshot_message_indices: dict[date, list[int]] = {}
    target_classes_by_day: dict[date, list[str]] = {}
    target_types_by_day: dict[date, list[str]] = {}
    target_count_by_day: dict[date, int] = {}
    target_observed_by_day: dict[date, bool] = {}
    coverage_status_by_day: dict[date, str] = {}
    hack_context_indices_by_day: dict[date, list[int]] = {}

    for day in all_dates:
        window_start = day - timedelta(days=args.lookback_days - 1)
        window_indices = sorted(
            idx
            for d, idxs in by_date.items()
            if window_start <= d <= day
            for idx in idxs
        )
        if len(window_indices) < args.min_messages:
            continue

        target_date = day + timedelta(days=1)
        coverage_status = target_coverage_status(target_date, reliable_months)
        if args.drop_uncovered_targets and coverage_status != "observed":
            continue
        target_hack_idxs = hack_by_event_date.get(target_date, [])
        target_classes = [hack_events[i].attack_class for i in target_hack_idxs]
        target_types = [hack_events[i].threat_type for i in target_hack_idxs]

        snapshot_message_indices[day] = window_indices
        target_classes_by_day[day] = target_classes
        target_types_by_day[day] = target_types
        target_count_by_day[day] = len(target_hack_idxs)
        target_observed_by_day[day] = coverage_status == "observed"
        coverage_status_by_day[day] = coverage_status

        if args.include_hackmageddon_context:
            hack_context_indices_by_day[day] = [
                idx
                for idx, event in enumerate(hack_events)
                if window_start <= event.event_date < day
            ]

    if not snapshot_message_indices:
        raise ValueError(
            f"No snapshots built. Increase --lookback-days or reduce --min-messages "
            f"(current: {args.lookback_days} days, {args.min_messages} min messages). "
            f"Message date range: {all_dates[0] if all_dates else 'empty'} .. {all_dates[-1] if all_dates else 'empty'}"
        )

    days = sorted(snapshot_message_indices.keys())
    day_to_id = {day: idx for idx, day in enumerate(days)}

    input_indices = sorted({idx for idxs in snapshot_message_indices.values() for idx in idxs})
    input_records = [records[idx] for idx in input_indices]
    input_embeddings = message_embeddings[input_indices]
    old_to_new_msg = {old: new for new, old in enumerate(input_indices)}

    future_binary: list[float] = []
    future_count: list[float] = []
    future_count_bucket: list[int] = []
    is_label_observed: list[bool] = []
    future_multilabel = np.zeros((len(days), len(ATTACK_CLASSES)), dtype=np.float32)
    future_type_multilabel = np.zeros((len(days), len(THREAT_TYPE_LABELS)), dtype=np.float32)
    future_type_counts = np.zeros((len(days), len(THREAT_TYPE_LABELS)), dtype=np.float32)
    for i, day in enumerate(days):
        classes = target_classes_by_day.get(day, [])
        threat_types = target_types_by_day.get(day, [])
        event_count = target_count_by_day.get(day, 0)
        future_binary.append(float(bool(classes)))
        future_count.append(float(event_count))
        future_count_bucket.append(_count_bucket(event_count))
        is_label_observed.append(bool(target_observed_by_day.get(day, False)))
        for cls in classes:
            if cls in ATTACK_CLASS_TO_ID:
                future_multilabel[i, ATTACK_CLASS_TO_ID[cls]] = 1.0
        for threat_type in threat_types:
            idx = THREAT_TYPE_TO_ID.get(threat_type)
            if idx is None:
                raise ValueError(f"Unmapped threat type: {threat_type!r}")
            future_type_multilabel[i, idx] = 1.0
            future_type_counts[i, idx] += 1.0

    entity_to_id: dict[tuple[str, str], int] = {}
    message_entities: list[list[int]] = []
    entity_type_counts: Counter[str] = Counter()
    entity_sources: dict[int, set[str]] = defaultdict(set)
    entity_message_counts: Counter[int] = Counter()
    for record in input_records:
        ids: list[int] = []
        for entity in record.entities:
            key = entity_key(entity)
            if key is None:
                continue
            if key not in entity_to_id:
                entity_to_id[key] = len(entity_to_id)
                entity_type_counts[key[0]] += 1
            entity_id = entity_to_id[key]
            source = canonical_text(entity.get("source")).lower()
            if source:
                entity_sources[entity_id].add(source)
            ids.append(entity_id)
        unique_ids = sorted(set(ids))
        for entity_id in unique_ids:
            entity_message_counts[entity_id] += 1
        message_entities.append(unique_ids)

    entity_types = sorted({etype for etype, _ in entity_to_id})
    entity_type_to_id = {etype: idx for idx, etype in enumerate(entity_types)}
    groups = sorted({record.group for record in input_records})
    group_to_id = {group: idx for idx, group in enumerate(groups)}

    data = HeteroData()

    min_date = min(record.post_date for record in input_records)
    max_date = max(record.post_date for record in input_records)
    day_span = max(1, (max_date - min_date).days)
    data["message"].x = torch.tensor(
        _build_message_features(
            input_records,
            input_embeddings,
            min_date,
            day_span,
            include_absolute_time=not args.no_absolute_time,
        ),
        dtype=torch.float32,
    )
    data["message"].post_id = [record.post_id for record in input_records]
    data["message"].date = [record.post_date.isoformat() for record in input_records]

    entity_keys_by_id = [None] * len(entity_to_id)
    for key, idx in entity_to_id.items():
        entity_keys_by_id[idx] = key
    entity_features = []
    for idx, key in enumerate(entity_keys_by_id):
        assert key is not None
        etype, _ = key
        sources = entity_sources.get(idx, set())
        entity_features.append(
            one_hot(entity_type_to_id[etype], len(entity_types))
            + [math.log1p(entity_message_counts[idx]), float("regex" in sources), float("ner" in sources)]
        )
    data["entity"].x = torch.tensor(
        entity_features or np.zeros((0, len(entity_types) + 3)), dtype=torch.float32
    )
    data["entity"].entity_type = [key[0] for key in entity_keys_by_id if key is not None]
    data["entity"].value = [key[1] for key in entity_keys_by_id if key is not None]

    group_counts = Counter(record.group for record in input_records)
    data["group"].x = torch.tensor(
        [[math.log1p(group_counts[g]), group_counts[g] / max(len(input_records), 1)] for g in groups],
        dtype=torch.float32,
    )
    data["group"].name = groups

    msg_by_day: dict[date, list[int]] = {
        day: [old_to_new_msg[idx] for idx in idxs if idx in old_to_new_msg]
        for day, idxs in snapshot_message_indices.items()
    }

    day_features = []
    for day in days:
        message_ids = msg_by_day.get(day, [])
        message_pool = (
            _pool_mean(input_embeddings[message_ids], embedding_dim)
            if message_ids
            else np.zeros(embedding_dim, dtype=np.float32)
        )
        weekday = one_hot(day.weekday(), 7)
        scalar_features = [
            math.log1p(len(message_ids)),
            *weekday,
        ]
        if not args.no_absolute_time:
            min_snap = days[0]
            max_snap = days[-1]
            snap_span = max(1, (max_snap - min_snap).days)
            scalar_features.insert(1, (day - min_snap).days / snap_span)
        if args.include_hackmageddon_context and hack_embeddings is not None:
            hack_idxs = hack_context_indices_by_day.get(day, [])
            hack_pool = (
                _pool_mean(hack_embeddings[hack_idxs], embedding_dim)
                if hack_idxs
                else np.zeros(embedding_dim, dtype=np.float32)
            )
            hack_count = len(hack_idxs)
            hack_class_counts = Counter(hack_events[i].attack_class for i in hack_idxs)
            hack_features = [math.log1p(hack_count)] + [
                math.log1p(hack_class_counts.get(cls, 0)) for cls in ATTACK_CLASSES
            ]
            day_features.append(
                np.concatenate([message_pool, hack_pool, np.array(scalar_features + hack_features, dtype=np.float32)])
            )
        else:
            day_features.append(np.concatenate([message_pool, np.array(scalar_features, dtype=np.float32)]))

    data["day"].x = torch.tensor(np.vstack(day_features).astype(np.float32), dtype=torch.float32)
    data["day"].date = [day.isoformat() for day in days]
    data["day"].target_date = [(day + timedelta(days=1)).isoformat() for day in days]
    data["day"].coverage_status = [coverage_status_by_day.get(day, "uncovered_month") for day in days]
    data["day"].observed_message_count = torch.tensor([len(msg_by_day.get(day, [])) for day in days], dtype=torch.long)
    data["day"].y_future_binary = torch.tensor(future_binary, dtype=torch.float32)
    data["day"].y_future_count = torch.tensor(future_count, dtype=torch.float32)
    data["day"].y_future_count_bucket = torch.tensor(future_count_bucket, dtype=torch.long)
    data["day"].y_future_multilabel = torch.tensor(future_multilabel, dtype=torch.float32)
    data["day"].y_future_type_multilabel = torch.tensor(future_type_multilabel, dtype=torch.float32)
    data["day"].y_future_type_counts = torch.tensor(future_type_counts, dtype=torch.float32)
    data["day"].is_label_observed = torch.tensor(is_label_observed, dtype=torch.bool)
    data["day"].train_mask = data["day"].is_label_observed.clone()

    msg_entity_edges = [(mid, eid) for mid, eids in enumerate(message_entities) for eid in eids]
    msg_group_edges = [(mid, group_to_id[record.group]) for mid, record in enumerate(input_records)]
    msg_day_edges = [
        (mid, day_to_id[day])
        for day, message_ids in msg_by_day.items()
        for mid in message_ids
        if day in day_to_id
    ]
    raw_msg_msg_edges, raw_msg_msg_weights = message_similarity_edges(
        input_embeddings, args.message_top_k, args.similarity_threshold
    )
    msg_msg_edges: list[tuple[int, int]] = []
    msg_msg_weights: list[float] = []
    for (src, dst), weight in zip(raw_msg_msg_edges, raw_msg_msg_weights, strict=True):
        if input_records[src].post_date <= input_records[dst].post_date:
            msg_msg_edges.append((src, dst))
            msg_msg_weights.append(weight)

    pair_stats: dict[tuple[int, int], dict[str, Any]] = {}
    for record, eids in zip(input_records, message_entities, strict=True):
        for left, right in itertools.combinations(eids, 2):
            pair = (min(left, right), max(left, right))
            stats = pair_stats.setdefault(pair, {"count": 0, "groups": set(), "last_date": None})
            stats["count"] += 1
            stats["groups"].add(record.group)
            if stats["last_date"] is None or record.post_date > stats["last_date"]:
                stats["last_date"] = record.post_date

    ent_ent_edges: list[tuple[int, int]] = []
    ent_ent_weights: list[float] = []
    max_record_date = max((r.post_date for r in input_records), default=None)
    for (left, right), stats in pair_stats.items():
        recency = 1.0
        if max_record_date is not None and stats["last_date"] is not None and args.entity_half_life_days > 0:
            age = (max_record_date - stats["last_date"]).days
            recency = 0.5 ** (age / args.entity_half_life_days)
        weight = float(stats["count"]) * (1.0 + math.log1p(len(stats["groups"]))) * recency
        ent_ent_edges.extend([(left, right), (right, left)])
        ent_ent_weights.extend([weight, weight])

    day_day_edges = [(i, i + 1) for i in range(max(0, len(days) - 1))]

    # Keep edge name "observed_before_snapshot" so HGTFutureThreatForecaster can be reused.
    add_edge_pair(data, "message", "mentions", "entity", msg_entity_edges, "mentioned_by")
    add_edge_pair(data, "message", "posted_in", "group", msg_group_edges, "has_message")
    add_edge_pair(data, "message", "observed_before_snapshot", "day", msg_day_edges, "has_observed_message")
    add_edge_pair(data, "message", "similar_to", "message", msg_msg_edges, "rev_similar_to", msg_msg_weights)
    add_edge_pair(data, "entity", "co_occurs", "entity", ent_ent_edges, "rev_co_occurs", ent_ent_weights)
    data[("day", "next", "day")].edge_index = build_edge_index(day_day_edges)

    data.label_to_id = ATTACK_CLASS_TO_ID
    data.id_to_label = {idx: cls for cls, idx in ATTACK_CLASS_TO_ID.items()}
    data.threat_type_to_id = THREAT_TYPE_TO_ID
    data.id_to_threat_type = {idx: label for label, idx in THREAT_TYPE_TO_ID.items()}
    data.count_bucket_to_id = {label: idx for idx, label in enumerate(COUNT_BUCKET_LABELS)}
    data.id_to_count_bucket = {idx: label for idx, label in enumerate(COUNT_BUCKET_LABELS)}

    observed_counts = [len(msg_by_day.get(day, [])) for day in days]
    observed_future_binary = [value for value, observed in zip(future_binary, is_label_observed, strict=True) if observed]
    observed_future_count = [value for value, observed in zip(future_count, is_label_observed, strict=True) if observed]
    metadata: dict[str, Any] = {
        "task": "weekly_7day_lookback_nextday_threat_forecast_multitask",
        "num_classified_rows": len(rows),
        "num_records_with_date": len(records),
        "num_input_messages": len(input_records),
        "num_snapshots": len(days),
        "num_positive_snapshots": int(sum(future_binary)),
        "positive_ratio": round(sum(future_binary) / max(len(days), 1), 4),
        "num_label_observed_snapshots": int(sum(is_label_observed)),
        "num_label_uncovered_snapshots": int(len(is_label_observed) - sum(is_label_observed)),
        "num_label_observed_positive_snapshots": int(sum(observed_future_binary)),
        "label_observed_positive_ratio": round(
            sum(observed_future_binary) / max(len(observed_future_binary), 1), 4
        ),
        "label_observed_event_count": {
            "min": int(min(observed_future_count)) if observed_future_count else 0,
            "median": float(np.median(observed_future_count)) if observed_future_count else 0.0,
            "mean": float(np.mean(observed_future_count)) if observed_future_count else 0.0,
            "max": int(max(observed_future_count)) if observed_future_count else 0,
        },
        "count_bucket_labels": COUNT_BUCKET_LABELS,
        "count_bucket_observed_counts": {
            label: int(sum(1 for bucket, observed in zip(future_count_bucket, is_label_observed, strict=True) if observed and bucket == idx))
            for idx, label in enumerate(COUNT_BUCKET_LABELS)
        },
        "attack_class_labels": ATTACK_CLASSES,
        "attack_class_positive_counts": {
            cls: int(future_multilabel[:, i].sum()) for i, cls in enumerate(ATTACK_CLASSES)
        },
        "threat_type_labels": THREAT_TYPE_LABELS,
        "threat_type_positive_counts": {
            label: int(future_type_multilabel[:, i].sum()) for i, label in enumerate(THREAT_TYPE_LABELS)
        },
        "threat_type_event_counts": {
            label: int(future_type_counts[:, i].sum()) for i, label in enumerate(THREAT_TYPE_LABELS)
        },
        "target_source": "hackmageddon_date_reported_next_calendar_day",
        "target_time_granularity": "daily",
        "num_hackmageddon_events": len(hack_events),
        "num_hackmageddon_positive_days": len(hack_by_event_date),
        "hackmageddon_reliable_months": [f"{year:04d}-{month:02d}" for year, month in sorted(reliable_months)],
        "snapshot_observed_message_count": {
            "min": int(min(observed_counts)) if observed_counts else 0,
            "median": float(np.median(observed_counts)) if observed_counts else 0.0,
            "mean": float(np.mean(observed_counts)) if observed_counts else 0.0,
            "max": int(max(observed_counts)) if observed_counts else 0,
        },
        "node_types": {nt: int(data[nt].num_nodes) for nt in data.node_types},
        "edge_types": {"|".join(et): int(data[et].edge_index.size(1)) for et in data.edge_types},
        "entity_type_counts": dict(sorted(entity_type_counts.items())),
        "parameters": {
            "lookback_days": args.lookback_days,
            "min_messages": args.min_messages,
            "message_top_k": args.message_top_k,
            "similarity_threshold": args.similarity_threshold,
            "entity_half_life_days": args.entity_half_life_days,
            "include_hackmageddon_context": args.include_hackmageddon_context,
            "coverage_month_min_positive_days": args.coverage_month_min_positive_days,
            "drop_uncovered_targets": args.drop_uncovered_targets,
            "include_absolute_time": not args.no_absolute_time,
            "random_seed": args.random_seed,
            **embedding_metadata,
        },
    }
    return data, metadata
