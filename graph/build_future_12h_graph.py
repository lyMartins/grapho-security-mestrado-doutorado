#!/usr/bin/env python3
"""Build a heterogeneous graph for 12-hour threat forecasting."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import normalize
from torch_geometric.data import HeteroData

try:
    from graph.build_graph import (
        add_edge_pair,
        build_edge_index,
        canonical_text,
        entity_key,
        message_similarity_edges,
        one_hot,
        parse_iso_date,
    )
    from graph.embeddings import encode_texts
    from graph.label_mapping import ATTACK_TO_THREAT_LABEL, build_label_mapping
    from graph.telegram_html import HtmlMessage, normalize_text, parse_telegram_html_dir
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    from build_graph import (
        add_edge_pair,
        build_edge_index,
        canonical_text,
        entity_key,
        message_similarity_edges,
        one_hot,
        parse_iso_date,
    )
    from embeddings import encode_texts
    from label_mapping import ATTACK_TO_THREAT_LABEL, build_label_mapping
    from telegram_html import HtmlMessage, normalize_text, parse_telegram_html_dir


@dataclass(frozen=True)
class TimedMessageRecord:
    post_id: str
    telegram_id: int | None
    source_file: str
    group: str
    post_datetime: datetime
    text: str
    cleaned_text: str
    threat_label: str
    severity: str
    confidence: str
    is_cyber_relevant: bool
    ioc_present: bool
    requires_human_review: bool
    entities: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class HackEventRecord:
    event_id: str
    event_date: date
    text: str
    threat_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deterministic-dir", type=Path, default=Path("sentinel_replica_jsons_deterministic"))
    parser.add_argument("--telegram-html-dir", type=Path, default=Path("ChatExport_2026-03-18/DB-telegram"))
    parser.add_argument("--hackmageddon-csv", type=Path, default=Path("hackmargeddon_data/hackmageddon_normalized.csv"))
    parser.add_argument("--output", type=Path, default=Path("graph/output/future_12h_dataset.pt"))
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--snapshot-hour", type=int, default=12)
    parser.add_argument("--horizon-hours", type=int, default=12)
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--embedding-backend", choices=["qwen", "tfidf_svd"], default="qwen")
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-cache", type=Path, default=Path("graph/cache/qwen3_embedding_0_6b"))
    parser.add_argument("--embedding-dim", type=int, default=128, help="Only used by tfidf_svd.")
    parser.add_argument("--max-features", type=int, default=20000, help="Only used by tfidf_svd.")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--message-top-k", type=int, default=8)
    parser.add_argument("--similarity-threshold", type=float, default=0.70)
    parser.add_argument("--entity-half-life-days", type=float, default=30.0)
    parser.add_argument("--include-hackmageddon-context", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--min-snapshots", type=int, default=2)
    return parser.parse_args()


def _parse_telegram_id(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_datetime_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def load_classified_rows(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        for row in payload:
            if isinstance(row, dict):
                source_file = canonical_text(row.get("source_file")) or path.name
                row = dict(row)
                row["source_file"] = source_file
                rows.append(row)
    return rows


def join_timestamps(rows: list[dict[str, Any]], html_messages: list[HtmlMessage]) -> tuple[list[TimedMessageRecord], dict[str, int]]:
    by_id = {message.message_id: message for message in html_messages}
    by_day_text: dict[tuple[str, str], list[HtmlMessage]] = defaultdict(list)
    for message in html_messages:
        by_day_text[(message.datetime_utc.date().isoformat(), message.text)].append(message)

    records: list[TimedMessageRecord] = []
    stats = Counter()
    for index, row in enumerate(rows):
        telegram_id = _parse_telegram_id(row.get("_id"))
        post_day = canonical_text(row.get("date"))
        text = canonical_text(row.get("message"))
        post_datetime = parse_datetime_utc(row.get("datetime_utc"))
        if post_datetime is not None:
            stats["matched_by_json_datetime"] += 1
        else:
            match = by_id.get(telegram_id) if telegram_id is not None else None
            if match is not None:
                post_datetime = match.datetime_utc
                stats["matched_by_id"] += 1
            else:
                match = None

        if post_datetime is None:
            candidates = by_day_text.get((post_day, normalize_text(text)), [])
            if len(candidates) == 1:
                post_datetime = candidates[0].datetime_utc
                stats["matched_by_text"] += 1
            else:
                stats["unmatched"] += 1
                continue

        classification = row.get("classification") or {}
        if not isinstance(classification, dict):
            classification = {}
        cleaning = row.get("cleaning") or {}
        if not isinstance(cleaning, dict):
            cleaning = {}
        entities = classification.get("entities") or []
        if not isinstance(entities, list):
            entities = []
        deterministic_entities = [entity for entity in entities if isinstance(entity, dict)]
        source_file = canonical_text(row.get("source_file"))
        records.append(
            TimedMessageRecord(
                post_id=canonical_text(row.get("id")) or f"{source_file}:{index}",
                telegram_id=telegram_id,
                source_file=source_file,
                group=Path(source_file).stem,
                post_datetime=post_datetime,
                text=text,
                cleaned_text=canonical_text(cleaning.get("normalized_message")),
                threat_label="unlabeled",
                severity="",
                confidence="",
                is_cyber_relevant=False,
                ioc_present=bool(deterministic_entities),
                requires_human_review=False,
                entities=tuple(deterministic_entities),
            )
        )
    return records, dict(stats)


def load_hack_events(path: Path) -> list[HackEventRecord]:
    events: list[HackEventRecord] = []
    if not path.exists():
        return events
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            event_date = parse_iso_date(canonical_text(row.get("date_reported")))
            if event_date is None:
                continue
            attack = canonical_text(row.get("attack_norm") or row.get("attack_raw")).lower()
            label = ATTACK_TO_THREAT_LABEL.get(attack, "other_cyber")
            text = " ".join(
                part
                for part in [
                    canonical_text(row.get("author")),
                    canonical_text(row.get("target")),
                    canonical_text(row.get("description")),
                    canonical_text(row.get("attack_norm") or row.get("attack_raw")),
                    canonical_text(row.get("target_class")),
                    canonical_text(row.get("attack_class")),
                ]
                if part
            )
            events.append(
                HackEventRecord(
                    event_id=canonical_text(row.get("row_hash")) or canonical_text(row.get("id_raw")),
                    event_date=event_date,
                    text=text,
                    threat_label=label,
                )
            )
    return events


def _pool_mean(vectors: np.ndarray, dim: int) -> np.ndarray:
    if vectors.size == 0:
        return np.zeros(dim, dtype=np.float32)
    pooled = vectors.mean(axis=0, keepdims=True).astype(np.float32)
    return normalize(pooled, norm="l2", axis=1)[0].astype(np.float32)


def _dates_touched_by_window(start: datetime, horizon_hours: int) -> list[date]:
    if horizon_hours <= 0:
        raise ValueError("--horizon-hours must be greater than zero")
    end_exclusive = start + timedelta(hours=horizon_hours)
    last_inclusive = end_exclusive - timedelta(microseconds=1)
    dates = []
    current = start.date()
    while current <= last_inclusive.date():
        dates.append(current)
        current += timedelta(days=1)
    return dates


def _build_message_features(records: list[TimedMessageRecord], embeddings: np.ndarray, min_dt: datetime, span_seconds: float) -> np.ndarray:
    extras = []
    for record in records:
        temporal = (record.post_datetime - min_dt).total_seconds() / max(span_seconds, 1.0)
        extras.append(
            [
                temporal,
                record.post_datetime.hour / 23.0,
                float(record.ioc_present),
                math.log1p(len(record.entities)),
            ]
        )
    return np.hstack([embeddings, np.array(extras, dtype=np.float32)]).astype(np.float32)


def build_graph(args: argparse.Namespace) -> tuple[HeteroData, dict[str, Any]]:
    if not 0 <= args.snapshot_hour <= 23:
        raise ValueError("--snapshot-hour must be between 0 and 23")
    if args.lookback_days < 0:
        raise ValueError("--lookback-days must be non-negative")

    rows = load_classified_rows(args.deterministic_dir)
    html_messages = parse_telegram_html_dir(args.telegram_html_dir)
    timed_records, timestamp_stats = join_timestamps(rows, html_messages)
    if not timed_records:
        row_dates = sorted({canonical_text(row.get("date")) for row in rows if canonical_text(row.get("date"))})
        html_dates = sorted({message.datetime_utc.date().isoformat() for message in html_messages})
        row_range = f"{row_dates[0]}..{row_dates[-1]}" if row_dates else "empty"
        html_range = f"{html_dates[0]}..{html_dates[-1]}" if html_dates else "empty"
        raise ValueError(
            "No classified messages could be matched to Telegram HTML timestamps. "
            f"classified date range={row_range}; html date range={html_range}. "
            "Use an HTML export covering the same messages/dates as the classified JSONs."
        )

    timed_record_indices = list(range(len(timed_records)))
    texts = [record.cleaned_text or record.text for record in timed_records]
    hack_events = load_hack_events(args.hackmageddon_csv)
    hack_texts = [event.text for event in hack_events]
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
    message_embeddings = all_embeddings[: len(timed_records)]
    hack_embeddings = all_embeddings[len(timed_records) :]
    embedding_dim = int(message_embeddings.shape[1])

    snapshot_time = time(hour=args.snapshot_hour, tzinfo=timezone.utc)
    by_date: dict[date, list[int]] = defaultdict(list)
    for index, record in enumerate(timed_records):
        by_date[record.post_datetime.date()].append(index)

    hack_labels_by_day: dict[date, set[str]] = defaultdict(set)
    for event in hack_events:
        if event.threat_label != "not_a_threat":
            hack_labels_by_day[event.event_date].add(event.threat_label)

    snapshot_message_indices: dict[date, list[int]] = {}
    hack_context_indices_by_day: dict[date, list[int]] = {}
    gold_labels_by_day: dict[date, set[str]] = {}
    target_dates_by_day: dict[date, list[date]] = {}
    for day, indices in sorted(by_date.items()):
        cutoff = datetime.combine(day, snapshot_time)
        same_day_before_cutoff = [index for index in indices if timed_records[index].post_datetime < cutoff]
        if not same_day_before_cutoff:
            continue

        lookback_start = cutoff - timedelta(days=args.lookback_days)
        window_indices = [
            index
            for index in timed_record_indices
            if lookback_start <= timed_records[index].post_datetime < cutoff
        ]
        if not window_indices:
            continue

        target_dates = _dates_touched_by_window(cutoff, args.horizon_hours)
        labels: set[str] = set()
        for target_date in target_dates:
            labels.update(hack_labels_by_day.get(target_date, set()))

        hack_context_indices_by_day[day] = [
            index
            for index, event in enumerate(hack_events)
            if lookback_start.date() <= event.event_date < day
        ]
        snapshot_message_indices[day] = sorted(window_indices, key=lambda index: timed_records[index].post_datetime)
        gold_labels_by_day[day] = labels
        target_dates_by_day[day] = target_dates

    input_indices = sorted({index for indices in snapshot_message_indices.values() for index in indices})
    input_records = [timed_records[index] for index in input_indices]
    input_embeddings = message_embeddings[input_indices]
    days = sorted(gold_labels_by_day)
    if len(days) < args.min_snapshots:
        raise ValueError(f"Need at least {args.min_snapshots} snapshots, found {len(days)}")
    day_to_id = {day: index for index, day in enumerate(days)}
    old_to_new_msg = {old: new for new, old in enumerate(input_indices)}

    observed_labels = set().union(*gold_labels_by_day.values())
    label_to_id = build_label_mapping(set(observed_labels))
    future_multilabel = np.zeros((len(days), len(label_to_id)), dtype=np.float32)
    future_binary = []
    for day in days:
        labels = gold_labels_by_day.get(day, set())
        future_binary.append(float(bool(labels)))
        for label in labels:
            future_multilabel[day_to_id[day], label_to_id[label]] = 1.0

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

    entity_types = sorted({entity_type for entity_type, _ in entity_to_id})
    entity_type_to_id = {entity_type: index for index, entity_type in enumerate(entity_types)}
    groups = sorted({record.group for record in input_records})
    group_to_id = {group: index for index, group in enumerate(groups)}

    data = HeteroData()
    min_dt = min(record.post_datetime for record in input_records)
    max_dt = max(record.post_datetime for record in input_records)
    data["message"].x = torch.tensor(
        _build_message_features(input_records, input_embeddings, min_dt, (max_dt - min_dt).total_seconds()),
        dtype=torch.float32,
    )
    data["message"].post_id = [record.post_id for record in input_records]
    data["message"].telegram_id = [record.telegram_id or -1 for record in input_records]
    data["message"].datetime_utc = [record.post_datetime.isoformat() for record in input_records]
    data["message"].date = [record.post_datetime.date().isoformat() for record in input_records]
    data["message"].threat_label = [record.threat_label for record in input_records]

    entity_keys_by_id = [None] * len(entity_to_id)
    for key, index in entity_to_id.items():
        entity_keys_by_id[index] = key
    entity_features = []
    for index, key in enumerate(entity_keys_by_id):
        assert key is not None
        entity_type, _ = key
        sources = entity_sources.get(index, set())
        entity_features.append(
            one_hot(entity_type_to_id[entity_type], len(entity_types))
            + [
                math.log1p(entity_message_counts[index]),
                float("regex" in sources),
                float("ner" in sources),
            ]
        )
    data["entity"].x = torch.tensor(entity_features or np.zeros((0, len(entity_types) + 3)), dtype=torch.float32)
    data["entity"].entity_type = [key[0] for key in entity_keys_by_id if key is not None]
    data["entity"].value = [key[1] for key in entity_keys_by_id if key is not None]

    group_counts = Counter(record.group for record in input_records)
    data["group"].x = torch.tensor(
        [[math.log1p(group_counts[group]), group_counts[group] / max(len(input_records), 1)] for group in groups],
        dtype=torch.float32,
    )
    data["group"].name = groups

    hack_by_day: dict[date, list[int]] = defaultdict(list)
    for index, event in enumerate(hack_events):
        hack_by_day[event.event_date].append(index)
    msg_by_day: dict[date, list[int]] = {
        day: [old_to_new_msg[index] for index in indices if index in old_to_new_msg]
        for day, indices in snapshot_message_indices.items()
    }

    day_features = []
    min_day = days[0]
    max_day = days[-1]
    day_span = max(1, (max_day - min_day).days)
    for day in days:
        message_ids = msg_by_day.get(day, [])
        message_pool = _pool_mean(input_embeddings[message_ids], embedding_dim) if message_ids else np.zeros(embedding_dim, dtype=np.float32)
        hack_ids = hack_context_indices_by_day.get(day, [])
        hack_pool = _pool_mean(hack_embeddings[hack_ids], embedding_dim) if hack_ids else np.zeros(embedding_dim, dtype=np.float32)
        hack_count = len(hack_ids)
        hack_threat_count = sum(1 for index in hack_ids if hack_events[index].threat_label != "not_a_threat")
        weekday = one_hot(day.weekday(), 7)
        message_features = [
            math.log1p(len(message_ids)),
            args.snapshot_hour / 23.0,
            (day - min_day).days / day_span,
            *weekday,
        ]
        if args.include_hackmageddon_context:
            hack_features = [
                math.log1p(hack_count),
                math.log1p(hack_threat_count),
                float(hack_threat_count > 0),
            ]
            day_features.append(
                np.concatenate([message_pool, hack_pool, np.array(message_features + hack_features, dtype=np.float32)])
            )
        else:
            day_features.append(np.concatenate([message_pool, np.array(message_features, dtype=np.float32)]))
    data["day"].x = torch.tensor(np.vstack(day_features).astype(np.float32), dtype=torch.float32)
    data["day"].date = [day.isoformat() for day in days]
    data["day"].target_dates = [[target.isoformat() for target in target_dates_by_day[day]] for day in days]
    data["day"].observed_message_count = torch.tensor([len(msg_by_day.get(day, [])) for day in days], dtype=torch.long)
    data["day"].y_future_binary = torch.tensor(future_binary, dtype=torch.float32)
    data["day"].y_future_multilabel = torch.tensor(future_multilabel, dtype=torch.float32)
    data["day"].train_mask = torch.ones(len(days), dtype=torch.bool)

    msg_entity_edges = [(msg_id, entity_id) for msg_id, ids in enumerate(message_entities) for entity_id in ids]
    msg_group_edges = [(msg_id, group_to_id[record.group]) for msg_id, record in enumerate(input_records)]
    msg_day_edges = [
        (msg_id, day_to_id[day])
        for day, message_ids in msg_by_day.items()
        for msg_id in message_ids
        if day in day_to_id
    ]
    raw_msg_msg_edges, raw_msg_msg_weights = message_similarity_edges(input_embeddings, args.message_top_k, args.similarity_threshold)
    msg_msg_edges: list[tuple[int, int]] = []
    msg_msg_weights: list[float] = []
    for (src, dst), weight in zip(raw_msg_msg_edges, raw_msg_msg_weights, strict=True):
        if input_records[src].post_datetime <= input_records[dst].post_datetime:
            msg_msg_edges.append((src, dst))
            msg_msg_weights.append(weight)

    pair_stats: dict[tuple[int, int], dict[str, Any]] = {}
    for record, ids in zip(input_records, message_entities, strict=True):
        for left, right in itertools.combinations(ids, 2):
            pair = (min(left, right), max(left, right))
            stats = pair_stats.setdefault(pair, {"count": 0, "groups": set(), "last_date": None})
            stats["count"] += 1
            stats["groups"].add(record.group)
            record_date = record.post_datetime.date()
            if stats["last_date"] is None or record_date > stats["last_date"]:
                stats["last_date"] = record_date
    ent_ent_edges: list[tuple[int, int]] = []
    ent_ent_weights: list[float] = []
    max_record_date = max((record.post_datetime.date() for record in input_records), default=None)
    for (left, right), stats in pair_stats.items():
        recency = 1.0
        if max_record_date is not None and stats["last_date"] is not None and args.entity_half_life_days > 0:
            age = (max_record_date - stats["last_date"]).days
            recency = 0.5 ** (age / args.entity_half_life_days)
        weight = float(stats["count"]) * (1.0 + math.log1p(len(stats["groups"]))) * recency
        ent_ent_edges.extend([(left, right), (right, left)])
        ent_ent_weights.extend([weight, weight])

    day_day_edges = [(index, index + 1) for index in range(max(0, len(days) - 1))]
    add_edge_pair(data, "message", "mentions", "entity", msg_entity_edges, "mentioned_by")
    add_edge_pair(data, "message", "posted_in", "group", msg_group_edges, "has_message")
    add_edge_pair(data, "message", "observed_before_snapshot", "day", msg_day_edges, "has_observed_message")
    add_edge_pair(data, "message", "similar_to", "message", msg_msg_edges, "rev_similar_to", msg_msg_weights)
    add_edge_pair(data, "entity", "co_occurs", "entity", ent_ent_edges, "rev_co_occurs", ent_ent_weights)
    data[("day", "next", "day")].edge_index = build_edge_index(day_day_edges)

    data.label_to_id = label_to_id
    data.id_to_label = {index: label for label, index in label_to_id.items()}
    observed_counts = [len(msg_by_day.get(day, [])) for day in days]
    metadata = {
        "task": "future_12h_threat_forecast",
        "num_classified_rows": len(rows),
        "num_html_messages": len(html_messages),
        "num_timed_messages": len(timed_records),
        "num_input_messages": len(input_records),
        "num_snapshots": len(days),
        "num_positive_snapshots": int(sum(future_binary)),
        "target_source": "hackmageddon_date_reported_target_window",
        "target_time_granularity": "daily_date_reported",
        "uses_llm_features": False,
        "message_text_source": "cleaning.normalized_message_or_raw_message",
        "entity_source_filter": "deterministic_regex_and_rule_ner",
        "num_hackmageddon_events": len(hack_events),
        "num_hackmageddon_positive_days": len(hack_labels_by_day),
        "timestamp_join": timestamp_stats,
        "snapshot_observed_message_count": {
            "min": int(min(observed_counts)) if observed_counts else 0,
            "median": float(np.median(observed_counts)) if observed_counts else 0.0,
            "mean": float(np.mean(observed_counts)) if observed_counts else 0.0,
            "max": int(max(observed_counts)) if observed_counts else 0,
        },
        "node_types": {node_type: int(data[node_type].num_nodes) for node_type in data.node_types},
        "edge_types": {"|".join(edge_type): int(data[edge_type].edge_index.size(1)) for edge_type in data.edge_types},
        "label_to_id": label_to_id,
        "entity_type_counts": dict(sorted(entity_type_counts.items())),
        "parameters": {
            "snapshot_hour": args.snapshot_hour,
            "horizon_hours": args.horizon_hours,
            "lookback_days": args.lookback_days,
            "message_top_k": args.message_top_k,
            "similarity_threshold": args.similarity_threshold,
            "include_hackmageddon_context": args.include_hackmageddon_context,
            "random_seed": args.random_seed,
            **embedding_metadata,
        },
    }
    return data, metadata


def main() -> None:
    args = parse_args()
    data, metadata = build_graph(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.output)
    metadata_output = args.metadata_output or args.output.with_name("future_12h_metadata.json")
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved graph: {args.output}")
    print(f"saved metadata: {metadata_output}")
    print(json.dumps(metadata["node_types"], ensure_ascii=False))
    print(json.dumps(metadata["edge_types"], ensure_ascii=False))
    print(f"positive snapshots: {metadata['num_positive_snapshots']}/{metadata['num_snapshots']}")


if __name__ == "__main__":
    main()
