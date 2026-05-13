#!/usr/bin/env python3
"""Build the Option A heterogeneous graph as a PyG HeteroData artifact."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from torch_geometric.data import HeteroData

try:
    from graph.label_mapping import ATTACK_TO_THREAT_LABEL, build_label_mapping
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    from label_mapping import ATTACK_TO_THREAT_LABEL, build_label_mapping


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class MessageRecord:
    post_id: str
    source_file: str
    group: str
    post_date: date | None
    text: str
    cleaned_text: str
    threat_label: str
    severity: str
    confidence: str
    is_cyber_relevant: bool
    ioc_present: bool
    requires_human_review: bool
    entities: tuple[dict[str, Any], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deterministic-dir",
        "--classified-dir",
        dest="classified_dir",
        type=Path,
        default=Path("sentinel_replica_jsons_deterministic"),
        help=(
            "Directory containing deterministic SENTINEL replica JSONs. "
            "--classified-dir is kept as a backward-compatible alias."
        ),
    )
    parser.add_argument("--hackmageddon-csv", type=Path, default=Path("hackmargeddon_data/hackmageddon_normalized.csv"))
    parser.add_argument("--matched-records", type=Path, default=Path("eval/output/matched_records.csv"))
    parser.add_argument("--output", type=Path, default=Path("graph/output/heterodata_option_a.pt"))
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--message-top-k", type=int, default=8)
    parser.add_argument("--similarity-threshold", type=float, default=0.35)
    parser.add_argument("--entity-half-life-days", type=float, default=30.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--min-messages", type=int, default=2)
    return parser.parse_args()


def parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    value = value[:10]
    if not DATE_RE.match(value):
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def canonical_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalize_entity_value(entity_type: str, value: Any) -> str:
    text = canonical_text(value)
    if entity_type == "cve":
        return text.upper()
    if entity_type in {"domain", "url", "email", "handle", "ip_address", "hash", "wallet"}:
        return text.lower()
    return text.casefold()


def entity_key(entity: dict[str, Any]) -> tuple[str, str] | None:
    entity_type = canonical_text(entity.get("type")).lower()
    value = normalize_entity_value(entity_type, entity.get("value"))
    if not entity_type or not value:
        return None
    return entity_type, value


def load_deterministic_records(directory: Path) -> list[MessageRecord]:
    records: list[MessageRecord] = []
    for path in sorted(directory.glob("*.json")):
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, list):
            continue
        for index, row in enumerate(payload):
            if not isinstance(row, dict):
                continue
            classification = row.get("classification") or {}
            if not isinstance(classification, dict):
                classification = {}
            source_file = canonical_text(row.get("source_file")) or path.name
            group = Path(source_file).stem
            post_id = canonical_text(row.get("id")) or f"{path.name}:{index}"
            entities = classification.get("entities") or []
            if not isinstance(entities, list):
                entities = []
            records.append(
                MessageRecord(
                    post_id=post_id,
                    source_file=source_file,
                    group=group,
                    post_date=parse_iso_date(canonical_text(row.get("date"))),
                    text=canonical_text(row.get("message")),
                    cleaned_text=canonical_text(classification.get("cleaned_text")),
                    threat_label=canonical_text(classification.get("threat_label")) or "not_a_threat",
                    severity=canonical_text(classification.get("severity")),
                    confidence=canonical_text(classification.get("confidence")),
                    is_cyber_relevant=bool(classification.get("is_cyber_relevant", False)),
                    ioc_present=bool(classification.get("ioc_present", False)),
                    requires_human_review=bool(classification.get("requires_human_review", False)),
                    entities=tuple(entity for entity in entities if isinstance(entity, dict)),
                )
            )
    return records


def load_gold_labels(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    labels: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            post_id = canonical_text(row.get("post_id"))
            label = canonical_text(row.get("gold_threat_label"))
            if post_id and label:
                labels[post_id] = label
    return labels


def load_hackmageddon_day_counts(path: Path) -> tuple[Counter[str], Counter[str]]:
    event_counts: Counter[str] = Counter()
    threat_day_counts: Counter[str] = Counter()
    if not path.exists():
        return event_counts, threat_day_counts
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            day = canonical_text(row.get("date_reported"))
            if not day:
                continue
            event_counts[day] += 1
            attack = canonical_text(row.get("attack_norm") or row.get("attack_raw")).lower()
            label = ATTACK_TO_THREAT_LABEL.get(attack, "other_cyber")
            if label != "not_a_threat":
                threat_day_counts[day] += 1
    return event_counts, threat_day_counts


def fit_message_embeddings(
    records: list[MessageRecord],
    embedding_dim: int,
    max_features: int,
    random_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    texts = [record.cleaned_text or record.text for record in records]
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1,
        strip_accents="unicode",
        lowercase=True,
    )
    tfidf = vectorizer.fit_transform(texts)
    max_components = max(1, min(embedding_dim, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
    if max_components < 2:
        dense = tfidf.toarray().astype(np.float32)
    else:
        svd = TruncatedSVD(n_components=max_components, random_state=random_seed)
        dense = svd.fit_transform(tfidf).astype(np.float32)
    if dense.shape[1] > embedding_dim:
        dense = dense[:, :embedding_dim]
    if dense.shape[1] < embedding_dim:
        dense = np.pad(dense, ((0, 0), (0, embedding_dim - dense.shape[1])))
    dense = normalize(dense, norm="l2", axis=1).astype(np.float32)
    metadata = {
        "embedding_backend": "tfidf_svd",
        "embedding_dim": embedding_dim,
        "max_features": max_features,
        "tfidf_features": int(tfidf.shape[1]),
        "svd_components": int(max_components),
    }
    return dense, metadata


def one_hot(index: int, size: int) -> list[float]:
    values = [0.0] * size
    if 0 <= index < size:
        values[index] = 1.0
    return values


def build_edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def add_edge_pair(
    data: HeteroData,
    src_type: str,
    relation: str,
    dst_type: str,
    edges: list[tuple[int, int]],
    reverse_relation: str,
    edge_weight: list[float] | None = None,
) -> None:
    data[(src_type, relation, dst_type)].edge_index = build_edge_index(edges)
    if edge_weight is not None:
        data[(src_type, relation, dst_type)].edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
    reverse_edges = [(dst, src) for src, dst in edges]
    data[(dst_type, reverse_relation, src_type)].edge_index = build_edge_index(reverse_edges)
    if edge_weight is not None:
        data[(dst_type, reverse_relation, src_type)].edge_weight = torch.tensor(edge_weight, dtype=torch.float32)


def message_similarity_edges(
    embeddings: np.ndarray,
    top_k: int,
    threshold: float,
) -> tuple[list[tuple[int, int]], list[float]]:
    if len(embeddings) < 2 or top_k <= 0:
        return [], []
    n_neighbors = min(top_k + 1, len(embeddings))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    seen: set[tuple[int, int]] = set()
    for src, (row_distances, row_indices) in enumerate(zip(distances, indices, strict=True)):
        for distance, dst in zip(row_distances, row_indices, strict=True):
            dst = int(dst)
            if src == dst:
                continue
            similarity = float(1.0 - distance)
            if similarity < threshold:
                continue
            edge = (src, dst)
            if edge in seen:
                continue
            seen.add(edge)
            edges.append(edge)
            weights.append(similarity)
    return edges, weights


def build_graph(args: argparse.Namespace) -> tuple[HeteroData, dict[str, Any]]:
    records = load_deterministic_records(args.classified_dir)
    if len(records) < args.min_messages:
        raise ValueError(f"Need at least {args.min_messages} messages, found {len(records)}")

    gold_by_post_id = load_gold_labels(args.matched_records)
    hack_day_counts, hack_threat_day_counts = load_hackmageddon_day_counts(args.hackmageddon_csv)
    observed_labels = {record.threat_label for record in records} | set(gold_by_post_id.values())
    label_to_id = build_label_mapping(observed_labels)

    embeddings, embedding_metadata = fit_message_embeddings(
        records,
        embedding_dim=args.embedding_dim,
        max_features=args.max_features,
        random_seed=args.random_seed,
    )

    groups = sorted({record.group for record in records})
    group_to_id = {group: index for index, group in enumerate(groups)}
    dated_records = [record for record in records if record.post_date is not None]
    days = sorted({record.post_date.isoformat() for record in dated_records})
    day_to_id = {day: index for index, day in enumerate(days)}

    entity_to_id: dict[tuple[str, str], int] = {}
    message_entities: list[list[int]] = []
    entity_type_counts: Counter[str] = Counter()
    entity_sources: dict[int, set[str]] = defaultdict(set)
    entity_message_counts: Counter[int] = Counter()
    for record in records:
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

    data = HeteroData()
    message_extra = []
    min_day = min((record.post_date for record in dated_records), default=None)
    max_day = max((record.post_date for record in dated_records), default=None)
    date_span = max(1, (max_day - min_day).days if min_day and max_day else 1)
    severity_to_score = {"": 0.0, "informational": 0.1, "low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
    confidence_to_score = {"": 0.0, "low": 0.33, "medium": 0.66, "high": 1.0}
    for record in records:
        if record.post_date and min_day:
            day_value = (record.post_date - min_day).days / date_span
        else:
            day_value = 0.0
        message_extra.append(
            [
                day_value,
                float(record.is_cyber_relevant),
                float(record.ioc_present),
                float(record.requires_human_review),
                severity_to_score.get(record.severity.lower(), 0.0),
                confidence_to_score.get(record.confidence.lower(), 0.0),
                math.log1p(len(record.entities)),
            ]
        )
    data["message"].x = torch.tensor(np.hstack([embeddings, np.array(message_extra, dtype=np.float32)]), dtype=torch.float32)
    data["message"].post_id = [record.post_id for record in records]
    data["message"].source_file = [record.source_file for record in records]
    data["message"].date = [record.post_date.isoformat() if record.post_date else "" for record in records]
    data["message"].y_weak = torch.tensor([label_to_id[record.threat_label] for record in records], dtype=torch.long)
    data["message"].y_gold = torch.tensor(
        [label_to_id.get(gold_by_post_id.get(record.post_id, ""), -1) for record in records],
        dtype=torch.long,
    )
    data["message"].gold_mask = data["message"].y_gold >= 0
    data["message"].train_mask = data["message"].y_weak >= 0
    data["message"].sample_weight = torch.where(data["message"].gold_mask, torch.tensor(3.0), torch.tensor(1.0))

    entity_features = []
    entity_keys_by_id = [None] * len(entity_to_id)
    for key, index in entity_to_id.items():
        entity_keys_by_id[index] = key
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

    group_counts = Counter(record.group for record in records)
    data["group"].x = torch.tensor(
        [[math.log1p(group_counts[group]), group_counts[group] / len(records)] for group in groups],
        dtype=torch.float32,
    )
    data["group"].name = groups

    message_counts_by_day = Counter(record.post_date.isoformat() for record in dated_records)
    day_features = []
    for day in days:
        day_features.append(
            [
                math.log1p(message_counts_by_day[day]),
                math.log1p(hack_day_counts[day]),
                math.log1p(hack_threat_day_counts[day]),
                float(hack_threat_day_counts[day] > 0),
            ]
        )
    data["day"].x = torch.tensor(day_features or np.zeros((0, 4)), dtype=torch.float32)
    data["day"].date = days

    msg_entity_edges = [(msg_id, entity_id) for msg_id, ids in enumerate(message_entities) for entity_id in ids]
    msg_group_edges = [(msg_id, group_to_id[record.group]) for msg_id, record in enumerate(records)]
    msg_day_edges = [
        (msg_id, day_to_id[record.post_date.isoformat()])
        for msg_id, record in enumerate(records)
        if record.post_date is not None
    ]
    msg_msg_edges, msg_msg_weights = message_similarity_edges(embeddings, args.message_top_k, args.similarity_threshold)

    pair_stats: dict[tuple[int, int], dict[str, Any]] = {}
    max_record_date = max_day
    for record, ids in zip(records, message_entities, strict=True):
        for left, right in itertools.combinations(ids, 2):
            pair = (min(left, right), max(left, right))
            stats = pair_stats.setdefault(pair, {"count": 0, "groups": set(), "last_date": None})
            stats["count"] += 1
            stats["groups"].add(record.group)
            if record.post_date and (stats["last_date"] is None or record.post_date > stats["last_date"]):
                stats["last_date"] = record.post_date
    ent_ent_edges: list[tuple[int, int]] = []
    ent_ent_weights: list[float] = []
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
    add_edge_pair(data, "message", "posted_on", "day", msg_day_edges, "has_message")
    add_edge_pair(data, "message", "similar_to", "message", msg_msg_edges, "rev_similar_to", msg_msg_weights)
    add_edge_pair(data, "entity", "co_occurs", "entity", ent_ent_edges, "rev_co_occurs", ent_ent_weights)
    add_edge_pair(data, "day", "next", "day", day_day_edges, "previous")

    data.label_to_id = label_to_id
    data.id_to_label = {index: label for label, index in label_to_id.items()}
    metadata = {
        "num_messages": len(records),
        "num_gold_messages": int(data["message"].gold_mask.sum().item()),
        "num_entities": len(entity_to_id),
        "num_groups": len(groups),
        "num_days": len(days),
        "node_types": {node_type: int(data[node_type].num_nodes) for node_type in data.node_types},
        "edge_types": {"|".join(edge_type): int(data[edge_type].edge_index.size(1)) for edge_type in data.edge_types},
        "label_to_id": label_to_id,
        "entity_type_counts": dict(sorted(entity_type_counts.items())),
        "parameters": {
            "message_top_k": args.message_top_k,
            "similarity_threshold": args.similarity_threshold,
            "entity_half_life_days": args.entity_half_life_days,
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
    metadata_output = args.metadata_output or args.output.with_name("metadata.json")
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved graph: {args.output}")
    print(f"saved metadata: {metadata_output}")
    print(json.dumps(metadata["node_types"], ensure_ascii=False))
    print(json.dumps(metadata["edge_types"], ensure_ascii=False))


if __name__ == "__main__":
    main()
