from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config import DatedMessageRecord, WeeklyHackEventRecord, _norm_attack_class, _norm_threat_type

try:
    from graph.build_graph import canonical_text, entity_key, parse_iso_date
except ModuleNotFoundError:
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).parent.parent.parent / "graph"))
    from build_graph import canonical_text, entity_key, parse_iso_date  # type: ignore[no-redef]


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


def parse_records(rows: list[dict[str, Any]]) -> list[DatedMessageRecord]:
    records: list[DatedMessageRecord] = []
    for index, row in enumerate(rows):
        post_date = parse_iso_date(canonical_text(row.get("date")))
        if post_date is None:
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
        det_entities = [e for e in entities if isinstance(e, dict)]
        source_file = canonical_text(row.get("source_file"))
        records.append(
            DatedMessageRecord(
                post_id=canonical_text(row.get("id")) or f"{source_file}:{index}",
                source_file=source_file,
                group=Path(source_file).stem,
                post_date=post_date,
                text=canonical_text(row.get("message")) or "",
                cleaned_text=canonical_text(cleaning.get("normalized_message")) or "",
                ioc_present=bool(det_entities),
                entities=tuple(det_entities),
            )
        )
    return records


def load_hack_events(path: Path) -> list[WeeklyHackEventRecord]:
    events: list[WeeklyHackEventRecord] = []
    if not path.exists():
        return events
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            event_date = parse_iso_date(canonical_text(row.get("date_reported")))
            if event_date is None:
                continue
            raw_class = canonical_text(row.get("attack_class")) or "Unknown"
            attack_class = _norm_attack_class(raw_class)
            attack_norm = canonical_text(row.get("attack_norm") or row.get("attack_raw"))
            threat_type = _norm_threat_type(attack_norm)
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
                WeeklyHackEventRecord(
                    event_id=canonical_text(row.get("row_hash")) or canonical_text(row.get("id_raw")) or "",
                    event_date=event_date,
                    attack_class=attack_class,
                    threat_type=threat_type,
                    text=text,
                )
            )
    return events
