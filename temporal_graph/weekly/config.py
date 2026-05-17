from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

try:
    from graph.label_mapping import ATTACK_TO_THREAT_LABEL, DEFAULT_THREAT_LABELS
except ModuleNotFoundError:
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).parent.parent.parent / "graph"))
    from label_mapping import ATTACK_TO_THREAT_LABEL, DEFAULT_THREAT_LABELS  # type: ignore[no-redef]

_ATTACK_CLASS_NORM: dict[str, str] = {
    "cc": "Cyber Crime",
    "cyber crime": "Cyber Crime",
    "ce": "Cyber Espionage",
    "cyber espionage": "Cyber Espionage",
    "h": "Hacktivism",
    "hacktivism": "Hacktivism",
    "cw": "Cyber Warfare",
    "cyber warfare": "Cyber Warfare",
    "unknown": "Unknown",
}
ATTACK_CLASSES = ["Cyber Crime", "Cyber Espionage", "Hacktivism", "Cyber Warfare", "Unknown"]
ATTACK_CLASS_TO_ID = {cls: idx for idx, cls in enumerate(ATTACK_CLASSES)}
THREAT_TYPE_LABELS = [
    "account_takeover",
    "malware",
    "other_cyber",
    "ransomware",
    "rare_threat",
]
_RARE_THREAT_TYPES = frozenset(DEFAULT_THREAT_LABELS) - frozenset(THREAT_TYPE_LABELS) - {"not_a_threat"}
THREAT_TYPE_TO_ID = {label: idx for idx, label in enumerate(THREAT_TYPE_LABELS)}
COUNT_BUCKET_LABELS = ["0", "1-2", "3-5", "6-10", "11-15", "16+"]


@dataclass(frozen=True)
class DatedMessageRecord:
    post_id: str
    source_file: str
    group: str
    post_date: date
    text: str
    cleaned_text: str
    ioc_present: bool
    entities: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class WeeklyHackEventRecord:
    event_id: str
    event_date: date
    attack_class: str
    threat_type: str
    text: str


def _norm_attack_class(raw: str) -> str:
    return _ATTACK_CLASS_NORM.get(raw.strip().lower(), "Unknown")


def _norm_threat_type(raw: str) -> str:
    label = ATTACK_TO_THREAT_LABEL.get(raw.strip().lower(), "other_cyber")
    return "rare_threat" if label in _RARE_THREAT_TYPES else label


def _count_bucket(count: int) -> int:
    if count <= 0:
        return 0
    if count <= 2:
        return 1
    if count <= 5:
        return 2
    if count <= 10:
        return 3
    if count <= 15:
        return 4
    return 5
