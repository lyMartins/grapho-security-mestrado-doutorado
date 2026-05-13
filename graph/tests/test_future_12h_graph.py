from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

from graph.build_future_12h_graph import build_graph
from graph.telegram_html import parse_telegram_html_dir


def write_json(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows), encoding="utf-8")


def write_html(path: Path, title: str, messages: list[tuple[int, str, str]]) -> None:
    parts = [
        "<html><body><div class='page_header'><div class='text bold'>",
        title,
        "</div></div><div class='history'>",
    ]
    for message_id, timestamp, text in messages:
        parts.extend(
            [
                f"<div class='message default clearfix' id='message{message_id}'>",
                f"<div class='pull_right date details' title='{timestamp}'>00:00</div>",
                f"<div class='text'>{text}</div>",
                "</div>",
            ]
        )
    parts.append("</div></body></html>")
    path.write_text("".join(parts), encoding="utf-8")


class Future12hGraphTest(unittest.TestCase):
    def test_parse_telegram_html_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            html_dir = Path(directory)
            write_html(
                html_dir / "messages.html",
                "Cyber Security - Information Security - IT Security - Experts",
                [(10, "01.01.2025 08:30:00 UTC-03:00", "Emotet sample")],
            )

            messages = parse_telegram_html_dir(html_dir)

            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0].message_id, 10)
            self.assertEqual(messages[0].datetime_utc.isoformat(), "2025-01-01T11:30:00+00:00")
            self.assertEqual(messages[0].text, "emotet sample")

    def test_build_future_graph_small_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            classified_dir = tmp_path / "classified"
            classified_dir.mkdir()
            html_dir = tmp_path / "html"
            html_dir.mkdir()

            write_html(
                html_dir / "messages.html",
                "Cyber Security - Information Security - IT Security - Experts",
                [
                    (1, "01.01.2025 08:00:00 UTC-03:00", "Morning cve intel"),
                    (2, "01.01.2025 14:00:00 UTC-03:00", "Afternoon emotet threat"),
                    (3, "02.01.2025 08:00:00 UTC-03:00", "Morning phishing note"),
                    (4, "02.01.2025 15:00:00 UTC-03:00", "Afternoon benign note"),
                ],
            )
            rows = [
                {
                    "source_file": "cybersecurityexperts.json",
                    "_id": 1,
                    "date": "2025-01-01",
                    "message": "Morning cve intel",
                    "id": "m1",
                    "classification": {
                        "cleaned_text": "morning cve intel",
                        "threat_label": "not_a_threat",
                        "entities": [{"type": "cve", "value": "CVE-2025-0001", "source": "regex"}],
                    },
                },
                {
                    "source_file": "cybersecurityexperts.json",
                    "_id": 2,
                    "date": "2025-01-01",
                    "message": "Afternoon emotet threat",
                    "id": "m2",
                    "classification": {
                        "cleaned_text": "afternoon emotet threat",
                        "threat_label": "malware",
                        "entities": [{"type": "malware", "value": "Emotet", "source": "ner"}],
                    },
                },
                {
                    "source_file": "cybersecurityexperts.json",
                    "_id": 3,
                    "date": "2025-01-02",
                    "message": "Morning phishing note",
                    "id": "m3",
                    "classification": {
                        "cleaned_text": "morning phishing note",
                        "threat_label": "phishing",
                        "entities": [{"type": "ttp", "value": "phishing", "source": "ner"}],
                    },
                },
                {
                    "source_file": "cybersecurityexperts.json",
                    "_id": 4,
                    "date": "2025-01-02",
                    "message": "Afternoon benign note",
                    "id": "m4",
                    "classification": {
                        "cleaned_text": "afternoon benign note",
                        "threat_label": "not_a_threat",
                        "entities": [],
                    },
                },
            ]
            write_json(classified_dir / "cybersecurityexperts.json", rows)

            hack = tmp_path / "hack.csv"
            with hack.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["row_hash", "date_reported", "author", "target", "description", "attack_norm"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "row_hash": "h1",
                        "date_reported": "2025-01-01",
                        "author": "?",
                        "target": "Org",
                        "description": "Malware incident",
                        "attack_norm": "malware",
                    }
                )

            args = argparse.Namespace(
                deterministic_dir=classified_dir,
                telegram_html_dir=html_dir,
                hackmageddon_csv=hack,
                output=tmp_path / "out.pt",
                metadata_output=None,
                snapshot_hour=12,
                horizon_hours=12,
                lookback_days=7,
                embedding_backend="tfidf_svd",
                qwen_model="Qwen/Qwen3-Embedding-0.6B",
                embedding_cache=tmp_path / "cache",
                embedding_dim=16,
                max_features=100,
                embedding_batch_size=8,
                message_top_k=1,
                similarity_threshold=0.0,
                entity_half_life_days=30.0,
                include_hackmageddon_context=False,
                random_seed=42,
                min_snapshots=2,
            )

            data, metadata = build_graph(args)

            self.assertEqual(set(data.node_types), {"message", "entity", "group", "day"})
            self.assertEqual(data["message"].num_nodes, 3)
            self.assertEqual(data["day"].num_nodes, 2)
            self.assertEqual(data["day"].y_future_binary.tolist(), [1.0, 0.0])
            self.assertEqual(data["day"].observed_message_count.tolist(), [1, 3])
            self.assertEqual(data["day"].x.size(1), 26)
            self.assertEqual(metadata["target_source"], "hackmageddon_date_reported_target_window")
            self.assertEqual(metadata["timestamp_join"]["matched_by_id"], 4)
            self.assertFalse(metadata["parameters"]["include_hackmageddon_context"])

    def test_build_future_graph_prefers_json_datetime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            classified_dir = tmp_path / "classified"
            classified_dir.mkdir()
            html_dir = tmp_path / "html"
            html_dir.mkdir()
            rows = [
                {
                    "source_file": "cybersecurityexperts.json",
                    "_id": 1,
                    "date": "2025-01-01",
                    "datetime_utc": "2025-01-01T08:00:00+00:00",
                    "message": "Morning cve intel",
                    "id": "m1",
                    "classification": {"cleaned_text": "morning cve intel", "threat_label": "not_a_threat", "entities": []},
                },
                {
                    "source_file": "cybersecurityexperts.json",
                    "_id": 2,
                    "date": "2025-01-01",
                    "datetime_utc": "2025-01-01T14:00:00+00:00",
                    "message": "Afternoon emotet threat",
                    "id": "m2",
                    "classification": {"cleaned_text": "afternoon emotet threat", "threat_label": "malware", "entities": []},
                },
                {
                    "source_file": "cybersecurityexperts.json",
                    "_id": 3,
                    "date": "2025-01-02",
                    "datetime_utc": "2025-01-02T08:00:00+00:00",
                    "message": "Morning phishing note",
                    "id": "m3",
                    "classification": {"cleaned_text": "morning phishing note", "threat_label": "phishing", "entities": []},
                },
                {
                    "source_file": "cybersecurityexperts.json",
                    "_id": 4,
                    "date": "2025-01-02",
                    "datetime_utc": "2025-01-02T15:00:00+00:00",
                    "message": "Afternoon benign note",
                    "id": "m4",
                    "classification": {"cleaned_text": "afternoon benign note", "threat_label": "not_a_threat", "entities": []},
                },
            ]
            write_json(classified_dir / "cybersecurityexperts.json", rows)
            hack = tmp_path / "hack.csv"
            hack.write_text(
                "\n".join(
                    [
                        "row_hash,date_reported,author,target,description,attack_norm",
                        "h1,2025-01-02,?,Org,Phishing incident,phishing",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                deterministic_dir=classified_dir,
                telegram_html_dir=html_dir,
                hackmageddon_csv=hack,
                output=tmp_path / "out.pt",
                metadata_output=None,
                snapshot_hour=12,
                horizon_hours=12,
                lookback_days=7,
                embedding_backend="tfidf_svd",
                qwen_model="Qwen/Qwen3-Embedding-0.6B",
                embedding_cache=tmp_path / "cache",
                embedding_dim=16,
                max_features=100,
                embedding_batch_size=8,
                message_top_k=1,
                similarity_threshold=0.0,
                entity_half_life_days=30.0,
                include_hackmageddon_context=False,
                random_seed=42,
                min_snapshots=2,
            )

            data, metadata = build_graph(args)

            self.assertEqual(data["day"].y_future_binary.tolist(), [0.0, 1.0])
            self.assertEqual(metadata["timestamp_join"]["matched_by_json_datetime"], 4)


if __name__ == "__main__":
    unittest.main()
