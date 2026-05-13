from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

from graph.build_graph import build_graph


def write_json(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows), encoding="utf-8")


class BuildGraphTest(unittest.TestCase):
    def test_build_graph_small_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            classified_dir = tmp_path / "classified"
            classified_dir.mkdir()
            rows = [
                {
                    "source_file": "group_a.json",
                    "date": "2025-01-01",
                    "message": "Emotet using CVE-2025-0001 from 10.0.0.1",
                    "id": "m1",
                    "classification": {
                        "cleaned_text": "emotet using cve 2025 0001 from 10.0.0.1",
                        "threat_label": "malware",
                        "severity": "high",
                        "confidence": "high",
                        "is_cyber_relevant": True,
                        "ioc_present": True,
                        "entities": [
                            {"type": "malware", "value": "Emotet", "source": "ner"},
                            {"type": "cve", "value": "CVE-2025-0001", "source": "regex"},
                        ],
                    },
                },
                {
                    "source_file": "group_b.json",
                    "date": "2025-01-02",
                    "message": "CVE-2025-0001 exploited by Emotet again",
                    "id": "m2",
                    "classification": {
                        "cleaned_text": "cve 2025 0001 exploited by emotet again",
                        "threat_label": "vulnerability_or_exploit",
                        "severity": "medium",
                        "confidence": "medium",
                        "is_cyber_relevant": True,
                        "ioc_present": True,
                        "entities": [
                            {"type": "cve", "value": "CVE-2025-0001", "source": "regex"},
                            {"type": "malware", "value": "emotet", "source": "ner"},
                        ],
                    },
                },
            ]
            write_json(classified_dir / "group_a.json", [rows[0]])
            write_json(classified_dir / "group_b.json", [rows[1]])

            matched = tmp_path / "matched.csv"
            with matched.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["post_id", "gold_threat_label"])
                writer.writeheader()
                writer.writerow({"post_id": "m1", "gold_threat_label": "malware"})

            hack = tmp_path / "hack.csv"
            with hack.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["date_reported", "attack_norm"])
                writer.writeheader()
                writer.writerow({"date_reported": "2025-01-01", "attack_norm": "malware"})

            args = argparse.Namespace(
                classified_dir=classified_dir,
                hackmageddon_csv=hack,
                matched_records=matched,
                output=tmp_path / "out.pt",
                metadata_output=None,
                embedding_dim=16,
                max_features=100,
                message_top_k=1,
                similarity_threshold=0.0,
                entity_half_life_days=30.0,
                random_seed=42,
                min_messages=2,
            )
            data, metadata = build_graph(args)

            self.assertEqual(set(data.node_types), {"message", "entity", "group", "day"})
            self.assertEqual(data["message"].num_nodes, 2)
            self.assertEqual(data["entity"].num_nodes, 2)
            self.assertEqual(data["message"].x.size(1), 23)
            self.assertEqual(int(data["message"].gold_mask.sum()), 1)
            self.assertEqual(data[("message", "mentions", "entity")].edge_index.size(1), 4)
            self.assertEqual(data[("entity", "co_occurs", "entity")].edge_index.size(1), 2)
            self.assertEqual(metadata["num_gold_messages"], 1)


if __name__ == "__main__":
    unittest.main()
