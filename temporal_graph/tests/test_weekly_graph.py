from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from build_weekly_graph import build_graph
from train_weekly import balanced_chronological_masks, stratified_masks


def write_json(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows), encoding="utf-8")


class WeeklyGraphTest(unittest.TestCase):
    def test_build_weekly_graph_marks_uncovered_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            classified_dir = tmp_path / "classified"
            classified_dir.mkdir()
            rows = [
                {
                    "source_file": "group.json",
                    "date": "2025-01-01",
                    "message": "first cve note",
                    "id": "m1",
                    "classification": {"entities": [{"type": "cve", "value": "CVE-2025-0001", "source": "regex"}]},
                    "cleaning": {"normalized_message": "first cve note"},
                },
                {
                    "source_file": "group.json",
                    "date": "2025-01-02",
                    "message": "second malware note",
                    "id": "m2",
                    "classification": {"entities": [{"type": "malware", "value": "Emotet", "source": "ner"}]},
                    "cleaning": {"normalized_message": "second malware note"},
                },
                {
                    "source_file": "group.json",
                    "date": "2025-02-01",
                    "message": "later benign note",
                    "id": "m3",
                    "classification": {"entities": []},
                    "cleaning": {"normalized_message": "later benign note"},
                },
            ]
            write_json(classified_dir / "group.json", rows)

            hack = tmp_path / "hack.csv"
            with hack.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "row_hash",
                        "date_reported",
                        "author",
                        "target",
                        "description",
                        "attack_norm",
                        "attack_class",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "row_hash": "h1",
                        "date_reported": "2025-01-02",
                        "author": "?",
                        "target": "Org",
                    "description": "Incident",
                    "attack_norm": "malware",
                    "attack_class": "CC",
                    }
                )
                writer.writerow(
                    {
                        "row_hash": "h2",
                        "date_reported": "2025-01-05",
                        "author": "?",
                        "target": "Org",
                    "description": "Incident",
                    "attack_norm": "ddos",
                    "attack_class": "H",
                    }
                )

            args = argparse.Namespace(
                deterministic_dir=classified_dir,
                hackmageddon_csv=hack,
                output=tmp_path / "out.pt",
                metadata_output=None,
                lookback_days=7,
                min_messages=1,
                embedding_backend="tfidf_svd",
                qwen_model="Qwen/Qwen3-Embedding-0.6B",
                embedding_cache=tmp_path / "cache",
                embedding_dim=8,
                max_features=100,
                embedding_batch_size=8,
                message_top_k=1,
                similarity_threshold=0.0,
                entity_half_life_days=30.0,
                include_hackmageddon_context=False,
                coverage_month_min_positive_days=2,
                drop_uncovered_targets=False,
                no_absolute_time=True,
                random_seed=42,
            )

            data, metadata = build_graph(args)

            self.assertEqual(data["day"].target_date, ["2025-01-02", "2025-01-03", "2025-02-02"])
            self.assertEqual(data["day"].coverage_status, ["observed", "observed", "uncovered_month"])
            self.assertEqual(data["day"].y_future_binary.tolist(), [1.0, 0.0, 0.0])
            self.assertEqual(data["day"].y_future_count.tolist(), [1.0, 0.0, 0.0])
            self.assertEqual(data["day"].y_future_count_bucket.tolist(), [1, 0, 0])
            malware_id = data.threat_type_to_id["malware"]
            self.assertEqual(data["day"].y_future_type_multilabel[:, malware_id].tolist(), [1.0, 0.0, 0.0])
            self.assertEqual(data["day"].y_future_type_counts[:, malware_id].tolist(), [1.0, 0.0, 0.0])
            self.assertEqual(data["day"].is_label_observed.tolist(), [True, True, False])
            self.assertEqual(data["day"].train_mask.tolist(), [True, True, False])
            self.assertEqual(data["message"].x.size(1), 10)
            self.assertEqual(data["day"].x.size(1), 16)
            self.assertEqual(metadata["num_label_observed_snapshots"], 2)
            self.assertEqual(
                metadata["count_bucket_observed_counts"],
                {"0": 1, "1-2": 1, "3-5": 0, "6-10": 0, "11-15": 0, "16+": 0},
            )
            self.assertEqual(metadata["threat_type_event_counts"]["malware"], 1)
            self.assertEqual(metadata["parameters"]["include_absolute_time"], False)

    def test_balanced_chronological_masks_require_both_classes(self) -> None:
        labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32)
        eligible = torch.ones(8, dtype=torch.bool)

        train, val, test = balanced_chronological_masks(labels, eligible, val_size=0.25, test_size=0.25)

        for mask in (train, val, test):
            values = labels[mask]
            self.assertTrue(bool((values == 0).any()))
            self.assertTrue(bool((values == 1).any()))
        self.assertTrue(torch.equal(train | val | test, eligible))
        self.assertFalse(bool((train & val).any()))
        self.assertFalse(bool((train & test).any()))
        self.assertFalse(bool((val & test).any()))

    def test_stratified_masks_preserve_bucket_distribution(self) -> None:
        labels = torch.tensor([0] * 12 + [1] * 12 + [2] * 12, dtype=torch.long)
        eligible = torch.ones(labels.numel(), dtype=torch.bool)

        train, val, test = stratified_masks(labels, eligible, val_size=0.25, test_size=0.25, seed=7)
        train_again, val_again, test_again = stratified_masks(
            labels, eligible, val_size=0.25, test_size=0.25, seed=7
        )

        self.assertTrue(torch.equal(train, train_again))
        self.assertTrue(torch.equal(val, val_again))
        self.assertTrue(torch.equal(test, test_again))
        self.assertEqual(int(train.sum().item()), 18)
        self.assertEqual(int(val.sum().item()), 9)
        self.assertEqual(int(test.sum().item()), 9)
        for mask in (train, val, test):
            values = labels[mask]
            self.assertEqual(int((values == 0).sum().item()), int((values == 1).sum().item()))
            self.assertEqual(int((values == 1).sum().item()), int((values == 2).sum().item()))
        self.assertTrue(torch.equal(train | val | test, eligible))
        self.assertFalse(bool((train & val).any()))
        self.assertFalse(bool((train & test).any()))
        self.assertFalse(bool((val & test).any()))


if __name__ == "__main__":
    unittest.main()
