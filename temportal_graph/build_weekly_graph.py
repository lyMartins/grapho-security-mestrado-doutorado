#!/usr/bin/env python3
"""Build a heterogeneous graph for weekly threat forecasting (7-day lookback → next-day label)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from weekly.builder import build_graph

_ROOT = Path(__file__).parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deterministic-dir", type=Path, default=_ROOT / "sentinel_replica_jsons_deterministic")
    parser.add_argument("--hackmageddon-csv", type=Path, default=_ROOT / "hackmargeddon_data/hackmageddon_normalized.csv")
    parser.add_argument("--output", type=Path, default=_ROOT / "temportal_graph/output/weekly_dataset.pt")
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--min-messages", type=int, default=3)
    parser.add_argument("--embedding-backend", choices=["qwen", "tfidf_svd"], default="qwen")
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-cache", type=Path, default=_ROOT / "graph/cache/qwen3_embedding_0_6b")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Only used by tfidf_svd.")
    parser.add_argument("--max-features", type=int, default=20000, help="Only used by tfidf_svd.")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--message-top-k", type=int, default=8)
    parser.add_argument("--similarity-threshold", type=float, default=0.70)
    parser.add_argument("--entity-half-life-days", type=float, default=30.0)
    parser.add_argument("--include-hackmageddon-context", action="store_true")
    parser.add_argument(
        "--coverage-month-min-positive-days",
        type=int,
        default=10,
        help="Treat a Hackmageddon month as label-observed only if it has at least this many positive days.",
    )
    parser.add_argument(
        "--drop-uncovered-targets",
        action="store_true",
        help="Drop snapshots whose D+1 target date is outside reliable Hackmageddon coverage.",
    )
    parser.add_argument(
        "--no-absolute-time",
        action="store_true",
        default=True,
        help="Remove globally normalized time coordinates from message/day features.",
    )
    parser.add_argument(
        "--include-absolute-time",
        action="store_false",
        dest="no_absolute_time",
        help="Restore globally normalized time coordinates for ablation/backward compatibility.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--exclude-groups",
        nargs="*",
        default=["bellingcat", "WokeIntelDrops", "itsectalk", "itsecalert"],
        help="Group names (source file stems) to exclude from the graph.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, metadata = build_graph(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.output)
    metadata_output = args.metadata_output or args.output.with_suffix("").parent / "weekly_metadata.json"
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved graph: {args.output}")
    print(f"saved metadata: {metadata_output}")
    print(json.dumps(metadata["node_types"], ensure_ascii=False))
    print(json.dumps(metadata["edge_types"], ensure_ascii=False))
    print(f"positive snapshots: {metadata['num_positive_snapshots']}/{metadata['num_snapshots']} ({metadata['positive_ratio']:.1%})")
    print(f"attack class distribution: {metadata['attack_class_positive_counts']}")


if __name__ == "__main__":
    main()
