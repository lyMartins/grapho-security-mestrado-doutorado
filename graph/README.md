# Option A Heterogeneous Graph

This folder implements the message-level graph described in `base.md`:

- node types: `message`, `entity`, `group`, `day`
- message labels: deterministic `threat_label`, optional Hackmageddon gold labels
- GNN path: PyTorch Geometric `HeteroData` + `HGTConv`

## Build

```bash
uv run python graph/build_graph.py \
  --deterministic-dir sentinel_replica_jsons_deterministic \
  --hackmageddon-csv hackmargeddon_data/hackmageddon_normalized.csv \
  --matched-records eval/output/matched_records.csv \
  --output graph/output/heterodata_option_a.pt
```

The build writes `graph/output/heterodata_option_a.pt` and `graph/output/metadata.json`.
`graph/output/` is ignored by Git because these files are generated artifacts.

## Train

```bash
uv run python graph/train_hgt.py \
  --data graph/output/heterodata_option_a.pt \
  --epochs 300 \
  --hidden-dim 128 \
  --num-layers 2
```

For a quick integration check:

```bash
uv run python graph/train_hgt.py \
  --data graph/output/heterodata_option_a.pt \
  --epochs 1 \
  --hidden-dim 32 \
  --num-layers 1 \
  --heads 2 \
  --metrics-output graph/output/train_metrics_smoke.json
```

By default, PNG plots are written to `graph/output/`:

- `graph/output/hgt_training_history.png`
- `graph/output/hgt_test_confusion_matrix.png`
- `graph/output/hgt_gold_confusion_matrix.png`

Use `--plots-dir` to choose another directory.

## Future 12h Forecast

This builds the intraday forecasting graph: each `day` node is a 12:00 UTC
snapshot, Telegram messages observed in the previous `--lookback-days` window
are the input signal, and Hackmageddon events whose reported date falls in the
next `--horizon-hours` target window are the gold label. Hackmageddon currently
has daily granularity, so a 12-hour horizon is evaluated against the reported
date touched by that future window.

The default embedding backend is `Qwen/Qwen3-Embedding-0.6B`; this is an
embedding encoder, not the Qwen LLM entity extractor. The builder uses
`sentinel_replica_jsons_deterministic` by default and does not use
`sentinel_replica_jsons_classified_eval_qwen3.5-9b`.

```bash
uv run python graph/build_future_12h_graph.py \
  --deterministic-dir sentinel_replica_jsons_deterministic \
  --telegram-html-dir ChatExport_2026-03-18/DB-telegram \
  --hackmageddon-csv hackmargeddon_data/hackmageddon_normalized.csv \
  --similarity-threshold 0.70 \
  --output graph/output/future_12h_dataset.pt
```

For a no-download smoke build, use `--embedding-backend tfidf_svd`.

The Telegram HTML export must cover the same messages/dates as the classified
JSONs. The current builder matches timestamps primarily by Telegram `_id`, with
a date+text fallback.

For a fresh Telegram collection with intraday timestamps:

```bash
uv run python coleta/coletar_telegram.py \
  --full-refresh \
  --start-date 2023-01-01 \
  --output-dir sentinel_replica_jsons
```

Newly collected records include `date`, `datetime_utc`, and Unix `timestamp`.
When `datetime_utc` is present, the 12h graph builder uses it directly and does
not require the Telegram HTML export.

```bash
uv run python graph/train_future_12h.py \
  --data graph/output/future_12h_dataset.pt \
  --epochs 100
```

By default Hackmageddon embeddings and counts are not included as input
features, avoiding label leakage. `--include-hackmageddon-context` enables the
retrospective ablation explicitly.

## Visualize

```bash
uv run python graph/plot_graph_html.py \
  --data graph/output/heterodata_option_a.pt \
  --output graph/output/option_a_graph.html \
  --max-messages 180
```

The PyVis HTML view uses a dynamic force layout and is sampled by default so the
browser remains usable. Gold-labeled messages are prioritized in the sample
unless `--no-gold-priority` is used.

For the future 12h forecasting graph:

```bash
uv run python graph/plot_graph_html.py \
  --data graph/output/future_12h_dataset.pt \
  --output graph/output/future_12h_graph.html \
  --max-messages 180
```

The visualizer detects this dataset automatically. Yellow-bordered day nodes
mean the snapshot has a Hackmageddon event in its future target window.

## Test

```bash
uv run python -m unittest graph.tests.test_build_graph
uv run python -m unittest graph.tests.test_future_12h_graph
```
