#!/usr/bin/env python3
"""Render an inspectable HTML view of the heterogeneous graph datasets."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import networkx as nx
import torch
from pyvis.network import Network


FORWARD_EDGE_TYPES = [
    ("message", "mentions", "entity"),
    ("message", "posted_in", "group"),
    ("message", "posted_on", "day"),
    ("message", "observed_before_snapshot", "day"),
    ("message", "similar_to", "message"),
    ("entity", "co_occurs", "entity"),
    ("day", "next", "day"),
]

NODE_COLORS = {
    "message": "#1f77b4",
    "entity": "#ff7f0e",
    "group": "#2ca02c",
    "day": "#9467bd",
}

EDGE_COLORS = {
    "mentions": "#ff7f0e",
    "posted_in": "#2ca02c",
    "posted_on": "#9467bd",
    "observed_before_snapshot": "#9467bd",
    "similar_to": "#8c564b",
    "co_occurs": "#d62728",
    "next": "#7f7f7f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("graph/output/heterodata_option_a.pt"))
    parser.add_argument("--output", type=Path, default=Path("graph/output/option_a_graph.html"))
    parser.add_argument("--max-messages", type=int, default=180)
    parser.add_argument("--max-entity-edges", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--date-start", default=None, help="Keep messages on/after YYYY-MM-DD.")
    parser.add_argument("--date-end", default=None, help="Keep messages on/before YYYY-MM-DD.")
    parser.add_argument("--labels", default=None, help="Comma-separated weak labels to include.")
    parser.add_argument("--no-gold-priority", action="store_true", help="Do not force gold messages into the sample.")
    parser.add_argument("--no-message-similarity", action="store_true", help="Hide message-message similarity edges.")
    parser.add_argument("--no-entity-cooccurrence", action="store_true", help="Hide entity-entity co-occurrence edges.")
    return parser.parse_args()


def tensor_list(value: torch.Tensor) -> list[Any]:
    return value.detach().cpu().tolist()


def node_id(node_type: str, index: int) -> str:
    return f"{node_type}:{index}"


def truncate(value: str, limit: int = 42) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "..."


def label_mapping(data: Any) -> dict[int, str]:
    if not hasattr(data, "id_to_label"):
        return {}
    return {int(index): label for index, label in dict(data.id_to_label).items()}


def has_field(data: Any, node_type: str, field: str) -> bool:
    return field in data[node_type]


def graph_task(data: Any) -> str:
    if has_field(data, "day", "y_future_binary"):
        return "future_12h"
    return "option_a"


def day_future_labels(data: Any, index: int, id_to_label: dict[int, str]) -> list[str]:
    if not has_field(data, "day", "y_future_multilabel"):
        return []
    row = data["day"].y_future_multilabel[index].detach().cpu()
    return [id_to_label.get(label_id, str(label_id)) for label_id, value in enumerate(row.tolist()) if value > 0]


def selected_message_indices(data: Any, args: argparse.Namespace) -> set[int]:
    rng = random.Random(args.seed)
    id_to_label = label_mapping(data)
    labels = set(args.labels.split(",")) if args.labels else None
    dates = list(data["message"].date)
    y_weak = tensor_list(data["message"].y_weak) if has_field(data, "message", "y_weak") else None
    gold_mask = tensor_list(data["message"].gold_mask) if has_field(data, "message", "gold_mask") else None

    candidates = []
    for index, date in enumerate(dates):
        if labels and y_weak is None:
            continue
        if labels and y_weak is not None and id_to_label.get(int(y_weak[index])) not in labels:
            continue
        if args.date_start and date < args.date_start:
            continue
        if args.date_end and date > args.date_end:
            continue
        candidates.append(index)

    selected: list[int] = []
    if gold_mask is not None and not args.no_gold_priority:
        selected.extend(index for index in candidates if bool(gold_mask[index]))

    remaining = [index for index in candidates if index not in set(selected)]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, args.max_messages - len(selected))])
    return set(selected[: args.max_messages])


def add_message_node(graph: nx.MultiDiGraph, data: Any, index: int, id_to_label: dict[int, str]) -> None:
    date = data["message"].date[index]
    post_id = data["message"].post_id[index]
    source = data["message"].source_file[index] if has_field(data, "message", "source_file") else ""
    datetime_utc = data["message"].datetime_utc[index] if has_field(data, "message", "datetime_utc") else ""
    threat_label = data["message"].threat_label[index] if has_field(data, "message", "threat_label") else ""
    is_gold = bool(data["message"].gold_mask[index].item()) if has_field(data, "message", "gold_mask") else False
    weak_label = ""
    gold_label = ""
    if has_field(data, "message", "y_weak"):
        y_weak = int(data["message"].y_weak[index].item())
        weak_label = id_to_label.get(y_weak, str(y_weak))
    if is_gold and has_field(data, "message", "y_gold"):
        y_gold = int(data["message"].y_gold[index].item())
        gold_label = id_to_label.get(y_gold, "")
    details = [
        f"message {index}",
        f"date: {date}",
        f"datetime_utc: {datetime_utc}" if datetime_utc else "",
        f"weak label: {weak_label}" if weak_label else "",
        f"gold label: {gold_label or '-'}" if is_gold else "",
        f"stored label: {threat_label}" if threat_label else "",
        f"source: {source}" if source else "",
        f"post_id: {post_id}",
    ]
    graph.add_node(
        node_id("message", index),
        type="message",
        label=truncate(f"msg {index}"),
        title="\n".join(line for line in details if line),
        gold=is_gold,
        weak_label=weak_label,
    )


def add_neighbor_node(graph: nx.MultiDiGraph, data: Any, node_type: str, index: int, id_to_label: dict[int, str]) -> None:
    node = node_id(node_type, index)
    if node in graph:
        return
    is_gold = False
    if node_type == "entity":
        entity_type = data["entity"].entity_type[index]
        value = data["entity"].value[index]
        label = truncate(f"{entity_type}:{value}", 36)
        title = f"entity {index}\ntype: {entity_type}\nvalue: {value}"
    elif node_type == "group":
        name = data["group"].name[index]
        label = truncate(name, 36)
        title = f"group {index}\nname: {name}"
    elif node_type == "day":
        date = data["day"].date[index]
        details = [f"day {index}", f"date: {date}"]
        if has_field(data, "day", "observed_message_count"):
            details.append(f"observed messages: {int(data['day'].observed_message_count[index].item())}")
        if has_field(data, "day", "target_dates"):
            details.append(f"target dates: {', '.join(data['day'].target_dates[index])}")
        if has_field(data, "day", "y_future_binary"):
            is_gold = bool(data["day"].y_future_binary[index].item())
            details.append(f"future Hackmageddon event: {is_gold}")
            labels = day_future_labels(data, index, id_to_label)
            details.append(f"future labels: {', '.join(labels) if labels else '-'}")
        else:
            features = tensor_list(data["day"].x[index])
            if len(features) >= 4:
                details.extend(
                    [
                        f"log messages: {features[0]:.3f}",
                        f"log Hackmageddon events: {features[1]:.3f}",
                        f"log Hackmageddon threat events: {features[2]:.3f}",
                        f"has Hackmageddon threat: {bool(features[3])}",
                    ]
                )
        label = date
        title = "\n".join(details)
    else:
        label = f"{node_type} {index}"
        title = label
    graph.add_node(node, type=node_type, label=label, title=title, gold=is_gold)


def edge_rows(data: Any, edge_type: tuple[str, str, str]) -> list[tuple[int, int, float]]:
    if edge_type not in data.edge_types:
        return []
    store = data[edge_type]
    edge_index = tensor_list(store.edge_index)
    weights = tensor_list(store.edge_weight) if hasattr(store, "edge_weight") else None
    rows = []
    for pos, (src, dst) in enumerate(zip(edge_index[0], edge_index[1], strict=True)):
        weight = float(weights[pos]) if weights is not None else 1.0
        rows.append((int(src), int(dst), weight))
    return rows


def build_networkx_graph(data: Any, args: argparse.Namespace) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    id_to_label = label_mapping(data)
    selected_messages = selected_message_indices(data, args)
    for index in sorted(selected_messages):
        add_message_node(graph, data, index, id_to_label)

    visible_entities: set[int] = set()
    visible_days: set[int] = set()

    for src_type, relation, dst_type in FORWARD_EDGE_TYPES:
        if relation == "similar_to" and args.no_message_similarity:
            continue
        if relation == "co_occurs" and args.no_entity_cooccurrence:
            continue

        rows = edge_rows(data, (src_type, relation, dst_type))
        if relation == "co_occurs":
            rows = [
                row
                for row in rows
                if row[0] in visible_entities and row[1] in visible_entities and row[0] < row[1]
            ]
            rows = sorted(rows, key=lambda row: row[2], reverse=True)[: args.max_entity_edges]
        elif relation == "next":
            rows = [row for row in rows if row[0] in visible_days and row[1] in visible_days]

        for src, dst, weight in rows:
            if src_type == "message" and src not in selected_messages:
                continue
            if dst_type == "message" and dst not in selected_messages:
                continue

            if src_type != "message":
                add_neighbor_node(graph, data, src_type, src, id_to_label)
            if dst_type != "message":
                add_neighbor_node(graph, data, dst_type, dst, id_to_label)

            if src_type == "entity":
                visible_entities.add(src)
            if dst_type == "entity":
                visible_entities.add(dst)
            if src_type == "day":
                visible_days.add(src)
            if dst_type == "day":
                visible_days.add(dst)

            graph.add_edge(
                node_id(src_type, src),
                node_id(dst_type, dst),
                relation=relation,
                weight=weight,
                title=f"{relation} ({weight:.3f})",
            )

    return graph


def graph_summary(graph: nx.MultiDiGraph) -> dict[str, Any]:
    node_counts: dict[str, int] = {}
    edge_counts: dict[str, int] = {}
    for _, attrs in graph.nodes(data=True):
        node_type = attrs["type"]
        node_counts[node_type] = node_counts.get(node_type, 0) + 1
    for _, _, attrs in graph.edges(data=True):
        relation = attrs["relation"]
        edge_counts[relation] = edge_counts.get(relation, 0) + 1
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "node_counts": dict(sorted(node_counts.items())),
        "edge_counts": dict(sorted(edge_counts.items())),
    }


def render_html(graph: nx.MultiDiGraph, output: Path, seed: int, task: str) -> None:
    net = Network(
        height="100vh",
        width="100%",
        bgcolor="#ffffff",
        font_color="#222222",
        directed=True,
        cdn_resources="in_line",
    )

    for node, attrs in graph.nodes(data=True):
        node_type = attrs["type"]
        color = {
            "background": NODE_COLORS.get(node_type, "#999999"),
            "border": "#f1c40f" if attrs.get("gold") else "#ffffff",
            "highlight": {
                "background": NODE_COLORS.get(node_type, "#999999"),
                "border": "#111827",
            },
        }
        size = 18 if node_type == "day" else 16
        if node_type == "message":
            size = 14
        label = attrs["label"] if node_type in {"group", "day"} or attrs.get("gold") else ""
        net.add_node(
            node,
            label=label,
            title=str(attrs["title"]).replace("\n", "<br>"),
            group=node_type,
            color=color,
            size=size,
            borderWidth=3 if attrs.get("gold") else 1,
        )

    for src, dst, attrs in graph.edges(data=True):
        relation = attrs["relation"]
        weight = float(attrs["weight"])
        net.add_edge(
            src,
            dst,
            title=str(attrs["title"]),
            color=EDGE_COLORS.get(relation, "#999999"),
            width=max(0.5, min(4.0, math.log1p(weight))),
            arrows="to",
            physics=relation not in {"next"},
        )

    # Barnes-Hut keeps the graph dynamic while avoiding excessive overlap.
    net.set_options(
        json.dumps(
            {
                "configure": {"enabled": True, "filter": ["physics"]},
                "interaction": {
                    "hover": True,
                    "navigationButtons": True,
                    "keyboard": True,
                    "tooltipDelay": 80,
                },
                "physics": {
                    "enabled": True,
                    "solver": "barnesHut",
                    "barnesHut": {
                        "gravitationalConstant": -18000,
                        "centralGravity": 0.18,
                        "springLength": 120,
                        "springConstant": 0.035,
                        "damping": 0.18,
                        "avoidOverlap": 0.45,
                    },
                    "stabilization": {
                        "enabled": True,
                        "iterations": 700,
                        "updateInterval": 25,
                        "fit": True,
                    },
                },
                "edges": {
                    "smooth": {"enabled": True, "type": "dynamic"},
                    "color": {"inherit": False},
                },
                "nodes": {
                    "shape": "dot",
                    "font": {"size": 12, "face": "arial"},
                },
                "groups": {
                    node_type: {"color": color}
                    for node_type, color in NODE_COLORS.items()
                },
                "layout": {"randomSeed": seed},
            }
        )
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(output), notebook=False, open_browser=False)

    summary = graph_summary(graph)
    legend = "".join(
        f'<span><i style="background:{color}"></i>{node_type}</span>'
        for node_type, color in NODE_COLORS.items()
    )
    summary_html = json.dumps(summary, indent=2, ensure_ascii=False)
    title = "Future 12h Graph" if task == "future_12h" else "Option A Graph"
    note = (
        "Nós de dia com borda amarela possuem evento Hackmageddon na janela futura."
        if task == "future_12h"
        else "Nós de mensagem com borda amarela possuem label gold."
    )
    panel = f"""
<div id="graph-summary">
  <h3>{title}</h3>
  <pre>{summary_html}</pre>
  <div class="legend">{legend}</div>
  <p>{note} Arraste os nós para ajustar a física; use o painel inferior para afinar parâmetros.</p>
</div>
<style>
  #graph-summary {{
    position: absolute;
    top: 12px;
    left: 12px;
    width: 290px;
    max-height: calc(100vh - 24px);
    overflow: auto;
    z-index: 10;
    padding: 12px;
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid #d0d7de;
    border-radius: 8px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
    font-family: Arial, sans-serif;
    font-size: 12px;
  }}
  #graph-summary h3 {{ margin: 0 0 8px; }}
  #graph-summary pre {{
    white-space: pre-wrap;
    background: #f1f5f9;
    padding: 8px;
    border-radius: 6px;
  }}
  #graph-summary .legend span {{ display: block; margin: 6px 0; }}
  #graph-summary .legend i {{
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
    vertical-align: -1px;
  }}
</style>
"""
    document = output.read_text(encoding="utf-8")
    output.write_text(document.replace("<body>", f"<body>{panel}", 1), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data = torch.load(args.data, weights_only=False, map_location="cpu")
    graph = build_networkx_graph(data, args)
    render_html(graph, args.output, args.seed, graph_task(data))
    print(f"saved graph html: {args.output}")
    print(json.dumps(graph_summary(graph), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
