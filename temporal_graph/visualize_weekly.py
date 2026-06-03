#!/usr/bin/env python3
"""Visualize 8 consecutive days of the weekly temporal graph using pyvis.

Layout (top → bottom):
  Groups → Days (sequential) → Messages → Entities → Hackmageddon labels
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from pyvis.network import Network

ROOT = Path(__file__).parent.parent
ATTACK_CLASSES = ["Cyber Crime", "Cyber Espionage", "Hacktivism", "Cyber Warfare", "Unknown"]

# Layer y-coordinates (positive = down in pyvis canvas)
Y_GROUP = -600
Y_DAY = -300
Y_MESSAGE = 0
Y_ENTITY = 350
Y_HACK = 700

CANVAS_W = 2000
PAD = 200


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=ROOT / "temporal_graph/output/weekly_dataset.pt")
    p.add_argument("--start-day", type=int, default=0, help="Index of first day in the 8-day window")
    p.add_argument("--output", type=Path, default=ROOT / "temporal_graph/output/graph_viz.html")
    p.add_argument("--max-messages", type=int, default=80)
    p.add_argument("--max-entities", type=int, default=50)
    return p.parse_args()


def _spread_x(items: list, canvas_w: int = CANVAS_W, pad: int = PAD) -> dict:
    """Map a list of items to evenly spread x-positions."""
    n = len(items)
    if n == 0:
        return {}
    if n == 1:
        return {items[0]: canvas_w // 2}
    span = canvas_w - 2 * pad
    return {item: pad + round(i / (n - 1) * span) for i, item in enumerate(items)}


def _date_frac(dt_str: str, date_min: str, date_max: str) -> float:
    from datetime import date as Date

    def p(s: str) -> Date:
        y, m, d = s.split("-")
        return Date(int(y), int(m), int(d))

    lo, hi, cur = p(date_min), p(date_max), p(dt_str)
    span = max((hi - lo).days, 1)
    return max(0.0, min(1.0, (cur - lo).days / span))


def main() -> None:
    args = parse_args()
    data = torch.load(args.dataset, weights_only=False)

    # 8-day window
    n_days = min(8, len(data["day"].date) - args.start_day)
    day_indices = list(range(args.start_day, args.start_day + n_days))
    day_set = set(day_indices)

    # ── Collect edges ──────────────────────────────────────────────────────────
    day_msg_ei = data[("day", "has_observed_message", "message")].edge_index
    msg_group_ei = data[("message", "posted_in", "group")].edge_index
    msg_entity_ei = data[("message", "mentions", "entity")].edge_index
    msg_sim_ei = data[("message", "similar_to", "message")].edge_index

    day_to_msgs: dict[int, list[int]] = defaultdict(list)
    for i in range(day_msg_ei.shape[1]):
        d, m = day_msg_ei[0, i].item(), day_msg_ei[1, i].item()
        if d in day_set:
            day_to_msgs[d].append(m)

    msg_set: set[int] = {m for msgs in day_to_msgs.values() for m in msgs}

    # Limit messages: keep those connected to the most days first
    if len(msg_set) > args.max_messages:
        freq: dict[int, int] = defaultdict(int)
        for msgs in day_to_msgs.values():
            for m in msgs:
                freq[m] += 1
        msg_set = set(sorted(msg_set, key=lambda m: -freq[m])[: args.max_messages])

    # Entities
    msg_to_entities: dict[int, list[int]] = defaultdict(list)
    entity_freq: dict[int, int] = defaultdict(int)
    for i in range(msg_entity_ei.shape[1]):
        m, e = msg_entity_ei[0, i].item(), msg_entity_ei[1, i].item()
        if m in msg_set:
            msg_to_entities[m].append(e)
            entity_freq[e] += 1

    entity_set: set[int] = {e for ents in msg_to_entities.values() for e in ents}
    if len(entity_set) > args.max_entities:
        entity_set = set(sorted(entity_set, key=lambda e: -entity_freq[e])[: args.max_entities])

    # Groups
    msg_to_group: dict[int, int] = {}
    for i in range(msg_group_ei.shape[1]):
        m, g = msg_group_ei[0, i].item(), msg_group_ei[1, i].item()
        if m in msg_set:
            msg_to_group[m] = g
    group_set = set(msg_to_group.values())

    # ── X positions ────────────────────────────────────────────────────────────
    usable = CANVAS_W - 2 * PAD
    day_x = {d: PAD + round(i / max(n_days - 1, 1) * usable) for i, d in enumerate(day_indices)}

    date_min = data["day"].date[day_indices[0]]
    date_max = data["day"].date[day_indices[-1]]

    def msg_x(m_idx: int) -> int:
        frac = _date_frac(data["message"].date[m_idx], date_min, date_max)
        return PAD + round(frac * usable)

    # Jitter messages that share the same date so they don't overlap
    date_buckets: dict[str, list[int]] = defaultdict(list)
    for m in sorted(msg_set):
        date_buckets[data["message"].date[m]].append(m)

    msg_pos: dict[int, tuple[int, int]] = {}
    for dt, bucket in date_buckets.items():
        base_x = PAD + round(_date_frac(dt, date_min, date_max) * usable)
        half = (len(bucket) - 1) / 2
        for j, m in enumerate(bucket):
            msg_pos[m] = (base_x + round((j - half) * 22), Y_MESSAGE)

    group_x = _spread_x(sorted(group_set))
    entity_x = _spread_x(sorted(entity_set))
    hack_x = _spread_x(list(range(len(ATTACK_CLASSES))))

    # ── Build Network ──────────────────────────────────────────────────────────
    net = Network(height="960px", width="100%", bgcolor="#0f172a", font_color="white")
    net.set_options("""
    {
      "physics": { "enabled": false },
      "interaction": { "hover": true, "navigationButtons": true },
      "edges": { "smooth": { "enabled": false } }
    }
    """)

    # Groups
    for g in sorted(group_set):
        name = data["group"].name[g]
        net.add_node(
            f"g_{g}", label=name,
            x=group_x[g], y=Y_GROUP,
            color={"background": "#3b82f6", "border": "#93c5fd"},
            size=22, shape="dot",
            title=f"Group: {name}",
            font={"size": 13, "color": "#e2e8f0"},
        )

    # Days
    for d in day_indices:
        dt = data["day"].date[d]
        is_pos = data["day"].y_future_binary[d].item() > 0.5
        count = int(data["day"].y_future_count[d].item())
        classes = [ATTACK_CLASSES[ci] for ci, v in enumerate(data["day"].y_future_multilabel[d].tolist()) if v > 0.5]
        color = {"background": "#10b981", "border": "#6ee7b7"} if is_pos else {"background": "#475569", "border": "#94a3b8"}
        title = f"Day: {dt}\nTarget next day: {'YES' if is_pos else 'no'} ({count} events)\n" + (", ".join(classes) or "—")
        net.add_node(
            f"d_{d}", label=dt[5:],
            x=day_x[d], y=Y_DAY,
            color=color, size=18, shape="box",
            title=title,
            font={"size": 12, "color": "#f1f5f9"},
        )

    # Day → Day sequential edges
    for i in range(len(day_indices) - 1):
        net.add_edge(
            f"d_{day_indices[i]}", f"d_{day_indices[i + 1]}",
            color="#64748b", width=2, arrows="to",
        )

    # Messages
    for m in sorted(msg_set):
        x, y = msg_pos[m]
        dt = data["message"].date[m]
        pid = data["message"].post_id[m]
        net.add_node(
            f"m_{m}", label="",
            x=x, y=y,
            color={"background": "#fbbf24", "border": "#fde68a"},
            size=8, shape="dot",
            title=f"Post: {pid[:12]}…\nDate: {dt}",
        )

    # Day → Message edges
    for d, msgs in day_to_msgs.items():
        if d not in day_set:
            continue
        for m in msgs:
            if m in msg_set:
                net.add_edge(f"d_{d}", f"m_{m}", color="#334155", width=1)

    # Message → Message similarity edges
    seen_sim: set[tuple[int, int]] = set()
    for i in range(msg_sim_ei.shape[1]):
        src, dst = msg_sim_ei[0, i].item(), msg_sim_ei[1, i].item()
        if src in msg_set and dst in msg_set and src != dst:
            key = (min(src, dst), max(src, dst))
            if key not in seen_sim:
                seen_sim.add(key)
                net.add_edge(f"m_{src}", f"m_{dst}", color="#78350f", width=1, dashes=True)

    # Message → Group edges
    for m, g in msg_to_group.items():
        if m in msg_set and g in group_set:
            net.add_edge(f"m_{m}", f"g_{g}", color="#1e40af", width=1)

    # Entities
    for e in sorted(entity_set):
        etype = data["entity"].entity_type[e]
        val = data["entity"].value[e]
        label = val[:18] + "…" if len(val) > 18 else val
        net.add_node(
            f"e_{e}", label=label,
            x=entity_x[e], y=Y_ENTITY,
            color={"background": "#f97316", "border": "#fed7aa"},
            size=11, shape="dot",
            title=f"[{etype}] {val}",
            font={"size": 9, "color": "#fef3c7"},
        )

    # Message → Entity edges
    for m, entities in msg_to_entities.items():
        if m not in msg_set:
            continue
        for e in entities:
            if e in entity_set:
                net.add_edge(f"m_{m}", f"e_{e}", color="#7c2d12", width=1)

    # Hackmageddon label nodes
    for ci, cls in enumerate(ATTACK_CLASSES):
        active = sum(1 for d in day_indices if data["day"].y_future_multilabel[d][ci].item() > 0.5)
        color = (
            {"background": "#dc2626", "border": "#fca5a5"}
            if active > 0
            else {"background": "#1e293b", "border": "#475569"}
        )
        net.add_node(
            f"h_{ci}", label=cls,
            x=hack_x[ci], y=Y_HACK,
            color=color, size=18 + active * 4, shape="ellipse",
            title=f"Attack class: {cls}\nActive in {active}/{n_days} days",
            font={"size": 11, "color": "#fef2f2"},
        )

    # Day → Hackmageddon edges (active only)
    for d in day_indices:
        for ci in range(len(ATTACK_CLASSES)):
            if data["day"].y_future_multilabel[d][ci].item() > 0.5:
                net.add_edge(
                    f"d_{d}", f"h_{ci}",
                    color="#991b1b", width=2, dashes=True, arrows="to",
                )

    # ── Render ─────────────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)

    start_date = data["day"].date[day_indices[0]]
    end_date = data["day"].date[day_indices[-1]]
    subtitle = (
        f"Weekly graph slice: {start_date} .. {end_date} | "
        f"days={n_days}, messages={len(msg_set)}, entities={len(entity_set)}, groups={len(group_set)}"
    )

    layer_legend = (
        "<div style='position:absolute;top:14px;left:18px;color:#94a3b8;"
        "font-family:monospace;font-size:12px;line-height:1.8;z-index:999;"
        "background:#0f172acc;padding:8px 14px;border-radius:8px;'>"
        "▲ Groups<br>● Days (green=hack next day)<br>"
        "● Messages<br>● Entities<br>▼ Hackmageddon labels"
        "</div>"
    )
    title_div = (
        f"<div style='position:absolute;top:14px;left:50%;transform:translateX(-50%);"
        f"color:#f1f5f9;font-family:sans-serif;font-size:14px;z-index:999;"
        f"background:#0f172acc;padding:6px 18px;border-radius:8px;white-space:nowrap;'>"
        f"{subtitle}</div>"
    )

    html = net.generate_html()
    html = html.replace("<body>", f"<body>{title_div}{layer_legend}")
    args.output.write_text(html, encoding="utf-8")
    print(f"Saved: {args.output}")
    print(subtitle)


if __name__ == "__main__":
    main()
