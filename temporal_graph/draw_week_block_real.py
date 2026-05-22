#!/usr/bin/env python3
"""Draw a real 7-day + 1-day temporal graph slice without sampling edges."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from textwrap import shorten

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import torch


ROOT = Path(__file__).parent.parent
ATTACK_CLASSES = ["Cyber Crime", "Cyber Espionage", "Hacktivism", "Cyber Warfare", "Unknown"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "temporal_graph/output/weekly_dataset.pt")
    parser.add_argument(
        "--start-day",
        type=int,
        default=508,
        help="First observed day. Default is the sparsest real 7-day window found in the dataset.",
    )
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "temporal_graph/output/weekly_block_real")
    return parser.parse_args()


def add_edge(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    lw: float,
    alpha: float,
    arrow: bool = False,
    linestyle: str = "-",
    rad: float = 0.0,
    zorder: int = 1,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>" if arrow else "-",
            mutation_scale=8,
            linewidth=lw,
            color=color,
            alpha=alpha,
            linestyle=linestyle,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
            zorder=zorder,
        )
    )


def scatter(ax, points: list[tuple[float, float]], size: float, color: str, edge: str, marker: str = "o") -> None:
    if not points:
        return
    xs, ys = zip(*points)
    ax.scatter(xs, ys, s=size, c=color, edgecolors=edge, linewidths=0.45, marker=marker, zorder=3)


def grid_positions(items: list[int], x_min: float, x_max: float, y_min: float, y_max: float) -> dict[int, tuple[float, float]]:
    if not items:
        return {}
    n = len(items)
    cols = max(1, int(n**0.5 * 1.8))
    rows = (n + cols - 1) // cols
    positions: dict[int, tuple[float, float]] = {}
    for idx, item in enumerate(items):
        row, col = divmod(idx, cols)
        x = (x_min + x_max) / 2 if cols == 1 else x_min + col * ((x_max - x_min) / (cols - 1))
        y = (y_min + y_max) / 2 if rows == 1 else y_max - row * ((y_max - y_min) / (rows - 1))
        positions[item] = (x, y)
    return positions


def main() -> None:
    args = parse_args()
    data = torch.load(args.dataset, weights_only=False)

    start_day = min(max(args.start_day, 0), len(data["day"].date) - 8)
    observed_days = list(range(start_day, start_day + 7))
    target_day = start_day + 7
    observed_set = set(observed_days)

    day_msg_ei = data[("day", "has_observed_message", "message")].edge_index
    msg_group_ei = data[("message", "posted_in", "group")].edge_index
    msg_entity_ei = data[("message", "mentions", "entity")].edge_index
    msg_sim_ei = data[("message", "similar_to", "message")].edge_index

    day_to_msgs: dict[int, list[int]] = defaultdict(list)
    for i in range(day_msg_ei.shape[1]):
        day_idx = int(day_msg_ei[0, i])
        msg_idx = int(day_msg_ei[1, i])
        if day_idx in observed_set:
            day_to_msgs[day_idx].append(msg_idx)

    msg_set = sorted({msg for msgs in day_to_msgs.values() for msg in msgs})

    msg_to_group: dict[int, int] = {}
    group_set: set[int] = set()
    for i in range(msg_group_ei.shape[1]):
        msg_idx = int(msg_group_ei[0, i])
        group_idx = int(msg_group_ei[1, i])
        if msg_idx in msg_set:
            msg_to_group[msg_idx] = group_idx
            group_set.add(group_idx)

    msg_to_entities: dict[int, list[int]] = defaultdict(list)
    entity_set: set[int] = set()
    for i in range(msg_entity_ei.shape[1]):
        msg_idx = int(msg_entity_ei[0, i])
        entity_idx = int(msg_entity_ei[1, i])
        if msg_idx in msg_set:
            msg_to_entities[msg_idx].append(entity_idx)
            entity_set.add(entity_idx)

    sim_edges: set[tuple[int, int]] = set()
    msg_lookup = set(msg_set)
    for i in range(msg_sim_ei.shape[1]):
        src = int(msg_sim_ei[0, i])
        dst = int(msg_sim_ei[1, i])
        if src in msg_lookup and dst in msg_lookup and src != dst:
            sim_edges.add((min(src, dst), max(src, dst)))

    day_x = {day_idx: float(rank) for rank, day_idx in enumerate(observed_days + [target_day])}
    group_pos = grid_positions(sorted(group_set), 0.15, 6.85, 3.06, 3.38)
    entity_pos = grid_positions(sorted(entity_set), 0.05, 6.95, -1.1, -0.28)

    msg_pos: dict[int, tuple[float, float]] = {}
    for day_idx in observed_days:
        msgs = sorted(set(day_to_msgs[day_idx]))
        msg_pos.update(grid_positions(msgs, day_x[day_idx] - 0.42, day_x[day_idx] + 0.42, 0.28, 1.32))

    fig, ax = plt.subplots(figsize=(18, 10.5), dpi=180)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.set_xlim(-0.85, 9.15)
    ax.set_ylim(-2.35, 3.8)
    ax.axis("off")

    title = "Faixa real do grafo temporal: 7 dias observados + alvo D+1"
    subtitle = (
        f"{data['day'].date[observed_days[0]]} a {data['day'].date[observed_days[-1]]} -> "
        f"alvo {data['day'].date[target_day]} | "
        f"mensagens={len(msg_set)}, entidades={len(entity_set)}, grupos={len(group_set)}, "
        f"arestas={sum(len(v) for v in day_to_msgs.values()) + len(msg_to_group) + sum(len(v) for v in msg_to_entities.values()) + len(sim_edges)}"
    )
    ax.text(4.15, 3.68, title, ha="center", va="center", color="#f8fafc", fontsize=18, fontweight="bold")
    ax.text(4.15, 3.5, subtitle, ha="center", va="center", color="#cbd5e1", fontsize=9)

    layers = [
        (3.22, "Grupos"),
        (2.24, "Dias"),
        (0.8, "Mensagens reais"),
        (-0.7, "Entidades reais"),
    ]
    for y, label in layers:
        ax.text(-0.78, y, label, ha="left", va="center", color="#94a3b8", fontsize=9, fontweight="bold")
        ax.hlines(y, -0.25, 8.95, color="#1e293b", linewidth=0.8, zorder=0)

    day_pos = {day_idx: (day_x[day_idx], 2.24) for day_idx in observed_days}
    target_pos = (day_x[target_day], 2.24)

    for left, right in zip(observed_days + [target_day], observed_days[1:] + [target_day]):
        if left in day_pos and right in day_pos:
            add_edge(ax, day_pos[left], day_pos[right], "#64748b", lw=1.8, alpha=0.62, arrow=True, zorder=0)
    add_edge(ax, day_pos[observed_days[-1]], target_pos, "#94a3b8", lw=2.2, alpha=0.75, arrow=True, zorder=1)

    for day_idx, msgs in day_to_msgs.items():
        for msg_idx in msgs:
            if msg_idx in msg_pos:
                add_edge(ax, day_pos[day_idx], msg_pos[msg_idx], "#64748b", lw=0.22, alpha=0.12, zorder=1)

    for msg_idx, group_idx in msg_to_group.items():
        if msg_idx in msg_pos and group_idx in group_pos:
            add_edge(ax, group_pos[group_idx], msg_pos[msg_idx], "#2563eb", lw=0.34, alpha=0.2, zorder=1)

    for msg_idx, entity_ids in msg_to_entities.items():
        for entity_idx in entity_ids:
            if msg_idx in msg_pos and entity_idx in entity_pos:
                add_edge(ax, msg_pos[msg_idx], entity_pos[entity_idx], "#ea580c", lw=0.3, alpha=0.18, zorder=1)

    for src, dst in sim_edges:
        add_edge(ax, msg_pos[src], msg_pos[dst], "#f59e0b", lw=0.28, alpha=0.22, linestyle="--", rad=0.12, zorder=2)

    scatter(ax, list(group_pos.values()), 190, "#2563eb", "#93c5fd")
    for group_idx, xy in group_pos.items():
        label = shorten(str(data["group"].name[group_idx]), width=18, placeholder="...")
        ax.text(xy[0], xy[1] - 0.17, label, ha="center", va="top", color="#e5e7eb", fontsize=6.2)

    for day_idx in observed_days:
        ax.scatter([day_pos[day_idx][0]], [day_pos[day_idx][1]], s=520, c="#334155", edgecolors="#94a3b8", linewidths=1.2, marker="s", zorder=4)
        ax.text(day_pos[day_idx][0], day_pos[day_idx][1] - 0.25, data["day"].date[day_idx][5:], ha="center", va="top", color="#e5e7eb", fontsize=8, fontweight="bold")

    ax.scatter([target_pos[0]], [target_pos[1]], s=650, c="#16a34a", edgecolors="#86efac", linewidths=1.5, marker="s", zorder=4)
    ax.text(target_pos[0], target_pos[1] - 0.25, data["day"].date[target_day][5:], ha="center", va="top", color="#e5e7eb", fontsize=8, fontweight="bold")
    ax.text(target_pos[0], target_pos[1] + 0.36, "alvo D+1", ha="center", va="bottom", color="#bbf7d0", fontsize=8, fontweight="bold")

    scatter(ax, list(msg_pos.values()), 10, "#fbbf24", "#fde68a")
    scatter(ax, list(entity_pos.values()), 14, "#f97316", "#fed7aa")

    y_values = data["day"].y_future_multilabel[observed_days[-1]].tolist()
    active_classes = [ATTACK_CLASSES[i] for i, value in enumerate(y_values) if value > 0.5]
    if not active_classes:
        active_classes = ["Unknown"]
    ax.text(
        target_pos[0] + 0.55,
        target_pos[1] + 0.5,
        "Rotulos Hackmageddon",
        ha="left",
        va="center",
        color="#94a3b8",
        fontsize=9,
        fontweight="bold",
    )
    for rank, attack_class in enumerate(active_classes):
        xy = (target_pos[0] + 1.0, target_pos[1] - rank * 0.62)
        add_edge(ax, target_pos, xy, "#ef4444", lw=1.6, alpha=0.82, arrow=True, linestyle="--", rad=0.08, zorder=2)
        ax.scatter([xy[0]], [xy[1]], s=520, c="#dc2626", edgecolors="#fca5a5", linewidths=1.4, zorder=4)
        ax.text(xy[0] + 0.28, xy[1], attack_class, ha="left", va="center", color="#e5e7eb", fontsize=8, fontweight="bold")

    legend = [
        ("grupo -> mensagem", "#2563eb", "-"),
        ("dia -> mensagem", "#64748b", "-"),
        ("mensagem -> entidade", "#ea580c", "-"),
        ("mensagem similar", "#f59e0b", "--"),
        ("alvo -> classe", "#ef4444", "--"),
    ]
    for i, (label, color, style) in enumerate(legend):
        x0 = 0.25 + i * 1.35
        y0 = -2.18
        ax.plot([x0, x0 + 0.28], [y0, y0], color=color, linewidth=2, linestyle=style)
        ax.text(x0 + 0.34, y0, label, ha="left", va="center", color="#cbd5e1", fontsize=7)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    svg_path = args.output_prefix.with_suffix(".svg")
    png_path = args.output_prefix.with_suffix(".png")
    fig.savefig(svg_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(png_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {svg_path}")
    print(f"Saved: {png_path}")
    print(subtitle)


if __name__ == "__main__":
    main()
