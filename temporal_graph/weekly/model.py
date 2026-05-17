from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import HGTConv, Linear


class WeeklyMultiTaskForecaster(nn.Module):
    """HGT encoder with count, type, and compatibility binary heads."""

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        hidden_dim: int,
        num_threat_types: int,
        num_count_buckets: int,
        num_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.node_types = list(metadata[0])
        self.projections = nn.ModuleDict(
            {node_type: Linear(-1, hidden_dim) for node_type in self.node_types}
        )
        self.convs = nn.ModuleList(
            [
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=metadata,
                    heads=heads,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleDict(
            {node_type: nn.LayerNorm(hidden_dim) for node_type in self.node_types}
        )
        self.dropout = nn.Dropout(dropout)
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.count_bucket_head = nn.Linear(hidden_dim, num_count_buckets)
        self.type_multilabel_head = nn.Linear(hidden_dim, num_threat_types)
        self.type_count_head = nn.Linear(hidden_dim, num_threat_types)

    def encode(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        h_dict = {
            node_type: self.dropout(self.projections[node_type](x).relu())
            for node_type, x in x_dict.items()
        }
        for conv in self.convs:
            conv_out = conv(h_dict, edge_index_dict)
            next_dict = {}
            for node_type, previous in h_dict.items():
                current = conv_out.get(node_type)
                if current is None:
                    current = previous
                next_dict[node_type] = self.norms[node_type](previous + self.dropout(current).relu())
            h_dict = next_dict
        return h_dict

    def _attention_pool(
        self,
        message_h: torch.Tensor,
        day_h: torch.Tensor,
        message_day_edge_index: torch.Tensor | None,
    ) -> torch.Tensor:
        pooled = day_h.new_zeros(day_h.shape)
        if message_day_edge_index is None or message_day_edge_index.numel() == 0:
            return pooled
        src = message_day_edge_index[0]
        dst = message_day_edge_index[1]
        for day_index in range(day_h.size(0)):
            mask = dst == day_index
            if not bool(mask.any()):
                continue
            msg_vectors = message_h[src[mask]]
            day_vectors = day_h[day_index].expand_as(msg_vectors)
            scores = self.attention(torch.cat([msg_vectors, day_vectors], dim=-1)).squeeze(-1)
            weights = torch.softmax(scores, dim=0).unsqueeze(-1)
            pooled[day_index] = (weights * msg_vectors).sum(dim=0)
        return pooled

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        h_dict = self.encode(x_dict, edge_index_dict)
        gru_out, _ = self.temporal_gru(h_dict["day"].unsqueeze(0))
        h_dict = {**h_dict, "day": gru_out.squeeze(0)}
        message_day_edge_index = edge_index_dict.get(("message", "observed_before_snapshot", "day"))
        pooled_messages = self._attention_pool(h_dict["message"], h_dict["day"], message_day_edge_index)
        day_repr = self.shared(torch.cat([h_dict["day"], pooled_messages], dim=-1))
        return {
            "count_bucket": self.count_bucket_head(day_repr),
            "type_multilabel": self.type_multilabel_head(day_repr),
            "type_counts": self.type_count_head(day_repr),
        }
