from __future__ import annotations
import torch
from torch import nn
from .config import ExperimentConfig
from .variants import Vocabularies


class NoPlayerEventEmbedding(nn.Module):
    def __init__(self, vocabularies: Vocabularies, d_model: int) -> None:
        super().__init__()
        self.event_type = nn.Embedding(
            vocabularies.n_event_types,
            32,
            padding_idx=0,
        )
        self.outcome = nn.Embedding(
            vocabularies.n_outcomes,
            16,
            padding_idx=0,
        )
        self.continuous = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        self.projection = nn.Linear(32 + 16 + 64, d_model)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        values = torch.cat(
            [
                self.event_type(batch['event_type']),
                self.outcome(batch['outcome']),
                self.continuous(batch['continuous']),
            ],
            dim=-1,
        )
        return self.projection(values)


class DeepSetsRegressor(nn.Module):
    def __init__(
        self,
        vocabularies: Vocabularies,
        config: ExperimentConfig,
    ) -> None:
        super().__init__()
        self.embedding = NoPlayerEventEmbedding(vocabularies, config.d_model)
        self.event_network = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
        )
        self.count_projection = nn.Linear(1, config.d_model)
        self.empty_representation = nn.Parameter(torch.empty(config.d_model))
        nn.init.normal_(self.empty_representation, mean=0.0, std=0.02)
        self.regression_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        valid = batch['valid_mask']
        encoded = self.event_network(self.embedding(batch))
        mask = valid.unsqueeze(-1).to(encoded.dtype)
        counts = valid.sum(dim=1, keepdim=True)
        pooled = (encoded * mask).sum(dim=1) / counts.clamp_min(1)
        normalized_count = torch.log1p(counts.to(encoded.dtype))
        normalized_count = normalized_count / torch.log1p(
            torch.tensor(
                valid.shape[1],
                dtype=encoded.dtype,
                device=encoded.device,
            )
        )
        pooled = pooled + self.count_projection(normalized_count)
        empty = counts.squeeze(1).eq(0)
        if empty.any():
            pooled = pooled.clone()
            pooled[empty] = self.empty_representation
        return self.regression_head(pooled).squeeze(-1)


def valid_event_order(valid_mask: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(valid_mask.shape[1], device=valid_mask.device)
    positions = positions.expand(valid_mask.shape[0], -1)
    sort_key = (
        positions
        + (~valid_mask).to(positions.dtype) * valid_mask.shape[1]
    )
    return sort_key.argsort(dim=1)


def compact_valid_events(
    events: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    order = valid_event_order(valid_mask)
    return events.gather(
        1,
        order.unsqueeze(-1).expand(-1, -1, events.shape[-1]),
    )


class GRURegressor(nn.Module):
    def __init__(
        self,
        vocabularies: Vocabularies,
        config: ExperimentConfig,
    ) -> None:
        super().__init__()
        self.embedding = NoPlayerEventEmbedding(vocabularies, config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)
        self.gru = nn.GRU(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=1,
            batch_first=True,
        )
        self.empty_representation = nn.Parameter(torch.empty(config.d_model))
        nn.init.normal_(self.empty_representation, mean=0.0, std=0.02)
        self.regression_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        valid = batch['valid_mask']
        events = self.input_dropout(self.input_norm(self.embedding(batch)))
        events = compact_valid_events(events, valid)
        outputs, _ = self.gru(events)
        lengths = valid.sum(dim=1)
        last_positions = (lengths - 1).clamp_min(0)
        rows = torch.arange(events.shape[0], device=events.device)
        representation = outputs[rows, last_positions]
        empty = lengths.eq(0)
        if empty.any():
            representation = representation.clone()
            representation[empty] = self.empty_representation
        return self.regression_head(representation).squeeze(-1)


def build_sequence_baseline(
    model_type: str,
    vocabularies: Vocabularies,
    config: ExperimentConfig,
) -> nn.Module:
    if model_type == 'deepsets':
        return DeepSetsRegressor(vocabularies, config)
    if model_type == 'gru':
        return GRURegressor(vocabularies, config)
    raise ValueError(f'Unknown sequence baseline: {model_type}')


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
