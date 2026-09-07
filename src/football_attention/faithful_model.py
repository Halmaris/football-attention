from __future__ import annotations
import math
import torch
from torch import nn
from .config import ExperimentConfig
from .model import EncoderBlock
from .sequence_baselines import NoPlayerEventEmbedding, valid_event_order
from .variants import Vocabularies


class EventEmbedding(nn.Module):
    def __init__(
        self,
        vocabularies: Vocabularies,
        d_model: int,
        *,
        use_player_id: bool = True,
    ) -> None:
        super().__init__()
        self.use_player_id = use_player_id
        self.player = nn.Embedding(vocabularies.n_players, 48, padding_idx=0)
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
        self.projection = nn.Linear(48 + 32 + 16 + 64, d_model)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        player = self.player(batch['player'])
        if not self.use_player_id:
            player = torch.zeros_like(player)
        values = torch.cat(
            [
                player,
                self.event_type(batch['event_type']),
                self.outcome(batch['outcome']),
                self.continuous(batch['continuous']),
            ],
            dim=-1,
        )
        return self.projection(values)


class CrossAttentionPool(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError('d_model must be divisible by n_heads.')
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.query = nn.Parameter(torch.empty(1, 1, d_model))
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.output_projection = nn.Linear(d_model, d_model, bias=False)
        self.output_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        for module in [
            self.query_projection,
            self.key_projection,
            self.value_projection,
            self.output_projection,
        ]:
            module.reset_parameters()

    def forward(
        self,
        events: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_events, d_model = events.shape
        query = self.query.expand(batch_size, -1, -1)
        query = self.query_projection(query).view(
            batch_size,
            1,
            self.n_heads,
            self.d_head,
        ).transpose(1, 2)
        keys = self.key_projection(events).view(
            batch_size,
            n_events,
            self.n_heads,
            self.d_head,
        ).transpose(1, 2)
        values = self.value_projection(events).view(
            batch_size,
            n_events,
            self.n_heads,
            self.d_head,
        ).transpose(1, 2)
        scores = query @ keys.transpose(-2, -1) / math.sqrt(self.d_head)
        scores = scores.masked_fill(
            ~valid_mask[:, None, None, :],
            float('-inf'),
        )
        attention = torch.softmax(scores, dim=-1)
        pooled = attention @ values
        pooled = pooled.transpose(1, 2).contiguous().view(batch_size, 1, d_model)
        pooled = self.output_projection(pooled).squeeze(1)
        pooled = self.output_norm(pooled)
        pooled = pooled + self.feed_forward(pooled)
        return pooled, attention.squeeze(2).mean(dim=1)


class FaithfulPreShotTransformer(nn.Module):
    def __init__(
        self,
        vocabularies: Vocabularies,
        sequence_length: int,
        config: ExperimentConfig,
        *,
        use_player_id: bool = True,
        use_position_encoding: bool = True,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.use_player_id = use_player_id
        self.use_position_encoding = use_position_encoding
        self.embedding = EventEmbedding(
            vocabularies,
            config.d_model,
            use_player_id=use_player_id,
        )
        self.position_embedding = (
            nn.Embedding(sequence_length, config.d_model)
            if use_position_encoding
            else None
        )
        self.empty_event = nn.Parameter(torch.empty(config.d_model))
        nn.init.normal_(self.empty_event, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList([
            EncoderBlock(config.d_model, config.n_heads, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.pool = CrossAttentionPool(
            config.d_model,
            config.n_heads,
            config.dropout,
        )
        self.regression_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, 1),
        )

    def embed_events(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.embedding(batch)

    def forward_from_embeddings(
        self,
        event_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        original_mask = valid_mask
        effective_mask = valid_mask.clone()
        events = event_embeddings.clone()
        empty_rows = ~effective_mask.any(dim=1)
        if empty_rows.any():
            effective_mask[empty_rows, 0] = True
            events[empty_rows, 0] = self.empty_event

        if self.position_embedding is not None:
            positions = torch.arange(events.shape[1], device=events.device)
            events = events + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            events, _ = block(events, effective_mask)
        pooled, attention = self.pool(events, effective_mask)
        predictions = self.regression_head(pooled).squeeze(-1)

        if return_attention:
            attention = attention.masked_fill(~original_mask, 0.0)
            mass = attention.sum(dim=-1, keepdim=True)
            attention = attention / mass.clamp_min(1e-8)
            return predictions, attention
        return predictions

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.embed_events(batch)
        return self.forward_from_embeddings(
            embeddings,
            batch['valid_mask'],
            return_attention=return_attention,
        )

    def randomize_pool_attention(self) -> None:
        self.pool.reset_parameters()


class FaithfulPreShotGRUAttention(nn.Module):
    def __init__(
        self,
        vocabularies: Vocabularies,
        sequence_length: int,
        config: ExperimentConfig,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.use_player_id = False
        self.embedding = NoPlayerEventEmbedding(
            vocabularies,
            config.d_model,
        )
        self.input_norm = nn.LayerNorm(config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)
        self.gru = nn.GRU(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=1,
            batch_first=True,
        )
        self.empty_event = nn.Parameter(torch.empty(config.d_model))
        nn.init.normal_(self.empty_event, mean=0.0, std=0.02)
        self.pool = CrossAttentionPool(
            config.d_model,
            config.n_heads,
            config.dropout,
        )
        self.regression_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1),
        )

    def embed_events(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.embedding(batch)

    def forward_from_embeddings(
        self,
        event_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        original_mask = valid_mask
        effective_mask = valid_mask.clone()
        events = event_embeddings.clone()
        empty_rows = ~effective_mask.any(dim=1)
        if empty_rows.any():
            effective_mask[empty_rows, 0] = True
            events[empty_rows, 0] = self.empty_event

        events = self.input_dropout(self.input_norm(events))
        order = valid_event_order(effective_mask)
        compact_events = events.gather(
            1,
            order.unsqueeze(-1).expand(-1, -1, events.shape[-1]),
        )
        lengths = effective_mask.sum(dim=1)
        compact_mask = (
            torch.arange(events.shape[1], device=events.device)[None, :]
            < lengths[:, None]
        )
        encoded, _ = self.gru(compact_events)
        pooled, compact_attention = self.pool(encoded, compact_mask)
        predictions = self.regression_head(pooled).squeeze(-1)

        if return_attention:
            attention = torch.zeros_like(compact_attention)
            attention.scatter_(1, order, compact_attention)
            attention = attention.masked_fill(~original_mask, 0.0)
            mass = attention.sum(dim=-1, keepdim=True)
            attention = attention / mass.clamp_min(1e-8)
            return predictions, attention
        return predictions

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_from_embeddings(
            self.embed_events(batch),
            batch['valid_mask'],
            return_attention=return_attention,
        )

    def randomize_pool_attention(self) -> None:
        self.pool.reset_parameters()


FaithfulModel = FaithfulPreShotTransformer | FaithfulPreShotGRUAttention


def build_faithful_model(
    model_type: str,
    vocabularies: Vocabularies,
    sequence_length: int,
    config: ExperimentConfig,
) -> FaithfulModel:
    if model_type in {'gru_attention', 'gru_attention_shuffled'}:
        return FaithfulPreShotGRUAttention(
            vocabularies,
            sequence_length,
            config,
        )
    if model_type in {
        'bottleneck',
        'faithful',
        'faithful_no_player_id',
        'faithful_no_position',
    }:
        return FaithfulPreShotTransformer(
            vocabularies,
            sequence_length,
            config,
            use_player_id=model_type not in {
                'faithful_no_player_id',
                'faithful_no_position',
            },
            use_position_encoding=model_type != 'faithful_no_position',
        )
    raise ValueError(f'Unknown faithful model type: {model_type}')
