from __future__ import annotations
import math
from typing import Optional
import torch
from torch import nn


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError('d_model must be divisible by n_heads.')
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        key_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_tokens, d_model = inputs.shape
        query, key, value = self.qkv(inputs).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size,
                n_tokens,
                self.n_heads,
                self.d_head,
            ).transpose(1, 2)

        query, key, value = map(split_heads, [query, key, value])
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.d_head)
        if key_mask is not None:
            scores = scores.masked_fill(~key_mask[:, None, None, :], float('-inf'))
        attention = torch.softmax(scores, dim=-1)
        values = self.dropout(attention) @ value
        values = values.transpose(1, 2).contiguous().view(
            batch_size,
            n_tokens,
            d_model,
        )
        return self.output(values), attention


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(d_model)
        self.attention = MultiheadSelfAttention(d_model, n_heads, dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.norm_feed_forward = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.feed_forward_dropout = nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values, attention = self.attention(
            self.norm_attention(inputs),
            key_mask,
        )
        outputs = inputs + self.attention_dropout(values)
        outputs = outputs + self.feed_forward_dropout(
            self.feed_forward(self.norm_feed_forward(outputs))
        )
        return outputs, attention
