"""Minimal nanoTabPFN-style model for maternal tabular tasks.

Architecture:
    feature/target encoding
    -> feature attention
    -> patient attention
    -> MLP
    -> repeat
    -> query-target decoder

The model uses no positional embeddings. Query patients can attend to context
patients, while context patients never attend to query patients.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FeatureEncoder(nn.Module):
    """Normalize from context rows only, then embed each scalar cell."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Linear(1, d_model)

    def forward(self, x: torch.Tensor, n_context: int) -> torch.Tensor:
        context = x[:, :n_context]
        mean = context.mean(dim=1, keepdim=True)
        std = context.std(dim=1, keepdim=True, correction=0).clamp_min(1e-6)
        x = ((x - mean) / std).clamp(-100.0, 100.0)
        return self.projection(x.unsqueeze(-1))


class TargetEncoder(nn.Module):
    """Embed context labels and use their mean as the query placeholder."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Linear(1, d_model)

    def forward(self, y_context: torch.Tensor, n_query: int) -> torch.Tensor:
        if y_context.ndim == 2:
            y_context = y_context.unsqueeze(-1)

        y_context = y_context.float()
        placeholder = y_context.mean(dim=1, keepdim=True).expand(
            -1, n_query, -1
        )
        y_all = torch.cat([y_context, placeholder], dim=1)
        return self.projection(y_all.unsqueeze(-1))


class MaternalTransformerBlock(nn.Module):
    """Feature attention, patient attention, then an MLP."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.feature_attention = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.patient_attention = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )

        self.norm_features = nn.LayerNorm(d_model)
        self.norm_patients = nn.LayerNorm(d_model)
        self.norm_mlp = nn.LayerNorm(d_model)

        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor, n_context: int) -> torch.Tensor:
        batch, rows, columns, d_model = x.shape

        # Attention across feature/target columns within each patient.
        feature_tokens = x.reshape(batch * rows, columns, d_model)
        feature_update, _ = self.feature_attention(
            feature_tokens,
            feature_tokens,
            feature_tokens,
            need_weights=False,
        )
        feature_tokens = self.norm_features(feature_tokens + feature_update)
        x = feature_tokens.reshape(batch, rows, columns, d_model)

        # Attention across patients within each feature column.
        patient_tokens = x.transpose(1, 2).reshape(
            batch * columns, rows, d_model
        )
        context = patient_tokens[:, :n_context]
        query = patient_tokens[:, n_context:]

        context_update, _ = self.patient_attention(
            context,
            context,
            context,
            need_weights=False,
        )
        query_update, _ = self.patient_attention(
            query,
            context,
            context,
            need_weights=False,
        )

        patient_tokens = torch.cat(
            [context + context_update, query + query_update],
            dim=1,
        )
        patient_tokens = self.norm_patients(patient_tokens)

        x = patient_tokens.reshape(batch, columns, rows, d_model).transpose(1, 2)

        mlp_update = self.fc2(F.gelu(self.fc1(x)))
        return self.norm_mlp(x + mlp_update)


class NanoMaternalPFN(nn.Module):
    """Small PFN backbone for binary maternal-health-shaped tasks."""

    def __init__(
        self,
        *,
        d_model: int = 96,
        n_heads: int = 4,
        hidden_dim: int = 192,
        n_layers: int = 3,
        n_outputs: int = 2,
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        if n_outputs < 2:
            raise ValueError("n_outputs must be at least 2")

        self.feature_encoder = FeatureEncoder(d_model)
        self.target_encoder = TargetEncoder(d_model)
        self.blocks = nn.ModuleList(
            [
                MaternalTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    hidden_dim=hidden_dim,
                )
                for _ in range(n_layers)
            ]
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_outputs),
        )

    def forward(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_query: torch.Tensor,
    ) -> torch.Tensor:
        """Return query logits with shape [batch, query_rows, n_outputs]."""

        self._validate_inputs(x_context, y_context, x_query)

        n_context = x_context.shape[1]
        n_query = x_query.shape[1]

        x_all = torch.cat([x_context, x_query], dim=1).float()
        x_tokens = self.feature_encoder(x_all, n_context)

        y_tokens = self.target_encoder(y_context, n_query)
        tokens = torch.cat([x_tokens, y_tokens], dim=2)

        for block in self.blocks:
            tokens = block(tokens, n_context)

        query_target_tokens = tokens[:, n_context:, -1, :]
        return self.decoder(query_target_tokens)

    @staticmethod
    def _validate_inputs(
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_query: torch.Tensor,
    ) -> None:
        if x_context.ndim != 3 or x_query.ndim != 3:
            raise ValueError("x_context and x_query must have shape [B, N, F]")
        if y_context.ndim not in (2, 3):
            raise ValueError("y_context must have shape [B, N] or [B, N, 1]")
        if x_context.shape[0] != x_query.shape[0]:
            raise ValueError("context and query batch sizes must match")
        if x_context.shape[2] != x_query.shape[2]:
            raise ValueError("context and query feature counts must match")
        if y_context.shape[0] != x_context.shape[0]:
            raise ValueError("x_context and y_context batch sizes must match")
        if y_context.shape[1] != x_context.shape[1]:
            raise ValueError("x_context and y_context row counts must match")
        if x_context.shape[1] < 2:
            raise ValueError("at least 2 context rows are required")
        if x_query.shape[1] < 1:
            raise ValueError("at least 1 query row is required")


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable model parameters."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
