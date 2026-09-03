"""Action chunk autoencoder used by the latent-action flow-matching variant.

Stage 2a (the autoencoder) learns a compact latent representation of an action chunk
of length `horizon` consisting of `action_dim`-D joint vectors.

The Stage 2 flow expert then:
  * encodes normalized (B, horizon, action_dim) -> (B, n_tokens, latent_dim)
  * runs the rectified-flow transport in latent-token space
  * decodes back to (B, horizon, action_dim) so the inherited CLS-DP action slicing
    and execution-time interfaces remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


class ActionChunkAutoencoder(nn.Module):
    """Lightweight token autoencoder for (B, horizon, action_dim) chunks.

    The architecture is intentionally simple: we only need a deterministic,
    reconstruction-capable mapping for the gate and for later experiments with the
    latent-action Stage 2 transport.
    """

    def __init__(
        self,
        *,
        horizon: int,
        action_dim: int,
        n_tokens: int,
        latent_dim: int,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        if horizon % n_tokens != 0:
            raise ValueError(
                f"horizon ({horizon}) must be divisible by n_tokens ({n_tokens})."
            )

        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.n_tokens = int(n_tokens)
        self.latent_dim = int(latent_dim)

        self.chunk_len = self.horizon // self.n_tokens
        in_dim = self.action_dim * self.chunk_len

        # A small MLP with explicit tokenization. (d_model/n_layers/n_heads are accepted
        # from config for future upgrades; today they do not alter this baseline.)
        hidden = max(latent_dim, d_model)
        self.to_latent = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Mish(),
            nn.Linear(hidden, self.latent_dim),
        )
        self.from_latent = nn.Sequential(
            nn.Linear(self.latent_dim, hidden),
            nn.Mish(),
            nn.Linear(hidden, in_dim),
        )

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode (B, horizon, action_dim) -> (B, n_tokens, latent_dim)."""
        if actions.ndim != 3:
            raise ValueError(f"actions must have shape (B, horizon, action_dim), got {actions.shape}")
        b, t, d = actions.shape
        if t != self.horizon or d != self.action_dim:
            raise ValueError(
                f"expected (B, horizon={self.horizon}, action_dim={self.action_dim}), got {actions.shape}"
            )
        x = actions.view(b, self.n_tokens, self.chunk_len * self.action_dim)
        return self.to_latent(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode (B, n_tokens, latent_dim) -> (B, horizon, action_dim)."""
        if latent.ndim != 3:
            raise ValueError(
                f"latent must have shape (B, n_tokens, latent_dim), got {latent.shape}"
            )
        b, n, ld = latent.shape
        if n != self.n_tokens or ld != self.latent_dim:
            raise ValueError(
                f"expected (B, n_tokens={self.n_tokens}, latent_dim={self.latent_dim}), got {latent.shape}"
            )
        x = self.from_latent(latent)
        return x.view(b, self.horizon, self.action_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(actions))

