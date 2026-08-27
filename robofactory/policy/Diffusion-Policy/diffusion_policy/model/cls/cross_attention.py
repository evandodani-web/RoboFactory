"""Cross-attention block used to inject the collaborative latent into the action-expert.

The paper injects z into the downsampling and bottleneck stages of the U-Net only, leaving
the upsampling path untouched, on the grounds that those features preserve multi-scale
spatial information and give stronger control representations.

The output projection is zero-initialised so each block is an exact identity at step 0.
The action-expert therefore starts as vanilla Diffusion Policy and learns to use z from
there, which is standard practice for bolting a conditioning branch onto an existing
architecture and noticeably stabilises early training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention1d(nn.Module):
    """Cross-attention from 1D temporal U-Net features into a latent token context."""

    def __init__(
        self,
        channels: int,
        context_dim: int,
        n_heads: int = 4,
        head_dim: int = None,
        n_groups: int = 8,
    ):
        super().__init__()
        if head_dim is None:
            head_dim = max(channels // n_heads, 1)
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim

        self.norm = nn.GroupNorm(min(n_groups, channels), channels)
        self.to_q = nn.Linear(channels, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, channels)

        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def _split_heads(self, x):
        batch, length, _ = x.shape
        return x.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, context):
        """
        Args:
            x:       (B, C, T) U-Net features
            context: (B, M, context_dim) latent tokens

        Returns:
            (B, C, T)
        """
        residual = x
        hidden = self.norm(x).transpose(1, 2)

        q = self._split_heads(self.to_q(hidden))
        k = self._split_heads(self.to_k(context))
        v = self._split_heads(self.to_v(context))

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[1], -1)

        out = self.to_out(attn).transpose(1, 2)
        return residual + out


class LatentTokenizer(nn.Module):
    """Projects a flat latent vector into a short sequence of conditioning tokens.

    The paper does not say how a single latent vector becomes an attention context. One
    token would work; a handful gives the cross-attention slightly more capacity at
    negligible cost.
    """

    def __init__(self, latent_dim: int, token_dim: int, n_tokens: int = 4):
        super().__init__()
        self.n_tokens = n_tokens
        self.token_dim = token_dim
        self.proj = nn.Linear(latent_dim, n_tokens * token_dim)
        self.token_embed = nn.Parameter(torch.randn(n_tokens, token_dim) * 0.02)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, latent):
        batch = latent.shape[0]
        tokens = self.proj(latent).view(batch, self.n_tokens, self.token_dim)
        return self.norm(tokens + self.token_embed)
