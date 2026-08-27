"""Observation-conditioned prior for the CLS-DP contextualizer.

Implements Eq. 7 of the paper:

    p_theta_i(z_t^i | o_t^i, l) = N(mu_rho, diag(sigma_rho^2))

The prior fuses frozen SigLIP image tokens (from the agent's *current* frame only) with
frozen SigLIP text tokens (the shared task instruction) through Transformer layers.

Two things worth knowing:

1. Only the current frame is used, never the L=3 history. The paper is explicit about
   this: conditioning the prior on longer histories makes it overly informative, which
   stops z from having to learn the collaborative dynamics distilled from the posterior.

2. The fusion deliberately cross-attends a small set of learned queries over a single
   concatenated [text ; image] context. That makes the text-vs-image attention split
   directly measurable, which is what Fig. 4 of the paper reports.
"""

import torch
import torch.nn as nn


class FusionLayer(nn.Module):
    """Pre-norm self-attention over queries, then cross-attention into the context."""

    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.0):
        super().__init__()
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, queries, context, context_padding_mask=None, return_attn=False):
        h = self.norm_self(queries)
        attn_out, _ = self.self_attn(h, h, h, need_weights=False)
        queries = queries + attn_out

        h = self.norm_cross(queries)
        attn_out, attn_weights = self.cross_attn(
            h,
            context,
            context,
            key_padding_mask=context_padding_mask,
            need_weights=return_attn,
            average_attn_weights=True,
        )
        queries = queries + attn_out

        queries = queries + self.ff(self.norm_ff(queries))
        return queries, attn_weights


class PriorNet(nn.Module):
    """Maps (SigLIP image tokens, SigLIP text tokens) -> Gaussian prior over z."""

    def __init__(
        self,
        feature_dim: int = 768,
        latent_dim: int = 256,
        d_model: int = 768,
        n_layers: int = 2,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        n_query: int = 1,
        log_sigma_min: float = -5.0,
        log_sigma_max: float = 2.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max

        self.text_proj = nn.Linear(feature_dim, d_model)
        self.image_proj = nn.Linear(feature_dim, d_model)
        # index 0 = text, index 1 = image
        self.modality_embed = nn.Parameter(torch.zeros(2, d_model))
        self.query = nn.Parameter(torch.randn(n_query, d_model) * 0.02)

        self.layers = nn.ModuleList(
            [
                FusionLayer(d_model, n_heads, dim_feedforward, dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(d_model)
        self.to_mu = nn.Linear(d_model, latent_dim)
        self.to_log_sigma = nn.Linear(d_model, latent_dim)

        # Start at sigma_rho = 1 so that, combined with the zero-initialised residual head
        # in MAKinematicsEncoder, the KL term begins at exactly zero.
        nn.init.zeros_(self.to_log_sigma.weight)
        nn.init.zeros_(self.to_log_sigma.bias)

    def forward(self, image_tokens, text_tokens, text_mask=None, return_attn=False):
        """
        Args:
            image_tokens: (B, M, feature_dim) cached SigLIP features for o_t^i
            text_tokens:  (B, Lt, feature_dim) cached SigLIP features for l
            text_mask:    (B, Lt) with 1 for real tokens, 0 for padding
            return_attn:  also return the text/image attention split (Fig. 4)

        Returns:
            mu:        (B, latent_dim)
            log_sigma: (B, latent_dim)
            info:      dict, populated only when return_attn is True
        """
        batch_size = image_tokens.shape[0]
        n_text = text_tokens.shape[1]

        text = self.text_proj(text_tokens) + self.modality_embed[0]
        image = self.image_proj(image_tokens) + self.modality_embed[1]
        context = torch.cat([text, image], dim=1)

        padding_mask = None
        if text_mask is not None:
            image_mask = torch.ones(
                image.shape[:2], device=image.device, dtype=text_mask.dtype
            )
            valid = torch.cat([text_mask, image_mask], dim=1)
            # nn.MultiheadAttention expects True where a key should be ignored.
            padding_mask = valid < 0.5

        queries = self.query.unsqueeze(0).expand(batch_size, -1, -1)

        attn_accumulator = []
        for layer in self.layers:
            queries, attn = layer(
                queries, context, padding_mask, return_attn=return_attn
            )
            if return_attn and attn is not None:
                attn_accumulator.append(attn)

        pooled = self.norm_out(queries).mean(dim=1)
        mu = self.to_mu(pooled)
        log_sigma = self.to_log_sigma(pooled).clamp(
            self.log_sigma_min, self.log_sigma_max
        )

        info = {}
        if return_attn and attn_accumulator:
            # (n_layers, B, n_query, n_text + n_image) -> per-modality mass
            weights = torch.stack(attn_accumulator, dim=0)
            text_mass = weights[..., :n_text].sum(-1)
            image_mass = weights[..., n_text:].sum(-1)
            total = (text_mass + image_mass).clamp_min(1e-8)
            info["text_attention"] = (text_mass / total).mean()
            info["image_attention"] = (image_mass / total).mean()

        return mu, log_sigma, info
