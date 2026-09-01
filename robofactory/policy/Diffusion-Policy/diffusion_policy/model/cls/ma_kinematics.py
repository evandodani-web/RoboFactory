"""Multi-agent kinematics encoder/decoder -- the privileged branch of the CLS-DP CVAE.

These two modules exist only during Stage 1 and are discarded at deployment.

Encoder (Eq. 8): reads the privileged future joint trajectories of *all* agents,
s^{1:N}_{t+1:t+H}, and predicts a *residual* on the prior:

    q_psi_i(z | s^{1:N}) = N(mu_rho + mu_E, diag(sigma_E^2))

Decoder: reconstructs all N agents' future trajectories from only (s_t^i, z). Because the
agent's own current state cannot explain what the other arms are about to do, the only way
to drive the reconstruction error down is for z to carry teammate information. That is the
entire distillation mechanism.

Both heads are zero-initialised so that at step 0 we get mu_E = 0 and sigma_E = 1, which
combined with PriorNet's sigma_rho = 1 makes the KL start at exactly zero. The residual
then grows only as far as reconstruction demands.
"""

import torch
import torch.nn as nn


class MAKinematicsEncoder(nn.Module):
    """Privileged posterior: all agents' future states -> residual latent parameters."""

    def __init__(
        self,
        state_dim: int = 8,
        n_agents: int = 2,
        n_future_states: int = 8,
        latent_dim: int = 256,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        log_sigma_min: float = -5.0,
        log_sigma_max: float = 2.0,
        deterministic: bool = False,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.n_future_states = n_future_states
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max
        # Deterministic variant: emit only the residual, with no scale head.
        self.deterministic = deterministic

        self.in_proj = nn.Linear(state_dim, d_model)
        self.agent_embed = nn.Parameter(torch.zeros(n_agents, d_model))
        self.time_embed = nn.Parameter(torch.zeros(n_future_states, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor is incompatible with norm_first and only warns; disable it.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm_out = nn.LayerNorm(d_model)
        self.to_mu_residual = nn.Linear(d_model, latent_dim)
        nn.init.zeros_(self.to_mu_residual.weight)
        nn.init.zeros_(self.to_mu_residual.bias)

        if deterministic:
            self.to_log_sigma = None
        else:
            self.to_log_sigma = nn.Linear(d_model, latent_dim)
            nn.init.zeros_(self.to_log_sigma.weight)
            nn.init.zeros_(self.to_log_sigma.bias)

    def forward(self, future_states):
        """
        Args:
            future_states: (B, N, F, state_dim), normalized

        Returns:
            mu_residual: (B, latent_dim)
            log_sigma:   (B, latent_dim), or None in the deterministic variant
        """
        batch_size = future_states.shape[0]

        tokens = self.in_proj(future_states)
        tokens = tokens + self.agent_embed[None, :, None, :]
        tokens = tokens + self.time_embed[None, None, :, :]
        tokens = tokens.flatten(1, 2)

        cls = self.cls_token.unsqueeze(0).expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        hidden = self.encoder(tokens)[:, 0]
        hidden = self.norm_out(hidden)

        mu_residual = self.to_mu_residual(hidden)
        if self.to_log_sigma is None:
            return mu_residual, None
        log_sigma = self.to_log_sigma(hidden).clamp(
            self.log_sigma_min, self.log_sigma_max
        )
        return mu_residual, log_sigma


class MAKinematicsDecoder(nn.Module):
    """Reconstructs all agents' future states from (own current state, latent)."""

    def __init__(
        self,
        state_dim: int = 8,
        n_agents: int = 2,
        n_future_states: int = 8,
        latent_dim: int = 256,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.n_future_states = n_future_states

        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        # index 0 = latent, index 1 = own state
        self.context_embed = nn.Parameter(torch.zeros(2, d_model))

        self.agent_embed = nn.Parameter(torch.randn(n_agents, d_model) * 0.02)
        self.time_embed = nn.Parameter(torch.randn(n_future_states, d_model) * 0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.norm_out = nn.LayerNorm(d_model)
        self.to_state = nn.Linear(d_model, state_dim)

    def forward(self, own_state, latent):
        """
        Args:
            own_state: (B, state_dim) normalized s_t^i
            latent:    (B, latent_dim)

        Returns:
            (B, N, F, state_dim) reconstruction of s^{1:N}_{t+1:t+F}
        """
        batch_size = own_state.shape[0]

        context = torch.stack(
            [
                self.latent_proj(latent) + self.context_embed[0],
                self.state_proj(own_state) + self.context_embed[1],
            ],
            dim=1,
        )

        queries = self.agent_embed[:, None, :] + self.time_embed[None, :, :]
        queries = queries.reshape(1, self.n_agents * self.n_future_states, -1)
        queries = queries.expand(batch_size, -1, -1)

        hidden = self.decoder(tgt=queries, memory=context)
        out = self.to_state(self.norm_out(hidden))
        return out.reshape(
            batch_size, self.n_agents, self.n_future_states, self.state_dim
        )
