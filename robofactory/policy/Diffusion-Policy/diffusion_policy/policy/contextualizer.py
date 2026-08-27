"""CLS-DP Stage 1: the contextualizer.

A conditional VAE whose posterior sees privileged multi-agent dynamics and whose prior sees
only the agent's own current RGB frame plus the shared task instruction. KL alignment
distills the privileged signal into the prior; the posterior branch is discarded afterwards.

Loss (Eq. 9):

    L_CT = beta * KL( q(z | s^{1:N}_{t+1:t+H}) || p(z | o_t^i, l) )
           + E_q[ || D(s_t^i, z) - s^{1:N}_{t+1:t+H} ||^2 ]

Because the posterior is *residual* -- mean mu_rho + mu_E against a prior mean of mu_rho --
the mean-difference term of the Gaussian KL collapses to exactly mu_E, and the prior mean
cancels out of the KL entirely. See `_residual_kl` below. That is not a micro-optimisation:
it is why mu_rho is trained purely by reconstruction, which is the whole distillation
mechanism. Replacing the residual posterior with an independent Gaussian breaks it.
"""

import torch
import torch.nn as nn

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.normalizer import LinearNormalizer


def _residual_kl(log_sigma_prior, mu_residual, log_sigma_posterior):
    """Closed-form KL(q || p) for the residual Gaussian parameterisation.

    With mu_q = mu_rho + mu_E, sigma_q = sigma_E, mu_p = mu_rho, sigma_p = sigma_rho:

        KL = sum_d [ log(sigma_rho/sigma_E)
                     + (sigma_E^2 + mu_E^2) / (2 sigma_rho^2)
                     - 1/2 ]

    Returns per-sample KL of shape (B,).
    """
    var_ratio = torch.exp(2.0 * (log_sigma_posterior - log_sigma_prior))
    mean_term = (mu_residual**2) / (2.0 * torch.exp(2.0 * log_sigma_prior))
    per_dim = (log_sigma_prior - log_sigma_posterior) + 0.5 * var_ratio + mean_term - 0.5
    return per_dim.sum(dim=-1)


class Contextualizer(ModuleAttrMixin):
    def __init__(
        self,
        prior_net: nn.Module,
        ma_encoder: nn.Module,
        ma_decoder: nn.Module,
        agent_id: int = 0,
        n_agents: int = 2,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.prior_net = prior_net
        self.ma_encoder = ma_encoder
        self.ma_decoder = ma_decoder
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.latent_dim = latent_dim
        self.normalizer = LinearNormalizer()

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    # ---------------------------------------------------------------- inference

    def prior_distribution(self, image_tokens, text_tokens, text_mask=None, return_attn=False):
        return self.prior_net(
            image_tokens, text_tokens, text_mask=text_mask, return_attn=return_attn
        )

    def sample_latent(
        self, image_tokens, text_tokens, text_mask=None, sample=True, return_attn=False
    ):
        """Draw z from the observation-conditioned prior. Used at Stage 2 and deployment."""
        mu, log_sigma, info = self.prior_distribution(
            image_tokens, text_tokens, text_mask=text_mask, return_attn=return_attn
        )
        if sample:
            latent = mu + torch.exp(log_sigma) * torch.randn_like(mu)
        else:
            latent = mu
        return latent, info

    # ----------------------------------------------------------------- training

    def compute_loss(self, batch, beta: float = 0.1):
        """
        batch keys:
            own_state:          (B, state_dim)          s_t^i
            future_states:      (B, N, F, state_dim)    s^{1:N}_{t+1:t+F}
            prior_image_tokens: (B, M, feature_dim)
            prior_text_tokens:  (B, Lt, feature_dim)
            prior_text_mask:    (B, Lt)
        """
        own_state = self.normalizer["state"].normalize(batch["own_state"])
        future = self.normalizer["state"].normalize(batch["future_states"])

        mu_prior, log_sigma_prior, _ = self.prior_net(
            batch["prior_image_tokens"],
            batch["prior_text_tokens"],
            text_mask=batch.get("prior_text_mask"),
        )
        mu_residual, log_sigma_posterior = self.ma_encoder(future)

        mu = mu_prior + mu_residual
        latent = mu + torch.exp(log_sigma_posterior) * torch.randn_like(mu)

        reconstruction = self.ma_decoder(own_state, latent)

        squared_error = (reconstruction - future) ** 2
        # ||.||_2^2 over (N, F, state_dim), averaged across the batch
        recon_loss = squared_error.sum(dim=(1, 2, 3)).mean()

        kl = _residual_kl(log_sigma_prior, mu_residual, log_sigma_posterior).mean()
        loss = beta * kl + recon_loss

        with torch.no_grad():
            metrics = self._diagnostics(squared_error, future, mu_residual,
                                        log_sigma_prior, log_sigma_posterior)
            metrics["ctx_loss"] = loss.item()
            metrics["ctx_kl"] = kl.item()
            metrics["ctx_recon"] = recon_loss.item()
            metrics["ctx_beta"] = beta

        return loss, metrics

    def _diagnostics(self, squared_error, future, mu_residual,
                     log_sigma_prior, log_sigma_posterior):
        """Split reconstruction error into own-agent and teammate components.

        `recon_others` is the metric that matters. The agent's own current state already
        explains much of its own future, so only the teammate term proves that z carries
        coordination information. It is compared against a batch-mean predictor: if
        `recon_others` does not beat `recon_others_baseline`, Stage 2 will not work.
        """
        agent_id = self.agent_id
        others = [j for j in range(self.n_agents) if j != agent_id]

        per_element_own = squared_error[:, agent_id].mean().item()

        if others:
            other_error = squared_error[:, others]
            per_element_others = other_error.mean().item()

            other_target = future[:, others]
            baseline = ((other_target - other_target.mean(dim=0, keepdim=True)) ** 2).mean()
            baseline_others = baseline.item()
        else:
            per_element_others = 0.0
            baseline_others = 0.0

        return {
            "ctx_recon_own": per_element_own,
            "ctx_recon_others": per_element_others,
            "ctx_recon_others_baseline": baseline_others,
            "ctx_mu_residual_norm": mu_residual.norm(dim=-1).mean().item(),
            "ctx_sigma_prior": torch.exp(log_sigma_prior).mean().item(),
            "ctx_sigma_posterior": torch.exp(log_sigma_posterior).mean().item(),
        }
