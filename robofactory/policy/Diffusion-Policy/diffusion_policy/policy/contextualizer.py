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

Three variants live here, selected by flags:

  * baseline / Study B      -- one latent, Gaussian posterior
  * deterministic / DET     -- `deterministic=True`; no scale heads, KL becomes its
                               unit-variance limit 0.5*||z_E||^2
  * factorized / FG         -- `factorize=True`; the prior latent is split into a self half
                               and a teammate half with separate decoders, and only the
                               teammate half receives the privileged residual

Independently of those, two optional probe decoders can be attached. They are pure
measurement by default and are described in `_probe_outputs`.
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


def _residual_l2(mu_residual):
    """Alignment term for the deterministic variant. Per-sample, shape (B,).

    This is exactly `_residual_kl` in the unit-variance limit: substituting
    sigma_prior = sigma_posterior = 1 collapses the Gaussian KL to

        0.5 * ||mu_E||^2

    so the deterministic variant is not a different objective, it is the same one with
    the scale parameters frozen out. Crucially the distillation geometry survives: the
    alignment term still depends only on the residual, so the prior output receives no
    gradient from it and is trained purely by reconstruction.
    """
    return 0.5 * (mu_residual**2).sum(dim=-1)


def _trajectory_mse(error):
    """||.||_2^2 over (agents, horizon, state_dim), averaged across the batch."""
    return error.sum(dim=(1, 2, 3)).mean()


class Contextualizer(ModuleAttrMixin):
    def __init__(
        self,
        prior_net: nn.Module,
        ma_encoder: nn.Module,
        ma_decoder: nn.Module = None,
        decoder_self: nn.Module = None,
        decoder_team: nn.Module = None,
        prior_probe: nn.Module = None,
        leak_probe: nn.Module = None,
        agent_id: int = 0,
        n_agents: int = 2,
        latent_dim: int = 256,
        deterministic: bool = False,
        factorize: bool = False,
        self_dim: int = 128,
        prior_probe_stop_grad: bool = True,
        leak_probe_stop_grad: bool = True,
        probe_weight: float = 1.0,
    ):
        super().__init__()
        self.prior_net = prior_net
        self.ma_encoder = ma_encoder
        self.ma_decoder = ma_decoder
        self.decoder_self = decoder_self
        self.decoder_team = decoder_team
        self.prior_probe = prior_probe
        self.leak_probe = leak_probe

        self.agent_id = agent_id
        self.n_agents = n_agents
        self.latent_dim = latent_dim
        self.other_ids = [j for j in range(n_agents) if j != agent_id]

        # When True the encoders emit z directly, no reparameterization happens anywhere,
        # and the KL is replaced by its unit-variance limit (see _residual_l2). Must match
        # the `deterministic` flag on prior_net and ma_encoder.
        self.deterministic = deterministic

        # When True the prior's latent is sliced into [z_self ; z_team]. The split is a
        # slicing convention rather than separate output heads, so prior_net is unchanged
        # and Stage 2 keeps consuming the full latent exactly as before.
        self.factorize = factorize
        self.self_dim = self_dim
        self.prior_probe_stop_grad = prior_probe_stop_grad
        self.leak_probe_stop_grad = leak_probe_stop_grad
        self.probe_weight = probe_weight

        if factorize:
            if n_agents < 2:
                raise ValueError(
                    f"factorize requires at least 2 agents, got {n_agents}: a teammate "
                    "decoder over zero teammates is meaningless"
                )
            if not 0 < self_dim < latent_dim:
                raise ValueError(
                    f"self_dim must lie strictly inside (0, {latent_dim}), got {self_dim}"
                )
            if decoder_self is None or decoder_team is None:
                raise ValueError(
                    "factorize requires both decoder_self and decoder_team"
                )
        elif ma_decoder is None:
            raise ValueError("the non-factorized variant requires ma_decoder")

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
        """Draw z from the observation-conditioned prior. Used at Stage 2 and deployment.

        Returns the *full* latent in both variants. Factorization is internal to Stage 1;
        the action expert conditions on the whole thing.
        """
        mu, log_sigma, info = self.prior_distribution(
            image_tokens, text_tokens, text_mask=text_mask, return_attn=return_attn
        )
        if log_sigma is None or not sample:
            latent = mu
        else:
            latent = mu + torch.exp(log_sigma) * torch.randn_like(mu)
        return latent, info

    def split_latent(self, latent):
        """Slice a full latent into (z_self, z_team). z_self is None when not factorized."""
        if not self.factorize:
            return None, latent
        return latent[..., : self.self_dim], latent[..., self.self_dim :]

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

        own_target = future[:, self.agent_id : self.agent_id + 1]
        team_target = future[:, self.other_ids] if self.other_ids else None

        mu_prior, log_sigma_prior, _ = self.prior_net(
            batch["prior_image_tokens"],
            batch["prior_text_tokens"],
            text_mask=batch.get("prior_text_mask"),
        )
        mu_residual, log_sigma_posterior = self.ma_encoder(future)

        if self.factorize:
            recon_loss, own_error, team_error, alignment, z_self, z_team = (
                self._factorized_forward(
                    own_state, own_target, team_target, mu_prior, mu_residual,
                    log_sigma_prior, log_sigma_posterior, beta,
                )
            )
        else:
            recon_loss, own_error, team_error, alignment, z_self, z_team = (
                self._monolithic_forward(
                    own_state, future, mu_prior, mu_residual,
                    log_sigma_prior, log_sigma_posterior,
                )
            )

        loss = beta * alignment + recon_loss

        probe_loss, probe_metrics = self._probe_outputs(
            own_state, team_target, z_self, z_team
        )
        if probe_loss is not None:
            loss = loss + self.probe_weight * probe_loss

        with torch.no_grad():
            metrics = self._diagnostics(
                own_error, team_error, team_target, mu_residual,
                log_sigma_prior, log_sigma_posterior,
            )
            metrics.update(probe_metrics)
            metrics.update(self._probe_comparisons(metrics))
            # Key stays `ctx_kl` across all variants so the logging and gate code is
            # shared; in the deterministic variant it holds the L2 alignment term.
            metrics["ctx_loss"] = loss.item()
            metrics["ctx_kl"] = alignment.item()
            metrics["ctx_recon"] = recon_loss.item()
            metrics["ctx_beta"] = beta

        return loss, metrics

    def _monolithic_forward(
        self, own_state, future, mu_prior, mu_residual,
        log_sigma_prior, log_sigma_posterior,
    ):
        """Baseline and deterministic variants: one latent, one decoder."""
        mu = mu_prior + mu_residual
        if self.deterministic:
            latent = mu
            alignment = _residual_l2(mu_residual).mean()
        else:
            latent = mu + torch.exp(log_sigma_posterior) * torch.randn_like(mu)
            alignment = _residual_kl(
                log_sigma_prior, mu_residual, log_sigma_posterior
            ).mean()

        squared_error = (self.ma_decoder(own_state, latent) - future) ** 2
        recon_loss = _trajectory_mse(squared_error)

        own_error = squared_error[:, self.agent_id : self.agent_id + 1]
        team_error = squared_error[:, self.other_ids] if self.other_ids else None
        return recon_loss, own_error, team_error, alignment, None, mu_prior

    def _factorized_forward(
        self, own_state, own_target, team_target, mu_prior, mu_residual,
        log_sigma_prior, log_sigma_posterior, beta,
    ):
        """Study FG: [z_self ; z_team] with separate decoders.

        Only the teammate half receives the privileged residual. The self half is trained
        purely by reconstructing this agent's own future, which is what confines the two
        jobs to separate capacity rather than letting them compete for one vector.
        """
        z_self, z_team = self.split_latent(mu_prior)

        if self.deterministic:
            z_team_final = z_team + mu_residual
            alignment = _residual_l2(mu_residual).mean()
        else:
            # The scale heads emit full-width vectors; the residual only covers the team
            # half, so the KL is evaluated on the matching slice of the prior's scale.
            _, log_sigma_prior_team = self.split_latent(log_sigma_prior)
            z_team_final = (
                z_team + mu_residual
                + torch.exp(log_sigma_posterior) * torch.randn_like(mu_residual)
            )
            alignment = _residual_kl(
                log_sigma_prior_team, mu_residual, log_sigma_posterior
            ).mean()

        own_error = (self.decoder_self(own_state, z_self) - own_target) ** 2
        team_error = (self.decoder_team(own_state, z_team_final) - team_target) ** 2
        recon_loss = _trajectory_mse(own_error) + _trajectory_mse(team_error)

        return recon_loss, own_error, team_error, alignment, z_self, z_team

    def _probe_outputs(self, own_state, team_target, z_self, z_team):
        """Optional read-out probes. Return (extra_loss_or_None, metrics).

        Both target teammates' futures and both receive `own_state`, the same conditioning
        the real decoders get, so the numbers are directly comparable to
        `ctx_recon_others`.

        prior_probe reads the *prior's* teammate latent, before the privileged residual is
        added. It answers the question Stage 1 otherwise cannot: the real reconstruction
        always runs on `z_prior + residual`, but deployment has no residual, so the gate
        can pass while the prior alone is useless. Reusing the trained decoder for this
        would evaluate it off-manifold, hence a separately trained read-out.

        leak_probe reads `z_self` and also targets teammates. It should do *badly*. Read it
        relative to prior_probe rather than in absolute terms: `own_state` alone correlates
        with teammates' futures in a coordinated task, so a low absolute error is not by
        itself evidence of leakage. If leak approaches prior_probe, the halves have merged.

        Both stop-gradient by default, making them pure measurement. Clearing
        `prior_probe_stop_grad` turns prior_probe into a prior-only reconstruction loss
        that actively forces the prior to be sufficient.
        """
        metrics = {}
        if team_target is None:
            return None, metrics

        total = None

        if self.prior_probe is not None:
            source = z_team.detach() if self.prior_probe_stop_grad else z_team
            error = (self.prior_probe(own_state, source) - team_target) ** 2
            probe_loss = _trajectory_mse(error)
            total = probe_loss if total is None else total + probe_loss
            metrics["ctx_probe_recon_others"] = error.mean().item()

        if self.leak_probe is not None and z_self is not None:
            source = z_self.detach() if self.leak_probe_stop_grad else z_self
            error = (self.leak_probe(own_state, source) - team_target) ** 2
            leak_loss = _trajectory_mse(error)
            total = leak_loss if total is None else total + leak_loss
            metrics["ctx_leak_recon_others"] = error.mean().item()

        return total, metrics

    @staticmethod
    def _probe_comparisons(metrics):
        """Derived numbers that only mean something once both halves are in hand.

        ctx_prior_gap  = prior-only error minus combined error. This is the headline
                         diagnostic: near zero means the prior carries the job on its own,
                         which is what deployment needs. Large means the privileged
                         residual is doing the work and the Stage 1 gate is flattering us.

        ctx_leak_ratio = leak error over prior-only error. Near 1 means z_self predicts
                         teammates about as well as z_team does, i.e. the halves merged
                         and the factorization is cosmetic. Well above 1 means it held.
        """
        derived = {}
        combined = metrics.get("ctx_recon_others")
        prior_only = metrics.get("ctx_probe_recon_others")
        leak = metrics.get("ctx_leak_recon_others")

        if combined is not None and prior_only is not None:
            derived["ctx_prior_gap"] = prior_only - combined
        if leak is not None and prior_only is not None and prior_only > 1e-12:
            derived["ctx_leak_ratio"] = leak / prior_only
        return derived

    def _diagnostics(self, own_error, team_error, team_target, mu_residual,
                     log_sigma_prior, log_sigma_posterior):
        """Split reconstruction error into own-agent and teammate components.

        `recon_others` is the metric that matters. The agent's own current state already
        explains much of its own future, so only the teammate term proves that z carries
        coordination information. It is compared against a batch-mean predictor: if
        `recon_others` does not beat `recon_others_baseline`, Stage 2 will not work.
        """
        per_element_own = own_error.mean().item()

        if team_error is not None:
            per_element_others = team_error.mean().item()
            baseline = ((team_target - team_target.mean(dim=0, keepdim=True)) ** 2).mean()
            baseline_others = baseline.item()
        else:
            per_element_others = 0.0
            baseline_others = 0.0

        # The deterministic variant has no scale parameters; report 1.0 (their implied
        # value) so the key set stays identical and the epoch-averaging code is shared.
        sigma_prior = 1.0 if log_sigma_prior is None else torch.exp(log_sigma_prior).mean().item()
        sigma_post = (
            1.0 if log_sigma_posterior is None else torch.exp(log_sigma_posterior).mean().item()
        )

        return {
            "ctx_recon_own": per_element_own,
            "ctx_recon_others": per_element_others,
            "ctx_recon_others_baseline": baseline_others,
            "ctx_mu_residual_norm": mu_residual.norm(dim=-1).mean().item(),
            "ctx_sigma_prior": sigma_prior,
            "ctx_sigma_posterior": sigma_post,
        }
