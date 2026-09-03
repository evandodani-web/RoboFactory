"""CLS-DP Stage 2, flow-matching action expert (Study FM).

Same U-Net, same ResNet-18 observation encoder, same FiLM conditioning, same z
cross-attention into the down and mid blocks, same frozen contextualizer prior, same
normalizer, same 6-step executed slice. The only thing that changes is the transport
between noise and actions: a straight-line ODE integrated in a handful of Euler steps
instead of the paper's 100-step DDPM reverse chain.

This is a subclass rather than a config branch inside the DDPM policy on purpose. Study B
reproduced the paper at 61% on LiftBarrier with that code path, so it is left byte-identical
and the only edits to the parent are additive (an optional noise_scheduler, and capturing
the denoiser kwargs so this class can rebuild at a different input width).

Two methods are overridden, `compute_loss` and `conditional_sample`. `predict_action` is
inherited unchanged: this class's `conditional_sample` always returns a
(B, horizon, action_dim) trajectory, even when the flow itself runs in a learned action
latent, so the inherited unnormalize-and-slice logic keeps working.

See docs/CLS-DP-variant-flow-matching.md for the derivation behind every flag.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.cls.flow_matching import RectifiedFlowTransport
from diffusion_policy.model.diffusion.cls_conditional_unet1d import CLSConditionalUnet1D
from diffusion_policy.policy.cls_diffusion_unet_image_policy import (
    CLSDiffusionUnetImagePolicy,
)

TC_SPACES = ("velocity", "clean")


def horizon_delta(x: torch.Tensor) -> torch.Tensor:
    """Finite difference along the horizon axis: (B, T, C) -> (B, T-1, C).

    In RoboFactory `state == action` -- both are the same 8-D commanded joint vector -- so
    the difference between consecutive actions *is* the commanded joint velocity. Scoring
    it is what the temporal-consistency term does.
    """
    return x[:, 1:] - x[:, :-1]


class CLSFlowMatchingUnetImagePolicy(CLSDiffusionUnetImagePolicy):
    def __init__(
        self,
        *,
        num_inference_steps: int = 4,
        solver: str = "euler",
        sigma_dist: str = "uniform",
        sigma_dist_loc: float = 0.0,
        sigma_dist_scale: float = 1.0,
        shift: float = 1.0,
        timestep_scale: float = 1000.0,
        temporal_consistency_weight: float = 0.0,
        tc_space: str = "velocity",
        action_autoencoder: Optional[nn.Module] = None,
        action_ae_ckpt: Optional[str] = None,
        **kwargs,
    ):
        if num_inference_steps is None:
            raise ValueError("the flow-matching expert needs num_inference_steps")
        if tc_space not in TC_SPACES:
            raise ValueError(f"tc_space must be one of {TC_SPACES}, got {tc_space!r}")
        if temporal_consistency_weight < 0:
            raise ValueError(
                f"temporal_consistency_weight must be >= 0, got "
                f"{temporal_consistency_weight}"
            )

        kwargs.setdefault("noise_scheduler", None)
        super().__init__(num_inference_steps=num_inference_steps, **kwargs)

        self.transport = RectifiedFlowTransport(
            sigma_dist=sigma_dist,
            sigma_dist_loc=sigma_dist_loc,
            sigma_dist_scale=sigma_dist_scale,
            shift=shift,
            timestep_scale=timestep_scale,
            solver=solver,
        )
        self.temporal_consistency_weight = temporal_consistency_weight
        self.tc_space = tc_space
        # Populated on every compute_loss so the workspace can log the split.
        self.last_loss_components: Dict[str, float] = {}

        # CLS-DP is global-conditioning only, so the inpainting mask generator the parent
        # builds is vacuous. Assert that rather than silently carrying dead masking code:
        # if someone gives the mask generator a non-zero obs_dim, this variant would start
        # dropping part of the flow loss without saying so.
        probe_mask = self.mask_generator((1, self.horizon, self.action_dim))
        if bool(probe_mask.any()):
            raise ValueError(
                "the flow-matching expert assumes a vacuous condition mask; got a mask "
                "generator that conditions on part of the trajectory."
            )

        self.action_autoencoder = action_autoencoder
        if action_autoencoder is not None:
            if action_ae_ckpt:
                payload = torch.load(open(action_ae_ckpt, "rb"), map_location="cpu")
                state_dicts = payload.get("state_dicts", {})
                ae_state = state_dicts.get("action_autoencoder")
                if ae_state is None:
                    raise KeyError(
                        f"autoencoder checkpoint {action_ae_ckpt} missing state_dicts['action_autoencoder']"
                    )
                action_autoencoder.load_state_dict(ae_state)

            # Latent action space: the flow runs on the autoencoder's tokens, so the
            # denoiser is narrower than the raw 8-D joint chunk. Rebuilding is why the
            # parent captures its construction kwargs. In the default raw path nothing is
            # rebuilt and this branch never runs.
            self.action_autoencoder.requires_grad_(False)
            self.action_autoencoder.eval()
            denoiser_kwargs = dict(self._denoiser_kwargs)

            down_dims = denoiser_kwargs.get("down_dims", None) or ()
            if len(down_dims) >= 1:
                down_factor = 2 ** (len(down_dims) - 1)
                if (self.action_autoencoder.n_tokens % down_factor) != 0:
                    raise ValueError(
                        "n_tokens must be divisible by 2**(len(down_dims)-1) to survive "
                        f"U-Net downsampling; got n_tokens={self.action_autoencoder.n_tokens} "
                        f"down_factor={down_factor}"
                    )

            denoiser_kwargs["input_dim"] = action_autoencoder.latent_dim
            self.model = CLSConditionalUnet1D(**denoiser_kwargs)
            self.flow_shape = (
                action_autoencoder.n_tokens,
                action_autoencoder.latent_dim,
            )
        else:
            self.flow_shape = (self.horizon, self.action_dim)

    @property
    def uses_latent_action_space(self) -> bool:
        return self.action_autoencoder is not None

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen alongside the prior, for the same reason: its norm statistics must match
        # what Stage 2a produced or the decoder drifts away from the manifold the flow was
        # trained against.
        if self.action_autoencoder is not None:
            self.action_autoencoder.eval()
        return self

    # ------------------------------------------------------------------ helpers

    def _denoise(self, sample, t_model, global_cond, cond_latent):
        return self.model(
            sample,
            t_model,
            local_cond=None,
            global_cond=global_cond,
            cond_latent=cond_latent,
        )

    def _encode_actions(self, nactions):
        """Normalized joint chunk -> whatever space the flow runs in."""
        if self.action_autoencoder is None:
            return nactions
        with torch.no_grad():
            return self.action_autoencoder.encode(nactions).detach()

    def _decode_actions(self, flow_sample):
        """Inverse of _encode_actions, back to a (B, horizon, action_dim) chunk."""
        if self.action_autoencoder is None:
            return flow_sample
        return self.action_autoencoder.decode(flow_sample)

    # ---------------------------------------------------------------- inference

    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        cond_latent=None,
        generator=None,
        **kwargs,
    ):
        """Integrate the flow ODE and return a (B, horizon, action_dim) trajectory.

        `condition_data` supplies only batch size, device and dtype; CLS-DP never inpaints,
        and the constructor already asserted the condition mask is vacuous. The signature
        matches the parent so the inherited `predict_action` needs no changes.
        """
        batch_size = condition_data.shape[0]

        def model_fn(sample, t_model):
            return self._denoise(sample, t_model, global_cond, cond_latent)

        flow_sample = self.transport.sample(
            model_fn,
            shape=(batch_size, *self.flow_shape),
            num_steps=self.num_inference_steps,
            device=condition_data.device,
            dtype=condition_data.dtype,
            generator=generator,
        )
        return self._decode_actions(flow_sample)

    # ----------------------------------------------------------------- training

    def compute_loss(self, batch, **kwargs):
        assert "valid_mask" not in batch

        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]

        _, prior_inputs = self.split_prior_inputs(batch)
        cond_latent = self.resolve_latent(prior_inputs)

        this_nobs = dict_apply(
            nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(batch_size, -1)

        # x1 is the target end of the transport: the clean chunk, or its frozen encoding.
        x1 = self._encode_actions(nactions)
        noise = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype)
        sigma = self.transport.sample_sigma(
            batch_size, device=x1.device, dtype=x1.dtype
        )

        x_sigma = self.transport.interpolate(x1, noise, sigma)
        velocity_target = self.transport.velocity_target(x1, noise)

        velocity_pred = self._denoise(
            x_sigma,
            self.transport.to_model_timestep(sigma),
            global_cond,
            cond_latent,
        )

        flow_loss = F.mse_loss(velocity_pred, velocity_target)
        loss = flow_loss

        tc_loss = None
        if self.temporal_consistency_weight > 0:
            if self.tc_space == "velocity":
                # Same space as the main loss, so both terms carry the same sigma^2
                # relation to clean-action error and the weight needs no schedule.
                tc_loss = F.mse_loss(
                    horizon_delta(velocity_pred), horizon_delta(velocity_target)
                )
            else:
                # Equivalent up to a bounded sigma^2 factor, which is the whole reason
                # this is tractable here: the DDPM analogue is (1-alphabar)/alphabar,
                # which is unbounded as alphabar -> 0.
                x1_pred = self.transport.implied_x1(x_sigma, sigma, velocity_pred)
                tc_loss = F.mse_loss(horizon_delta(x1_pred), horizon_delta(x1))
            loss = loss + self.temporal_consistency_weight * tc_loss

        self.last_loss_components = {
            "flow": float(flow_loss.detach()),
            "temporal_consistency": (
                float(tc_loss.detach()) if tc_loss is not None else 0.0
            ),
            "total": float(loss.detach()),
        }

        if "output_pred" in kwargs:
            return loss, velocity_pred
        return loss
