"""CLS-DP Stage 2: the action-expert.

A decentralized diffusion policy conditioned on the collaborative latent z_t^i. Structurally
this is the repo's DiffusionUnetImagePolicy with one addition: z enters the U-Net's
downsampling and bottleneck stages through cross-attention, while (O_t^i, S_t^i) continue to
enter through FiLM exactly as before.

Two details that matter for faithfulness:

  * z is sampled from the *prior* during Stage 2 training, not the posterior. That is what
    keeps training consistent with deployment, where the posterior does not exist.
  * The contextualizer is frozen and gradients never flow into it (the sg(.) in Eq. 11).

Action slicing follows this repo's convention rather than the paper's literal
A_t := a_{t:t+H-1}. With horizon=8 and n_obs_steps=3, window index To-1=2 is time t, so
slicing [2:10] on a length-8 tensor yields the 6 executed actions a_t..a_{t+5}. That matches
Table I's "Execution steps: 6" and keeps the `w/o CLS` ablation directly comparable to the
repo's DP. See docs/CLS-DP-implementation-notes.md section 1.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.cls_conditional_unet1d import CLSConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

# Keys that address the contextualizer rather than the observation encoder. They must be
# split out before the observation normalizer runs, which has no parameters for them.
PRIOR_KEYS = (
    "prior_image_tokens",
    "prior_text_tokens",
    "prior_text_mask",
    "cls_latent",
)


class CLSDiffusionUnetImagePolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        # May be None only for subclasses that do not denoise through a DDPM chain (see
        # CLSFlowMatchingUnetImagePolicy); they must then pass num_inference_steps.
        noise_scheduler: Optional[DDPMScheduler],
        obs_encoder: MultiImageObsEncoder,
        prior_net: nn.Module,
        horizon,
        n_action_steps,
        n_obs_steps,
        latent_dim=256,
        latent_sample=True,
        num_inference_steps=None,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        n_cond_tokens=4,
        cond_token_dim=256,
        cross_attn_heads=4,
        **kwargs,
    ):
        super().__init__()
        if not obs_as_global_cond:
            raise NotImplementedError(
                "CLS-DP only supports obs_as_global_cond=True; the inpainting variant "
                "would need a second conditioning route for z."
            )

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()[0]

        # Captured rather than passed inline so subclasses can rebuild the denoiser at a
        # different input width without having to redeclare this whole signature. The
        # values and the resulting module are exactly as before.
        self._denoiser_kwargs = dict(
            input_dim=action_dim,
            latent_dim=latent_dim,
            local_cond_dim=None,
            global_cond_dim=obs_feature_dim * n_obs_steps,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            n_cond_tokens=n_cond_tokens,
            cond_token_dim=cond_token_dim,
            cross_attn_heads=cross_attn_heads,
        )
        model = CLSConditionalUnet1D(**self._denoiser_kwargs)

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()

        # Frozen contextualizer prior. Kept as a submodule so Stage 2 checkpoints are
        # self-contained at eval time (SigLIP itself stays outside the checkpoint).
        self.prior_net = prior_net
        self.prior_net.requires_grad_(False)
        self.prior_net.eval()

        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.latent_dim = latent_dim
        self.latent_sample = latent_sample
        self.kwargs = kwargs

        if num_inference_steps is None:
            if noise_scheduler is None:
                raise ValueError(
                    "num_inference_steps must be given explicitly when noise_scheduler "
                    "is None; there is no training chain length to fall back on."
                )
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def train(self, mode: bool = True):
        super().train(mode)
        # The frozen prior must never leave eval mode, or its dropout/norm statistics
        # would drift away from what Stage 1 produced.
        self.prior_net.eval()
        return self

    # ------------------------------------------------------------------ latents

    @staticmethod
    def split_prior_inputs(obs_dict: Dict[str, torch.Tensor]):
        observation = {k: v for k, v in obs_dict.items() if k not in PRIOR_KEYS}
        prior_inputs = {k: obs_dict[k] for k in PRIOR_KEYS if k in obs_dict}
        return observation, prior_inputs

    def resolve_latent(self, prior_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return z, either passed in directly or sampled from the frozen prior."""
        if "cls_latent" in prior_inputs:
            return prior_inputs["cls_latent"].detach()

        if "prior_image_tokens" not in prior_inputs:
            raise KeyError(
                "CLS-DP needs either 'cls_latent' or "
                "('prior_image_tokens', 'prior_text_tokens') in the observation dict."
            )

        with torch.no_grad():
            mu, log_sigma, _ = self.prior_net(
                prior_inputs["prior_image_tokens"],
                prior_inputs["prior_text_tokens"],
                text_mask=prior_inputs.get("prior_text_mask"),
            )
            # log_sigma is None for a deterministic prior, which has no scale head to
            # sample from; latent_sample is then moot.
            if log_sigma is None or not self.latent_sample:
                latent = mu
            else:
                latent = mu + torch.exp(log_sigma) * torch.randn_like(mu)
        return latent.detach()

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
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]

            model_output = model(
                trajectory,
                t,
                local_cond=local_cond,
                global_cond=global_cond,
                cond_latent=cond_latent,
            )

            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict

        observation, prior_inputs = self.split_prior_inputs(obs_dict)
        cond_latent = self.resolve_latent(prior_inputs)

        nobs = self.normalizer.normalize(observation)
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        n_obs = self.n_obs_steps

        this_nobs = dict_apply(
            nobs, lambda x: x[:, :n_obs, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(batch_size, -1)

        cond_data = torch.zeros(
            size=(batch_size, self.horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=None,
            global_cond=global_cond,
            cond_latent=cond_latent,
            **self.kwargs,
        )

        action_pred = self.normalizer["action"].unnormalize(
            nsample[..., : self.action_dim]
        )

        start = n_obs - 1
        end = start + self.n_action_steps
        return {
            "action": action_pred[:, start:end],
            "action_pred": action_pred,
            "cls_latent": cond_latent,
        }

    # ----------------------------------------------------------------- training

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch, **kwargs):
        assert "valid_mask" not in batch

        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]

        _, prior_inputs = self.split_prior_inputs(batch)
        cond_latent = self.resolve_latent(prior_inputs)

        trajectory = nactions
        cond_data = trajectory

        this_nobs = dict_apply(
            nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(batch_size, -1)

        condition_mask = self.mask_generator(trajectory.shape)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        pred = self.model(
            noisy_trajectory,
            timesteps,
            local_cond=None,
            global_cond=global_cond,
            cond_latent=cond_latent,
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()

        if "output_pred" in kwargs:
            return loss, pred
        return loss
