"""ConditionalUnet1D variant with collaborative-latent cross-attention.

This is a separate file rather than a modification of conditional_unet1d.py so that
vanilla Diffusion Policy -- and therefore the `Ours w/o CLS` ablation -- stays byte
identical to the original.

Two conditioning routes, matching the paper:

  * (O_t^i, S_t^i)  ->  FiLM, via the existing global_cond path
  * z_t^i           ->  cross-attention, into the downsampling and bottleneck stages only

With the default down_dims of [256, 512, 1024] that is 3 down levels plus 2 mid blocks,
i.e. 5 injection sites. The upsampling path is deliberately left alone.
"""

import logging
from typing import Union

import einops
import torch
import torch.nn as nn

from diffusion_policy.model.cls.cross_attention import CrossAttention1d, LatentTokenizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalResidualBlock1D
from diffusion_policy.model.diffusion.conv1d_components import (
    Conv1dBlock,
    Downsample1d,
    Upsample1d,
)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)


class CLSConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim=256,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        n_cond_tokens=4,
        cond_token_dim=256,
        cross_attn_heads=4,
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList(
                [
                    ConditionalResidualBlock1D(
                        dim_in,
                        dim_out,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                    ),
                    ConditionalResidualBlock1D(
                        dim_in,
                        dim_out,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                    ),
                ]
            )

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        # Collaborative latent injection: down levels + bottleneck only.
        self.latent_tokenizer = LatentTokenizer(
            latent_dim=latent_dim, token_dim=cond_token_dim, n_tokens=n_cond_tokens
        )
        self.down_cross_attn = nn.ModuleList(
            [
                CrossAttention1d(
                    channels=dim_out,
                    context_dim=cond_token_dim,
                    n_heads=cross_attn_heads,
                    n_groups=n_groups,
                )
                for _, dim_out in in_out
            ]
        )
        self.mid_cross_attn = nn.ModuleList(
            [
                CrossAttention1d(
                    channels=mid_dim,
                    context_dim=cond_token_dim,
                    n_heads=cross_attn_heads,
                    n_groups=n_groups,
                )
                for _ in self.mid_modules
            ]
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        local_cond=None,
        global_cond=None,
        cond_latent=None,
        **kwargs,
    ):
        """
        sample: (B, T, input_dim)
        timestep: (B,) or int
        global_cond: (B, global_cond_dim)
        cond_latent: (B, latent_dim) collaborative latent z_t^i
        output: (B, T, input_dim)
        """
        sample = einops.rearrange(sample, "b h t -> b t h")

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        cond_tokens = None
        if cond_latent is not None:
            cond_tokens = self.latent_tokenizer(cond_latent)

        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, "b h t -> b t h")
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            if cond_tokens is not None:
                x = self.down_cross_attn[idx](x, cond_tokens)
            h.append(x)
            x = downsample(x)

        for idx, mid_module in enumerate(self.mid_modules):
            x = mid_module(x, global_feature)
            if cond_tokens is not None:
                x = self.mid_cross_attn[idx](x, cond_tokens)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, "b t h -> b h t")
        return x
