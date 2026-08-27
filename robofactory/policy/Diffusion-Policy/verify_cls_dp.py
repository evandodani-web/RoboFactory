"""Standalone correctness checks for the CLS-DP modules.

Runs on CPU in a few seconds and needs no data, no GPU and no SigLIP download. Run this
after touching anything under model/cls/, model/diffusion/cls_conditional_unet1d.py or
policy/contextualizer.py.

    python policy/Diffusion-Policy/verify_cls_dp.py

What it protects:

  * The residual-KL identity. This is the load-bearing property of the method: because the
    posterior mean is mu_rho + mu_E against a prior mean of mu_rho, the prior mean must
    cancel out of the KL entirely, leaving mu_rho to be trained by reconstruction alone.
    The test draws mu_rho at a large scale and asserts the closed form still matches
    torch's generic Gaussian KL exactly.
  * Zero-init behaviour: the CVAE starts with KL == 0 and the action-expert starts as
    vanilla Diffusion Policy (every cross-attention block is an identity).
  * Tensor shapes through every module, and that z actually influences the U-Net output
    once the zero-initialised output projections have moved.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from diffusion_policy.model.cls.cross_attention import CrossAttention1d, LatentTokenizer
from diffusion_policy.model.cls.ma_kinematics import (
    MAKinematicsDecoder,
    MAKinematicsEncoder,
)
from diffusion_policy.model.cls.prior_net import PriorNet
from diffusion_policy.model.diffusion.cls_conditional_unet1d import CLSConditionalUnet1D
from diffusion_policy.policy.contextualizer import _residual_kl

BATCH, N_AGENTS, N_FUTURE = 4, 3, 8
STATE_DIM, LATENT_DIM, FEATURE_DIM = 8, 256, 768
N_IMAGE_TOKENS, N_TEXT_TOKENS = 17, 64


def n_params(module):
    return sum(p.numel() for p in module.parameters())


def check(label, condition):
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    if not condition:
        raise AssertionError(label)


def main():
    torch.manual_seed(0)

    print("\n[1] PriorNet")
    prior = PriorNet(
        feature_dim=FEATURE_DIM,
        latent_dim=LATENT_DIM,
        d_model=768,
        n_layers=2,
        n_heads=8,
        dim_feedforward=2048,
    )
    image_tokens = torch.randn(BATCH, N_IMAGE_TOKENS, FEATURE_DIM)
    text_tokens = torch.randn(BATCH, N_TEXT_TOKENS, FEATURE_DIM)
    text_mask = torch.ones(BATCH, N_TEXT_TOKENS)
    text_mask[:, 40:] = 0

    mu_prior, log_sigma_prior, info = prior(
        image_tokens, text_tokens, text_mask, return_attn=True
    )
    print(f"       params {n_params(prior) / 1e6:.1f}M")
    check("mu shape", tuple(mu_prior.shape) == (BATCH, LATENT_DIM))
    attn_sum = info["text_attention"] + info["image_attention"]
    print(
        f"       Fig. 4 split: text {info['text_attention']:.3f} / "
        f"image {info['image_attention']:.3f}"
    )
    check("attention split normalises to 1", abs(attn_sum.item() - 1.0) < 1e-4)
    check("sigma_rho starts at 1", torch.allclose(log_sigma_prior, torch.zeros_like(log_sigma_prior)))

    print("\n[2] Multi-agent kinematics branch")
    encoder = MAKinematicsEncoder(
        state_dim=STATE_DIM,
        n_agents=N_AGENTS,
        n_future_states=N_FUTURE,
        latent_dim=LATENT_DIM,
    )
    decoder = MAKinematicsDecoder(
        state_dim=STATE_DIM,
        n_agents=N_AGENTS,
        n_future_states=N_FUTURE,
        latent_dim=LATENT_DIM,
    )
    future_states = torch.randn(BATCH, N_AGENTS, N_FUTURE, STATE_DIM)
    mu_residual, log_sigma_posterior = encoder(future_states)
    reconstruction = decoder(torch.randn(BATCH, STATE_DIM), mu_prior + mu_residual)

    budget = (n_params(encoder) + n_params(decoder)) / 1e6
    print(f"       encoder {n_params(encoder) / 1e6:.2f}M + decoder {n_params(decoder) / 1e6:.2f}M = {budget:.2f}M")
    print("       (Table III of the paper implies roughly 2.3M for this pair)")
    check("residual shape", tuple(mu_residual.shape) == (BATCH, LATENT_DIM))
    check(
        "reconstruction shape",
        tuple(reconstruction.shape) == (BATCH, N_AGENTS, N_FUTURE, STATE_DIM),
    )
    check("mu_E starts at 0", torch.allclose(mu_residual, torch.zeros_like(mu_residual)))
    check(
        "sigma_E starts at 1",
        torch.allclose(log_sigma_posterior, torch.zeros_like(log_sigma_posterior)),
    )

    print("\n[3] Residual KL")
    kl_at_init = _residual_kl(log_sigma_prior, mu_residual, log_sigma_posterior)
    check("KL is exactly 0 at init", kl_at_init.abs().max().item() < 1e-6)

    log_sigma_p = torch.randn(BATCH, LATENT_DIM) * 0.3
    mu_res = torch.randn(BATCH, LATENT_DIM) * 0.7
    log_sigma_q = torch.randn(BATCH, LATENT_DIM) * 0.3
    ours = _residual_kl(log_sigma_p, mu_res, log_sigma_q)

    # mu_rho is drawn at a deliberately large scale: if the closed form were wrong, the
    # prior mean would leak into the result and this comparison would blow up.
    mu_rho = torch.randn(BATCH, LATENT_DIM) * 5.0
    reference = torch.distributions.kl_divergence(
        torch.distributions.Normal(mu_rho + mu_res, log_sigma_q.exp()),
        torch.distributions.Normal(mu_rho, log_sigma_p.exp()),
    ).sum(-1)
    error = (ours - reference).abs().max().item()
    print(f"       max abs error vs torch KL: {error:.3e} (mu_rho drawn at scale 5.0)")
    check("closed form matches generic Gaussian KL", error < 1e-4)

    print("\n[4] Latent injection")
    tokenizer = LatentTokenizer(latent_dim=LATENT_DIM, token_dim=256, n_tokens=4)
    tokens = tokenizer(torch.randn(2, LATENT_DIM))
    check("tokenizer shape", tuple(tokens.shape) == (2, 4, 256))

    attn = CrossAttention1d(channels=256, context_dim=256, n_heads=4)
    features = torch.randn(2, 256, 8)
    context = torch.randn(2, 4, 256)
    check("cross-attention is identity at init", torch.allclose(attn(features, context), features))

    print("\n[5] CLSConditionalUnet1D")
    unet = CLSConditionalUnet1D(
        input_dim=STATE_DIM,
        latent_dim=LATENT_DIM,
        global_cond_dim=137 * 3,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        cond_predict_scale=True,
        diffusion_step_embed_dim=128,
    )
    sample = torch.randn(2, 8, STATE_DIM)
    global_cond = torch.randn(2, 137 * 3)
    timesteps = torch.tensor([3, 7])
    z_a, z_b = torch.randn(2, LATENT_DIM), torch.randn(2, LATENT_DIM)

    without_z = unet(sample, timesteps, global_cond=global_cond, cond_latent=None)
    with_z = unet(sample, timesteps, global_cond=global_cond, cond_latent=z_a)
    print(f"       params {n_params(unet) / 1e6:.1f}M")
    check("output shape", tuple(with_z.shape) == tuple(sample.shape))
    check("starts as vanilla DP", torch.allclose(without_z, with_z, atol=1e-6))
    check(
        "injection sites = down levels + bottleneck",
        len(unet.down_cross_attn) == 3 and len(unet.mid_cross_attn) == 2,
    )

    for block in list(unet.down_cross_attn) + list(unet.mid_cross_attn):
        torch.nn.init.normal_(block.to_out.weight, std=0.02)
    delta = (
        unet(sample, timesteps, global_cond=global_cond, cond_latent=z_a)
        - unet(sample, timesteps, global_cond=global_cond, cond_latent=z_b)
    ).abs().max().item()
    print(f"       once trained, changing z moves the output by {delta:.4f}")
    check("z influences the output", delta > 0)

    print("\nALL CLS-DP MODULE CHECKS PASSED\n")


if __name__ == "__main__":
    main()
