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

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.cls.cross_attention import CrossAttention1d, LatentTokenizer
from diffusion_policy.model.cls.ma_kinematics import (
    MAKinematicsDecoder,
    MAKinematicsEncoder,
)
from diffusion_policy.model.cls.prior_net import PriorNet
from diffusion_policy.model.diffusion.cls_conditional_unet1d import CLSConditionalUnet1D
from diffusion_policy.policy.contextualizer import _residual_kl, _residual_l2

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

    print("\n[6] Deterministic variant")
    det_prior = PriorNet(
        feature_dim=FEATURE_DIM, latent_dim=LATENT_DIM, d_model=128,
        n_layers=1, n_heads=4, dim_feedforward=256, deterministic=True,
    )
    det_encoder = MAKinematicsEncoder(
        state_dim=STATE_DIM, n_agents=N_AGENTS, n_future_states=N_FUTURE,
        latent_dim=LATENT_DIM, d_model=64, n_layers=1, n_heads=4,
        dim_feedforward=128, deterministic=True,
    )
    det_mu, det_log_sigma, _ = det_prior(image_tokens, text_tokens, text_mask)
    det_residual, det_post_log_sigma = det_encoder(future_states)

    check("prior emits z with no scale head", det_log_sigma is None)
    check("posterior emits a residual with no scale head", det_post_log_sigma is None)
    check("no scale parameters exist at all",
          det_prior.to_log_sigma is None and det_encoder.to_log_sigma is None)
    check("latent shape unchanged", tuple(det_mu.shape) == (BATCH, LATENT_DIM))

    # The deterministic alignment term is the unit-variance limit of the KL, not a
    # different objective. Verify that against the stochastic implementation directly.
    probe = torch.randn(BATCH, LATENT_DIM) * 0.8
    unit = torch.zeros(BATCH, LATENT_DIM)  # log sigma = 0  ->  sigma = 1
    gap = (_residual_kl(unit, probe, unit) - _residual_l2(probe)).abs().max().item()
    print(f"       KL at sigma=1 vs L2 alignment: max abs diff {gap:.3e}")
    check("deterministic loss == KL with unit variances", gap < 1e-5)

    # Mirrors the stochastic case, where zero-init makes the KL start at exactly 0: the
    # residual head is zero-initialised, so the alignment term also starts at its minimum
    # and produces no gradient anywhere on step 0.
    check("alignment starts at exactly 0", _residual_l2(det_residual).abs().max().item() == 0.0)

    # Once the residual moves off zero, the distillation geometry must still hold: the
    # alignment depends only on the residual, so the prior receives no gradient from it.
    det_prior.zero_grad()
    det_encoder.zero_grad()
    torch.nn.init.normal_(det_encoder.to_mu_residual.weight, std=0.05)
    _residual_l2(det_encoder(future_states)[0]).mean().backward()
    prior_grads = [p.grad for p in det_prior.parameters() if p.grad is not None]
    residual_grad = det_encoder.to_mu_residual.weight.grad.abs().max().item()
    print(f"       prior tensors with gradient from alignment: {len(prior_grads)}"
          f"   residual head grad: {residual_grad:.3e}")
    check("alignment gives the prior no gradient", len(prior_grads) == 0)
    check("alignment trains the residual head", residual_grad > 0)

    print("\n[7] Factorized variant")
    from diffusion_policy.policy.contextualizer import Contextualizer

    SELF_DIM = 96
    TEAM_DIM = LATENT_DIM - SELF_DIM
    N_OTHERS = N_AGENTS - 1

    def make_decoder(n_agents, latent):
        return MAKinematicsDecoder(
            state_dim=STATE_DIM, n_agents=n_agents, n_future_states=N_FUTURE,
            latent_dim=latent, d_model=64, n_layers=1, n_heads=4, dim_feedforward=128,
        )

    ctx = Contextualizer(
        prior_net=PriorNet(feature_dim=FEATURE_DIM, latent_dim=LATENT_DIM, d_model=128,
                           n_layers=1, n_heads=4, dim_feedforward=256, deterministic=True),
        ma_encoder=MAKinematicsEncoder(
            state_dim=STATE_DIM, n_agents=N_AGENTS, n_future_states=N_FUTURE,
            latent_dim=TEAM_DIM, d_model=64, n_layers=1, n_heads=4,
            dim_feedforward=128, deterministic=True),
        decoder_self=make_decoder(1, SELF_DIM),
        decoder_team=make_decoder(N_OTHERS, TEAM_DIM),
        prior_probe=make_decoder(N_OTHERS, TEAM_DIM),
        leak_probe=make_decoder(N_OTHERS, SELF_DIM),
        agent_id=1, n_agents=N_AGENTS, latent_dim=LATENT_DIM,
        deterministic=True, factorize=True, self_dim=SELF_DIM,
    )

    full = torch.randn(BATCH, LATENT_DIM)
    z_self, z_team = ctx.split_latent(full)
    check("split widths", tuple(z_self.shape) == (BATCH, SELF_DIM)
          and tuple(z_team.shape) == (BATCH, TEAM_DIM))
    check("split is a partition of the full latent",
          torch.equal(torch.cat([z_self, z_team], dim=-1), full))
    print(f"       agent_id=1, teammate ids {ctx.other_ids}")
    check("agent_id excluded from the teammate set",
          ctx.other_ids == [j for j in range(N_AGENTS) if j != 1])

    normalizer = LinearNormalizer()
    normalizer.fit(data={"state": torch.randn(256, STATE_DIM)}, last_n_dims=1)
    ctx.set_normalizer(normalizer)

    batch = {
        "own_state": torch.randn(BATCH, STATE_DIM),
        "future_states": torch.randn(BATCH, N_AGENTS, N_FUTURE, STATE_DIM),
        "prior_image_tokens": image_tokens,
        "prior_text_tokens": text_tokens,
        "prior_text_mask": text_mask,
    }
    loss, metrics = ctx.compute_loss(batch, beta=0.1)
    check("factorized loss is finite", math.isfinite(loss.item()))
    for key in ("ctx_recon_own", "ctx_recon_others", "ctx_recon_others_baseline",
                "ctx_probe_recon_others", "ctx_leak_recon_others",
                "ctx_prior_gap", "ctx_leak_ratio"):
        check(f"metric {key} present and finite",
              key in metrics and math.isfinite(metrics[key]))

    # The decoders must be shape-correct for their own slice of the problem.
    own_state_n = ctx.normalizer["state"].normalize(batch["own_state"])
    check("decoder_self emits one agent",
          tuple(ctx.decoder_self(own_state_n, z_self).shape)
          == (BATCH, 1, N_FUTURE, STATE_DIM))
    check("decoder_team emits N-1 agents",
          tuple(ctx.decoder_team(own_state_n, z_team).shape)
          == (BATCH, N_OTHERS, N_FUTURE, STATE_DIM))

    print("\n[8] Probe stop-gradient")
    ctx.zero_grad()
    loss, _ = ctx.compute_loss(batch, beta=0.1)
    loss.backward()
    check("stop-grad probes still train their own parameters",
          ctx.prior_probe.to_state.weight.grad.abs().max().item() > 0
          and ctx.leak_probe.to_state.weight.grad.abs().max().item() > 0)

    # With both probes detached, the only gradient the prior sees comes from the two
    # reconstruction paths. Isolate the probe terms and confirm they contribute nothing.
    ctx.zero_grad()
    mu_prior, _, _ = ctx.prior_net(image_tokens, text_tokens, text_mask)
    zs, zt = ctx.split_latent(mu_prior)
    team_target = ctx.normalizer["state"].normalize(batch["future_states"])[:, ctx.other_ids]
    probe_only, _ = ctx._probe_outputs(own_state_n, team_target, zs, zt)
    probe_only.backward()
    prior_grads = [p.grad for p in ctx.prior_net.parameters()
                   if p.grad is not None and p.grad.abs().max() > 0]
    print(f"       prior tensors with gradient from the probe terms: {len(prior_grads)}")
    check("detached probes give the prior no gradient", len(prior_grads) == 0)

    # Clearing the flag is what turns the probe into an intervention.
    ctx.prior_probe_stop_grad = False
    ctx.zero_grad()
    mu_prior, _, _ = ctx.prior_net(image_tokens, text_tokens, text_mask)
    zs, zt = ctx.split_latent(mu_prior)
    probe_only, _ = ctx._probe_outputs(own_state_n, team_target, zs, zt)
    probe_only.backward()
    attached = [p.grad for p in ctx.prior_net.parameters()
                if p.grad is not None and p.grad.abs().max() > 0]
    print(f"       with stop-grad cleared: {len(attached)}")
    check("clearing stop-grad reaches the prior", len(attached) > 0)
    ctx.prior_probe_stop_grad = True

    print("\n[9] Flow-matching transport + sampler correctness")
    from diffusion_policy.model.cls.flow_matching import RectifiedFlowTransport
    from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
    from diffusion_policy.policy.cls_flow_matching_unet_image_policy import (
        CLSFlowMatchingUnetImagePolicy,
    )

    transport = RectifiedFlowTransport(timestep_scale=1000.0)

    # --- transport endpoint identities
    B_flow = 4
    H = 8
    D = STATE_DIM
    x1 = torch.randn(B_flow, H, D)
    noise = torch.randn_like(x1)
    sigmas = torch.tensor([1.0, 0.5, 0.25, 0.0])

    # forward interpolation
    x_sigma = transport.interpolate(x1, noise, sigmas)
    # sigma=1 -> noise; sigma=0 -> data
    check("sigma=1 gives noise", torch.allclose(x_sigma[0], noise[0]))
    check("sigma=0 gives data", torch.allclose(x_sigma[-1], x1[-1]))

    v = transport.velocity_target(x1, noise)
    x1_recon = transport.implied_x1(x_sigma, sigmas, v)
    check("implied_x1 inversion holds", torch.allclose(x1_recon, x1, atol=1e-6))

    # --- Euler integration matches diffusers' reference on the same sigma grid
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )

    ref = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    N_STEPS = 4
    ref.set_timesteps(N_STEPS)

    # Our sigma schedule must match diffusers exactly for the reference comparison.
    ours_sigmas = transport.sigma_schedule(
        N_STEPS, device="cpu", dtype=torch.float32
    )
    check(
        "sigma schedule matches diffusers",
        torch.allclose(ours_sigmas, ref.sigmas, atol=0, rtol=0),
    )

    model = lambda x, t_model: 0.123 * x  # deterministic velocity prediction
    torch.manual_seed(0)
    eps = torch.randn_like(x1)

    # ref loop
    x_ref = eps.clone()
    ref.set_timesteps(N_STEPS)
    for t in ref.timesteps:
        mo = model(x_ref, t)
        x_ref = ref.step(mo, t, x_ref).prev_sample

    # ours loop
    torch.manual_seed(0)
    x_ours = transport.sample(
        model_fn=lambda x, t_model: model(x, t_model),
        shape=x1.shape,
        num_steps=N_STEPS,
        device=x1.device,
        dtype=x1.dtype,
        noise=eps,
    )
    check("Euler sampling matches diffusers", torch.allclose(x_ours, x_ref, atol=1e-6))

    # --- model-call count in the full policy path
    class DummyObsEncoder(torch.nn.Module):
        """Returns fixed-width features and ignores the actual observation."""

        def __init__(self, out_dim: int):
            super().__init__()
            self.out_dim = out_dim

        def output_shape(self):
            return (self.out_dim,)

        def forward(self, obs_dict):
            # obs_dict comes from dict_apply() and has shape (B*n_obs_steps, ...)
            # Use any tensor's leading dim as the effective batch size.
            any_tensor = next(iter(obs_dict.values()))
            batch = any_tensor.shape[0]
            return torch.zeros(batch, self.out_dim, device=any_tensor.device, dtype=any_tensor.dtype)

    OBS_FEATURE_DIM = 16
    N_OBS_STEPS = 3
    # Reuse the module-level LATENT_DIM constant; do not shadow it.

    shape_meta = {
        "action": {"shape": [STATE_DIM]},
        "obs": {"head_cam": {"shape": [3, 16, 16], "type": "rgb"}, "agent_pos": {"shape": [STATE_DIM], "type": "low_dim"}},
    }

    prior_net = PriorNet(
        feature_dim=64, latent_dim=LATENT_DIM, d_model=64, n_layers=1, n_heads=4, dim_feedforward=128
    )
    obs_encoder = DummyObsEncoder(OBS_FEATURE_DIM)

    policy = CLSFlowMatchingUnetImagePolicy(
        shape_meta=shape_meta,
        obs_encoder=obs_encoder,
        prior_net=prior_net,
        horizon=H,
        n_action_steps=H,
        n_obs_steps=N_OBS_STEPS,
        latent_dim=LATENT_DIM,
        num_inference_steps=3,
        down_dims=[32, 64],
        diffusion_step_embed_dim=32,
        kernel_size=3,
        n_cond_tokens=2,
        cond_token_dim=32,
        cross_attn_heads=2,
        solver="euler",
        temporal_consistency_weight=0.0,
        tc_space="velocity",
    )

    # Minimal normalizer for `predict_action()`.
    normalizer = LinearNormalizer()
    normalizer.fit(
        {
            "action": torch.randn(64, STATE_DIM),
            "agent_pos": torch.randn(64, STATE_DIM),
            "head_cam": torch.rand(8, 3, 16, 16),
        }
    )
    policy.set_normalizer(normalizer)

    obs_dict = {
        "head_cam": torch.rand(BATCH, N_OBS_STEPS, 3, 16, 16),
        "agent_pos": torch.randn(BATCH, N_OBS_STEPS, STATE_DIM),
        # Provide cls_latent directly to avoid depending on the frozen prior's exact shapes.
        "cls_latent": torch.randn(BATCH, LATENT_DIM),
    }

    policy.eval()
    calls = {"n": 0}
    orig = policy.model.forward

    def counted(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    policy.model.forward = counted
    with torch.no_grad():
        out = policy.predict_action(obs_dict)

    check("flow Euler model calls == num_inference_steps", calls["n"] == 3)
    check("flow action_pred spans full horizon", tuple(out["action_pred"].shape) == (BATCH, H, STATE_DIM))
    check("flow executed action is 6 steps", tuple(out["action"].shape) == (BATCH, 6, STATE_DIM))

    # --- timestep-scale embedding sanity
    pos = SinusoidalPosEmb(dim=128)
    sig = torch.linspace(0, 1, 10).to(torch.float32)
    emb_raw = pos(sig)  # (10,128)
    emb_scaled = pos(sig * transport.timestep_scale)
    var_raw = emb_raw.var(dim=0)
    var_scaled = emb_scaled.var(dim=0)
    # The scaled schedule should activate more dimensions with non-trivial variance.
    threshold = var_raw.max().item() * 0.05 + 1e-7
    n_raw = int((var_raw > threshold).sum().item())
    n_scaled = int((var_scaled > threshold).sum().item())
    check("timestep_scale makes embedding more informative", n_scaled >= n_raw)

    # --- sigma^2 relation between velocity-space and clean-space delta errors
    # If v_pred = v_target + dv, then x1_pred = x1 - sigma*dv, and all linear deltas
    # must scale by sigma^2 in squared error.
    sigma = torch.tensor(0.3, dtype=torch.float32)
    dv = torch.randn_like(v)
    v_pred = v + dv
    x1_pred = transport.implied_x1(x_sigma, sigmas, v_pred)
    # Compare per-sigma sample by selecting the row whose sigma is 0.5 (index 1).
    idx = 1
    sigma_i = sigmas[idx]

    def _delta(x: torch.Tensor) -> torch.Tensor:
        # (B, H, D) -> (B, H-1, D)
        return x[:, 1:] - x[:, :-1]

    vel_delta_err = _delta(v_pred)[idx] - _delta(v)[idx]
    clean_delta_err = _delta(x1_pred)[idx] - _delta(x1)[idx]
    mse_vel = vel_delta_err.pow(2).mean()
    mse_clean = clean_delta_err.pow(2).mean()
    ratio = (mse_clean / (mse_vel + 1e-12)).item()
    check(
        "clean delta mse ~= sigma^2 * velocity delta mse",
        abs(ratio - sigma_i.item() ** 2) < 1e-3,
    )

    print("\nALL CLS-DP MODULE CHECKS PASSED\n")

    # Keep the original final print for compatibility with any external log parsers.


if __name__ == "__main__":
    main()
