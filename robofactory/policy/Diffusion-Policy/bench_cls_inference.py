"""Micro-benchmark for the CLS-DP Stage 2 action expert.

Answers the one question the flow-matching variant exists to answer: how long does a single
policy call take, and how does that scale with sampler steps. Nothing else in this repo
measured it.

Weights do not affect timing, so by default this builds a randomly-initialised policy
straight from the Hydra config. That means the architecture can be benchmarked before any
training has happened, and the DDPM baseline can be compared against flow matching on the
same machine in one run. Pass --checkpoint to time real weights instead.

Reads the same config groups as training, so what is benchmarked is what would be trained.

Usage, from the `robofactory/` directory:

    # DDPM-100 against flow matching at several step counts, one table
    python policy/Diffusion-Policy/bench_cls_inference.py

    # just the flow head, both solvers, more repeats
    python policy/Diffusion-Policy/bench_cls_inference.py \
        --heads flow --solvers euler,midpoint --steps 1,2,4,8,16 --repeats 50

    # real weights
    python policy/Diffusion-Policy/bench_cls_inference.py \
        --heads flow --checkpoint checkpoints/LiftBarrier-rf_clsdpfm_Agent0_150/100.ckpt

Note the caveat that matters for interpreting the result: this measures the *policy call*.
A full episode also spends time in TOPP smoothing and simulator substeps, and the policy
runs only once per 6 executed steps, so end-to-end episode speedup is much smaller than
the ratio below. eval_multi_cls_dp.py reports both numbers separately.
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dill  # noqa: E402
import hydra  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from diffusion_policy.model.common.normalizer import LinearNormalizer  # noqa: E402

OmegaConf.register_new_resolver("eval", eval, replace=True)

CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "diffusion_policy", "config"
)


def compose(config_name, overrides):
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return hydra_compose(config_name=config_name, overrides=overrides)


def build_policy(args, head):
    overrides = [
        f"task_name={args.task_name}",
        f"n_agents={args.n_agents}",
        f"data_num={args.data_num}",
        "task.dataset.zarr_path=/nonexistent.zarr",
        f"sampler={head}",
        f"action_space={args.action_space}",
    ]
    if head == "flow":
        overrides.append(f"policy.solver={args.solver_for_build}")
    cfg = compose(args.config_name, overrides)
    policy = hydra.utils.instantiate(cfg.policy)

    if args.checkpoint:
        payload = torch.load(open(args.checkpoint, "rb"), pickle_module=dill)
        key = "ema_model" if "ema_model" in payload["state_dicts"] else "model"
        policy.load_state_dict(payload["state_dicts"][key])
        print(f"loaded weights from {args.checkpoint} ({key})")
    else:
        # Timing is independent of weight values, so a fitted-on-noise normalizer and
        # random init are sufficient and let this run before training exists.
        action_dim = cfg.task.shape_meta["action"]["shape"][0]
        cam = cfg.task.shape_meta["obs"]["head_cam"]["shape"]
        normalizer = LinearNormalizer()
        normalizer.fit(
            {
                "action": torch.randn(256, action_dim),
                "agent_pos": torch.randn(256, action_dim),
                "head_cam": torch.rand(8, *cam),
            }
        )
        policy.set_normalizer(normalizer)

    return policy, cfg


def make_obs(cfg, batch_size, device, n_image_tokens, n_text_tokens):
    cam = cfg.task.shape_meta["obs"]["head_cam"]["shape"]
    action_dim = cfg.task.shape_meta["action"]["shape"][0]
    n_obs = cfg.n_obs_steps
    feature_dim = cfg.feature_dim
    return {
        "head_cam": torch.rand(batch_size, n_obs, *cam, device=device),
        "agent_pos": torch.randn(batch_size, n_obs, action_dim, device=device),
        # The prior consumes only the current frame, so these are unbatched over time.
        "prior_image_tokens": torch.randn(
            batch_size, n_image_tokens, feature_dim, device=device
        ),
        "prior_text_tokens": torch.randn(
            batch_size, n_text_tokens, feature_dim, device=device
        ),
        "prior_text_mask": torch.ones(batch_size, n_text_tokens, device=device),
    }


def time_policy(policy, obs, repeats, warmup, device):
    calls = {"n": 0}
    inner = policy.model.forward

    def counting(*a, **k):
        calls["n"] += 1
        return inner(*a, **k)

    policy.model.forward = counting

    try:
        with torch.no_grad():
            for _ in range(warmup):
                policy.predict_action(obs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            calls["n"] = 0
            samples = []
            for _ in range(repeats):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                policy.predict_action(obs)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                samples.append((time.perf_counter() - start) * 1000.0)
    finally:
        policy.model.forward = inner

    return samples, calls["n"] / repeats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-name", default="cls_dp")
    p.add_argument("--heads", default="ddpm,flow", help="comma-separated: ddpm,flow")
    p.add_argument("--steps", default="1,2,4,8,16", help="flow step counts to sweep")
    p.add_argument("--solvers", default="euler", help="comma-separated: euler,midpoint")
    p.add_argument("--ddpm-steps", type=int, default=100)
    p.add_argument("--action-space", default="raw", choices=["raw", "latent"])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--task-name", default="LiftBarrier-rf")
    p.add_argument("--n-agents", type=int, default=2)
    p.add_argument("--data-num", type=int, default=150)
    p.add_argument("--image-tokens", type=int, default=197, help="1 + 14x14 SigLIP grid")
    p.add_argument("--text-tokens", type=int, default=64)
    args = p.parse_args()

    device = torch.device(args.device)
    heads = [h.strip() for h in args.heads.split(",") if h.strip()]
    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    steps = [int(s) for s in args.steps.split(",") if s.strip()]

    print(f"device={device}  batch={args.batch_size}  repeats={args.repeats}")
    print(f"{'head':>10} {'solver':>9} {'steps':>6} {'unet calls':>11} "
          f"{'ms/action':>10} {'stdev':>7} {'vs ddpm':>8}")
    print("-" * 68)

    baseline_ms = None
    for head in heads:
        args.solver_for_build = solvers[0] if head == "flow" else "euler"
        policy, cfg = build_policy(args, head)
        policy.to(device).eval()
        obs = make_obs(
            cfg, args.batch_size, device, args.image_tokens, args.text_tokens
        )

        if head == "ddpm":
            configs = [("ddpm", args.ddpm_steps)]
        else:
            configs = [(s, n) for s in solvers for n in steps]

        for solver, n_steps in configs:
            policy.num_inference_steps = n_steps
            if head == "flow":
                policy.transport.solver = solver
            samples, unet_calls = time_policy(
                policy, obs, args.repeats, args.warmup, device
            )
            mean_ms = statistics.mean(samples)
            stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
            if head == "ddpm":
                baseline_ms = mean_ms
            speedup = f"{baseline_ms / mean_ms:.1f}x" if baseline_ms else "-"
            print(
                f"{head:>10} {solver:>9} {n_steps:>6} {unet_calls:>11.1f} "
                f"{mean_ms:>10.2f} {stdev:>7.2f} {speedup:>8}"
            )

        del policy
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print()
    print("ms/action is one predict_action call: prior + ResNet-18 + sampler.")
    print("A full episode also pays TOPP smoothing and simulator substeps, and calls the")
    print("policy once per 6 executed steps, so episode speedup is much smaller than the")
    print("ratio above. eval_multi_cls_dp.py reports both numbers separately.")


if __name__ == "__main__":
    main()
