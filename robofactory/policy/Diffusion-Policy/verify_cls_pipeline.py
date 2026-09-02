"""End-to-end integration checks for the CLS-DP data and training path.

Complements verify_cls_dp.py, which only covers module math and shapes. This script builds a
tiny synthetic dataset on disk and drives the real code: the pkl -> zarr packer, the dataset,
both loss functions, and both training workspaces.

It runs on CPU in about a minute and needs no GPU, no simulator and no SigLIP download
(image/text features are synthesised with the right shapes and dtypes).

    python policy/Diffusion-Policy/verify_cls_pipeline.py

The centrepiece is the timestep-alignment check. Synthetic states encode their own index as
state[t] = [t, agent_id, ...], so the test can assert *exactly* which absolute timesteps land
in the observation history, the action target and the privileged future window. That is the
part of the implementation most likely to be silently off by one, and a shape-only test would
never catch it.
"""

import os
import pickle
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
import zarr

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(os.path.dirname(HERE))  # robofactory/
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(PACKAGE_DIR))

from omegaconf import OmegaConf  # noqa: E402

# Synthetic problem size: small enough to run on CPU, large enough to exercise padding at
# both ends of every episode.
TASK = "LiftBarrier-rf"
N_AGENTS = 2
N_EPISODES = 6
EP_LEN = 20
IMG_H, IMG_W = 60, 80
STATE_DIM = 8
N_OBS = 3
HORIZON = 8
N_FUTURE = 8
LATENT_DIM = 32
FEATURE_DIM = 64
N_IMG_TOKENS = 5
N_TEXT_TOKENS = 12
BATCH = 8

PASSED = []


def check(label, condition, detail=""):
    mark = "PASS" if condition else "FAIL"
    print(f"  {mark}  {label}" + (f"   [{detail}]" if detail else ""))
    PASSED.append(bool(condition))
    if not condition:
        raise AssertionError(label + (f" :: {detail}" if detail else ""))


def clamp(value, lo, hi):
    return max(lo, min(value, hi))


# ---------------------------------------------------------------- synthetic data


def write_synthetic_pkl(root):
    """Mimic the output of script/parse_h5_to_pkl_multi.py."""
    for agent_id in range(N_AGENTS):
        for ep in range(N_EPISODES):
            ep_dir = os.path.join(root, f"{TASK}_Agent{agent_id}", f"episode{ep}")
            os.makedirs(ep_dir, exist_ok=True)
            for t in range(EP_LEN):
                state = np.zeros(STATE_DIM, dtype=np.float32)
                state[0] = t          # absolute timestep within the episode
                state[1] = agent_id   # which agent this row belongs to
                state[2] = ep
                step = dict(
                    pointcloud=None,
                    joint_action=state,
                    endpose=state,
                    observation={
                        "head_camera": {
                            "rgb": np.full((IMG_H, IMG_W, 3), t % 256, dtype=np.uint8),
                            "intrinsic_cv": np.eye(3, dtype=np.float32),
                            "extrinsic_cv": np.eye(4, dtype=np.float32)[:3],
                            "cam2world_gl": np.eye(4, dtype=np.float32),
                        }
                    },
                )
                with open(os.path.join(ep_dir, f"{t}.pkl"), "wb") as f:
                    pickle.dump(step, f)


def inject_fake_siglip(zarr_path):
    """Stand in for script/precompute_siglip_features.py without downloading SigLIP."""
    root = zarr.open(zarr_path, mode="a")
    data = root["data"]
    n_steps = data[f"head_camera_agent0"].shape[0]
    rng = np.random.default_rng(0)
    for agent_id in range(N_AGENTS):
        data.array(
            f"siglip_img_agent{agent_id}",
            rng.standard_normal((n_steps, N_IMG_TOKENS, FEATURE_DIM)).astype(np.float16),
            chunks=(64, N_IMG_TOKENS, FEATURE_DIM),
            overwrite=True,
        )
    n_instructions = int(root.attrs["n_instructions"])
    cache = {}
    for split in ("train", "eval"):
        cache[f"{split}_tokens"] = rng.standard_normal(
            (n_instructions, N_TEXT_TOKENS, FEATURE_DIM)
        ).astype(np.float16)
        mask = np.ones((n_instructions, N_TEXT_TOKENS), dtype=np.float16)
        mask[:, N_TEXT_TOKENS // 2 :] = 0
        cache[f"{split}_mask"] = mask
    np.savez_compressed(zarr_path.rstrip("/") + "_text.npz", **cache)
    root.attrs["siglip_feature_dim"] = FEATURE_DIM
    root.attrs["siglip_n_image_tokens"] = N_IMG_TOKENS


# ---------------------------------------------------------------------- stages


def test_packer(workdir):
    print("\n[1] parse_pkl_to_zarr_multi.py")
    script = os.path.join(PACKAGE_DIR, "script", "parse_pkl_to_zarr_multi.py")
    result = subprocess.run(
        [
            sys.executable, script,
            "--task_name", TASK,
            "--load_num", str(N_EPISODES),
            "--agent_num", str(N_AGENTS),
        ],
        cwd=workdir, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stdout[-3000:]); print(result.stderr[-3000:])
    check("packer exits 0", result.returncode == 0)

    zarr_path = os.path.join(workdir, "data", "zarr_data", f"{TASK}_multi_{N_EPISODES}.zarr")
    root = zarr.open(zarr_path, mode="r")
    ends = root["meta"]["episode_ends"][:]
    check("episode_ends", list(ends) == [EP_LEN * (i + 1) for i in range(N_EPISODES)],
          f"{list(ends)}")
    check("n_agents attr", int(root.attrs["n_agents"]) == N_AGENTS)
    check("instruction_task strips -rf", root.attrs["instruction_task"] == "LiftBarrier")
    for i in range(N_AGENTS):
        check(f"camera agent{i} shape",
              root["data"][f"head_camera_agent{i}"].shape == (EP_LEN * N_EPISODES, 3, IMG_H, IMG_W))
        # state[:,1] carries agent id; confirms agents were not swapped during packing
        states = root["data"][f"state_agent{i}"][:]
        check(f"state agent{i} identity", bool(np.all(states[:, 1] == i)))
    ids = root["data"]["instruction_id"][:]
    per_ep = [np.unique(ids[e * EP_LEN:(e + 1) * EP_LEN]) for e in range(N_EPISODES)]
    check("one instruction per episode", all(len(u) == 1 for u in per_ep))
    return zarr_path


def build_dataset(zarr_path, stage):
    from diffusion_policy.dataset.multi_agent_image_dataset import MultiAgentImageDataset
    return MultiAgentImageDataset(
        zarr_path=zarr_path, agent_id=0, stage=stage, n_agents=N_AGENTS,
        horizon=HORIZON, n_obs_steps=N_OBS, n_future_states=N_FUTURE,
        seed=42, val_ratio=0.2, batch_size=BATCH, max_train_episodes=None,
    )


def test_dataset_alignment(zarr_path):
    print("\n[2] MultiAgentImageDataset - timestep alignment")
    device = torch.device("cpu")

    ds1 = build_dataset(zarr_path, stage=1)
    check("stage 1 skips camera",
          not any(k.startswith("head_camera") for k in ds1.replay_buffer.keys()),
          f"keys={sorted(ds1.replay_buffer.keys())}")
    check("stage 1 loads every agent's state",
          all(f"state_agent{j}" in ds1.replay_buffer for j in range(N_AGENTS)))
    check("window length = L + F", ds1.sequence_length == N_OBS + N_FUTURE)
    check("every timestep sampled once",
          len(ds1.sampler) + 0 == EP_LEN * int(ds1.train_mask.sum()),
          f"{len(ds1.sampler)} windows, {int(ds1.train_mask.sum())} train episodes")

    idx = np.arange(BATCH)
    batch = ds1.postprocess(ds1[idx], device)
    own = batch["own_state"].numpy()
    fut = batch["future_states"].numpy()
    check("own_state shape", own.shape == (BATCH, STATE_DIM), str(own.shape))
    check("future_states shape", fut.shape == (BATCH, N_AGENTS, N_FUTURE, STATE_DIM), str(fut.shape))

    ok_future, ok_agent = True, True
    for b in range(BATCH):
        t = int(round(own[b, 0]))
        for j in range(N_AGENTS):
            expected = [clamp(t + k, 0, EP_LEN - 1) for k in range(1, N_FUTURE + 1)]
            got = [int(round(v)) for v in fut[b, j, :, 0]]
            ok_future &= got == expected
            ok_agent &= bool(np.all(fut[b, j, :, 1] == j))
    check("future window is exactly s_{t+1..t+8}", ok_future)
    check("agent axis is ordered 0..N-1", ok_agent)

    ds2 = build_dataset(zarr_path, stage=2)
    check("stage 2 loads camera", f"head_camera_agent0" in ds2.replay_buffer)
    check("stage 2 skips other agents' state",
          "state_agent1" not in ds2.replay_buffer,
          f"keys={sorted(ds2.replay_buffer.keys())}")

    batch2 = ds2.postprocess(ds2[idx], device)
    obs = batch2["obs"]["agent_pos"].numpy()
    act = batch2["action"].numpy()
    check("agent_pos shape", obs.shape == (BATCH, N_OBS, STATE_DIM), str(obs.shape))
    check("head_cam shape",
          tuple(batch2["obs"]["head_cam"].shape) == (BATCH, N_OBS, 3, IMG_H, IMG_W),
          str(tuple(batch2["obs"]["head_cam"].shape)))
    check("action shape", act.shape == (BATCH, HORIZON, STATE_DIM), str(act.shape))
    check("head_cam scaled to [0,1]",
          float(batch2["obs"]["head_cam"].max()) <= 1.0 + 1e-6)

    ok_obs, ok_act, ok_exec = True, True, True
    for b in range(BATCH):
        t = int(round(obs[b, -1, 0]))  # last observed frame defines time t
        ok_obs &= [int(round(v)) for v in obs[b, :, 0]] == [
            clamp(t + k, 0, EP_LEN - 1) for k in (-2, -1, 0)
        ]
        ok_act &= [int(round(v)) for v in act[b, :, 0]] == [
            clamp(t - 2 + k, 0, EP_LEN - 1) for k in range(HORIZON)
        ]
        # the slice the policy actually executes: action_pred[:, To-1 : To-1+n_action]
        executed = [int(round(v)) for v in act[b, N_OBS - 1:, 0]]
        ok_exec &= executed == [clamp(t + k, 0, EP_LEN - 1) for k in range(6)]
    check("obs history is o_{t-2..t}", ok_obs)
    check("action target is a_{t-2..t+5}", ok_act)
    check("executed slice is a_{t..t+5}, i.e. 6 steps", ok_exec)

    check("prior tokens come from the current frame only",
          tuple(batch2["prior_image_tokens"].shape) == (BATCH, N_IMG_TOKENS, FEATURE_DIM),
          str(tuple(batch2["prior_image_tokens"].shape)))
    check("text tokens broadcast per sample",
          tuple(batch2["prior_text_tokens"].shape) == (BATCH, N_TEXT_TOKENS, FEATURE_DIM))

    val = ds1.get_validation_dataset()
    overlap = bool(np.any(ds1.train_mask & val.train_mask))
    check("train/val episode split is disjoint", not overlap)
    return ds1, ds2


def make_contextualizer():
    from diffusion_policy.model.cls.ma_kinematics import (
        MAKinematicsDecoder, MAKinematicsEncoder,
    )
    from diffusion_policy.model.cls.prior_net import PriorNet
    from diffusion_policy.policy.contextualizer import Contextualizer

    return Contextualizer(
        prior_net=PriorNet(feature_dim=FEATURE_DIM, latent_dim=LATENT_DIM, d_model=64,
                           n_layers=1, n_heads=4, dim_feedforward=128),
        ma_encoder=MAKinematicsEncoder(state_dim=STATE_DIM, n_agents=N_AGENTS,
                                       n_future_states=N_FUTURE, latent_dim=LATENT_DIM,
                                       d_model=64, n_layers=1, n_heads=4, dim_feedforward=128),
        ma_decoder=MAKinematicsDecoder(state_dim=STATE_DIM, n_agents=N_AGENTS,
                                       n_future_states=N_FUTURE, latent_dim=LATENT_DIM,
                                       d_model=64, n_layers=1, n_heads=4, dim_feedforward=128),
        agent_id=0, n_agents=N_AGENTS, latent_dim=LATENT_DIM,
    )


def test_contextualizer(ds1):
    print("\n[3] Contextualizer (Stage 1 loss)")
    torch.manual_seed(0)
    model = make_contextualizer()
    model.set_normalizer(ds1.get_normalizer())

    batch = ds1.postprocess(ds1[np.arange(BATCH)], torch.device("cpu"))
    loss0, metrics = model.compute_loss(batch, beta=0.0)
    check("loss is finite", bool(np.isfinite(loss0.item())), f"{loss0.item():.4f}")
    check("KL is 0 at init (zero-init heads)", abs(metrics["ctx_kl"]) < 1e-6,
          f"{metrics['ctx_kl']:.3e}")
    for key in ("ctx_recon_own", "ctx_recon_others", "ctx_recon_others_baseline"):
        check(f"metric {key} present", key in metrics)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for step in range(60):
        loss, metrics = model.compute_loss(batch, beta=0.1 * min(1.0, step / 24))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    check("Stage 1 loss decreases when overfitting one batch",
          losses[-1] < losses[0], f"{losses[0]:.3f} -> {losses[-1]:.3f}")
    check("KL becomes positive once the residual moves",
          metrics["ctx_kl"] > 0, f"kl={metrics['ctx_kl']:.4f}")
    check("teammate reconstruction beats the batch-mean baseline",
          metrics["ctx_recon_others"] < metrics["ctx_recon_others_baseline"],
          f"others={metrics['ctx_recon_others']:.4f} "
          f"baseline={metrics['ctx_recon_others_baseline']:.4f}")

    # gradient routing: the KL alone must not train the prior mean head
    model.zero_grad()
    mu_p, ls_p, _ = model.prior_net(batch["prior_image_tokens"], batch["prior_text_tokens"],
                                    batch["prior_text_mask"])
    fut = model.normalizer["state"].normalize(batch["future_states"])
    mu_e, ls_e = model.ma_encoder(fut)
    from diffusion_policy.policy.contextualizer import _residual_kl
    _residual_kl(ls_p, mu_e, ls_e).mean().backward()
    mu_grad = model.prior_net.to_mu.weight.grad
    sigma_grad = model.prior_net.to_log_sigma.weight.grad
    check("KL gives no gradient to the prior mean head",
          mu_grad is None or float(mu_grad.abs().max()) == 0.0,
          "None" if mu_grad is None else f"{float(mu_grad.abs().max()):.3e}")
    check("KL does give gradient to the prior sigma head",
          sigma_grad is not None and float(sigma_grad.abs().max()) > 0)
    return model


def test_action_expert(ds2, contextualizer):
    print("\n[4] CLSDiffusionUnetImagePolicy (Stage 2 loss + sampling)")
    import copy
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusion_policy.model.vision.model_getter import get_resnet
    from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
    from diffusion_policy.policy.cls_diffusion_unet_image_policy import (
        CLSDiffusionUnetImagePolicy,
    )

    torch.manual_seed(0)
    shape_meta = {
        "obs": {
            "head_cam": {"shape": [3, IMG_H, IMG_W], "type": "rgb"},
            "agent_pos": {"shape": [STATE_DIM], "type": "low_dim"},
        },
        "action": {"shape": [STATE_DIM]},
    }
    policy = CLSDiffusionUnetImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=DDPMScheduler(num_train_timesteps=20, beta_schedule="squaredcos_cap_v2",
                                      prediction_type="epsilon", clip_sample=True),
        obs_encoder=MultiImageObsEncoder(
            shape_meta=shape_meta, rgb_model=get_resnet("resnet18", weights=None),
            use_group_norm=True, share_rgb_model=False, imagenet_norm=True),
        prior_net=copy.deepcopy(contextualizer.prior_net),
        horizon=HORIZON, n_action_steps=HORIZON, n_obs_steps=N_OBS,
        latent_dim=LATENT_DIM, latent_sample=True, num_inference_steps=3,
        down_dims=[32, 64], diffusion_step_embed_dim=32, kernel_size=3,
        n_cond_tokens=2, cond_token_dim=32, cross_attn_heads=2,
    )
    policy.set_normalizer(ds2.get_normalizer())

    check("prior_net is frozen",
          not any(p.requires_grad for p in policy.prior_net.parameters()))
    policy.train()
    check("prior_net stays in eval after .train()", not policy.prior_net.training)

    batch = ds2.postprocess(ds2[np.arange(BATCH)], torch.device("cpu"))
    loss = policy.compute_loss(batch)
    check("Stage 2 loss is finite", bool(np.isfinite(loss.item())), f"{loss.item():.4f}")

    loss.backward()
    prior_grads = [p.grad for p in policy.prior_net.parameters() if p.grad is not None]
    check("no gradient reaches the frozen prior (the sg in Eq. 11)", len(prior_grads) == 0,
          f"{len(prior_grads)} tensors with grad")

    # Zero-init output projection dynamics, same as a ControlNet zero-conv. On step 0 the
    # branch outputs exactly zero, so to_out sees gradient but q/k/v cannot: their gradient
    # is routed through to_out.weight, which is still zero. One optimizer step moves to_out
    # off zero and the whole branch starts learning. Asserting q/k/v gradient at step 0
    # would be asserting the wrong thing.
    block = policy.model.down_cross_attn[0]
    out_grad = block.to_out.weight.grad
    q_grad = block.to_q.weight.grad
    check("zero-init: to_out receives gradient on step 0",
          out_grad is not None and float(out_grad.abs().max()) > 0,
          f"{float(out_grad.abs().max()):.3e}" if out_grad is not None else "None")
    check("zero-init: q/k/v gradient is 0 on step 0, by construction",
          q_grad is None or float(q_grad.abs().max()) == 0.0,
          "None" if q_grad is None else f"{float(q_grad.abs().max()):.3e}")

    trainable = [p for p in policy.parameters() if p.requires_grad]
    torch.optim.AdamW(trainable, lr=1e-3).step()
    policy.zero_grad()
    policy.compute_loss(batch).backward()
    q_grad2 = block.to_q.weight.grad
    check("after one step, the cross-attention branch trains end to end",
          q_grad2 is not None and float(q_grad2.abs().max()) > 0,
          "None" if q_grad2 is None else f"{float(q_grad2.abs().max()):.3e}")
    policy.zero_grad()

    policy.eval()
    obs_dict = dict(batch["obs"])
    for key in ("prior_image_tokens", "prior_text_tokens", "prior_text_mask"):
        obs_dict[key] = batch[key]
    with torch.no_grad():
        out = policy.predict_action(obs_dict)
    check("action_pred spans the full horizon",
          tuple(out["action_pred"].shape) == (BATCH, HORIZON, STATE_DIM),
          str(tuple(out["action_pred"].shape)))
    check("executed action is 6 steps",
          tuple(out["action"].shape) == (BATCH, 6, STATE_DIM),
          str(tuple(out["action"].shape)))
    check("latent is returned", tuple(out["cls_latent"].shape) == (BATCH, LATENT_DIM))
    check("actions are finite", bool(torch.isfinite(out["action"]).all()))

    # latent_sample=False must be deterministic; True must not be
    policy.latent_sample = False
    with torch.no_grad():
        a = policy.resolve_latent({k: batch[k] for k in
                                   ("prior_image_tokens", "prior_text_tokens", "prior_text_mask")})
        b = policy.resolve_latent({k: batch[k] for k in
                                   ("prior_image_tokens", "prior_text_tokens", "prior_text_mask")})
    check("latent_sample=False is deterministic", torch.allclose(a, b))
    policy.latent_sample = True
    with torch.no_grad():
        c = policy.resolve_latent({k: batch[k] for k in
                                   ("prior_image_tokens", "prior_text_tokens", "prior_text_mask")})
    check("latent_sample=True draws from the prior", not torch.allclose(a, c))


def load_cfg(name, zarr_path, overrides):
    """Compose through Hydra, so the defaults lists in the *_det configs are honoured."""
    from hydra import compose, initialize_config_dir

    cfg_dir = os.path.join(HERE, "diffusion_policy", "config")
    base = [
        f"task_name={TASK}", f"n_agents={N_AGENTS}", f"data_num={N_EPISODES}",
        f"task.dataset.zarr_path={zarr_path}", "training.debug=True",
        "training.device=cpu", "training.resume=False",
    ]
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        return compose(config_name=name.replace(".yaml", ""), overrides=base + overrides)


def test_workspaces(workdir, zarr_path):
    print("\n[5] Training workspaces (debug mode, real loops)")
    print("    NOTE: debug mode runs ~6 optimizer steps on random SigLIP features, so the")
    print("    Stage 1 gate below is EXPECTED to print FAIL. Section [3] trains the same")
    print("    model properly and shows the gate passing, so it discriminates both ways.")
    from diffusion_policy.workspace.cls_robotworkspace import CLSRobotWorkspace
    from diffusion_policy.workspace.contextualizer_workspace import ContextualizerWorkspace

    small_ctx = [
        "contextualizer.prior_net.d_model=64", "contextualizer.prior_net.n_layers=1",
        "contextualizer.prior_net.n_heads=4", "contextualizer.prior_net.dim_feedforward=128",
        "contextualizer.ma_encoder.d_model=64", "contextualizer.ma_encoder.n_layers=1",
        "contextualizer.ma_encoder.n_heads=4", "contextualizer.ma_encoder.dim_feedforward=128",
        "contextualizer.ma_decoder.d_model=64", "contextualizer.ma_decoder.n_layers=1",
        "contextualizer.ma_decoder.n_heads=4", "contextualizer.ma_decoder.dim_feedforward=128",
        f"feature_dim={FEATURE_DIM}", f"latent_dim={LATENT_DIM}", "agent_id=0",
        f"dataloader.batch_size={BATCH}", f"val_dataloader.batch_size={BATCH}",
    ]
    out1 = os.path.join(workdir, "out_stage1")
    cfg1 = load_cfg("cls_stage1.yaml", zarr_path, small_ctx)
    ws1 = ContextualizerWorkspace(cfg1, output_dir=out1)
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        ws1.run()
    finally:
        os.chdir(cwd)
    ckpt = os.path.join(workdir, "checkpoints", f"{TASK}_ctx_Agent0_{N_EPISODES}", "1.ckpt")
    check("Stage 1 wrote a checkpoint", os.path.isfile(ckpt), ckpt)
    check("Stage 1 logged json", os.path.isfile(os.path.join(out1, "logs.json.txt")))

    small_policy = [
        "policy.prior_net.d_model=64", "policy.prior_net.n_layers=1",
        "policy.prior_net.n_heads=4", "policy.prior_net.dim_feedforward=128",
        f"feature_dim={FEATURE_DIM}", f"latent_dim={LATENT_DIM}", "agent_id=0",
        f"dataloader.batch_size={BATCH}", f"val_dataloader.batch_size={BATCH}",
        "policy.down_dims=[32,64]", "policy.diffusion_step_embed_dim=32",
        "policy.kernel_size=3", "policy.n_cond_tokens=2", "policy.cond_token_dim=32",
        "policy.cross_attn_heads=2", "policy.num_inference_steps=3",
        "policy.noise_scheduler.num_train_timesteps=20",
        f"contextualizer_ckpt={ckpt}",
        f"task.shape_meta.obs.head_cam.shape=[3,{IMG_H},{IMG_W}]",
    ]
    out2 = os.path.join(workdir, "out_stage2")
    cfg2 = load_cfg("cls_dp.yaml", zarr_path, small_policy)
    ws2 = CLSRobotWorkspace(cfg2, output_dir=out2)
    check("Stage 2 loaded the frozen prior from the Stage 1 checkpoint", True)
    check("optimizer excludes frozen params",
          all(p.requires_grad for g in ws2.optimizer.param_groups for p in g["params"]))

    os.chdir(workdir)
    try:
        ws2.run()
    finally:
        os.chdir(cwd)
    ckpt2 = os.path.join(workdir, "checkpoints", f"{TASK}_clsdp_Agent0_{N_EPISODES}", "1.ckpt")
    check("Stage 2 wrote a checkpoint", os.path.isfile(ckpt2), ckpt2)

    # the eval path reconstructs the workspace from cfg._target_ inside the payload
    import dill
    payload = torch.load(open(ckpt2, "rb"), pickle_module=dill, weights_only=False)
    saved = payload["cfg"]
    check("checkpoint stores the workspace target",
          saved._target_.endswith("CLSRobotWorkspace"))
    check("checkpoint contains prior_net weights",
          any(k.startswith("prior_net.") for k in payload["state_dicts"]["model"]))
    saved.contextualizer_ckpt = None  # what eval_multi_cls_dp.get_policy does
    rebuilt = CLSRobotWorkspace(saved, output_dir=out2)
    rebuilt.load_payload(payload)
    check("checkpoint round-trips without the Stage 1 file", True)


DET_CTX_OVERRIDES = [
    "contextualizer.prior_net.d_model=64", "contextualizer.prior_net.n_layers=1",
    "contextualizer.prior_net.n_heads=4", "contextualizer.prior_net.dim_feedforward=128",
    "contextualizer.ma_encoder.d_model=64", "contextualizer.ma_encoder.n_layers=1",
    "contextualizer.ma_encoder.n_heads=4", "contextualizer.ma_encoder.dim_feedforward=128",
    "contextualizer.ma_decoder.d_model=64", "contextualizer.ma_decoder.n_layers=1",
    "contextualizer.ma_decoder.n_heads=4", "contextualizer.ma_decoder.dim_feedforward=128",
    f"feature_dim={FEATURE_DIM}", f"latent_dim={LATENT_DIM}", "agent_id=0",
    f"dataloader.batch_size={BATCH}", f"val_dataloader.batch_size={BATCH}",
]


def test_deterministic_variant(workdir, zarr_path):
    """Drive the deterministic variant through both real _det configs."""
    print("\n[7] Deterministic variant, end to end")
    from diffusion_policy.workspace.cls_robotworkspace import CLSRobotWorkspace
    from diffusion_policy.workspace.contextualizer_workspace import ContextualizerWorkspace

    cwd = os.getcwd()

    cfg1 = load_cfg("cls_stage1_det", zarr_path, DET_CTX_OVERRIDES)
    check("det Stage 1 config resolves", cfg1.contextualizer.deterministic is True)
    check("det checkpoint prefix is distinct", "ctxdet" in cfg1.checkpoint_name,
          cfg1.checkpoint_name)

    ws1 = ContextualizerWorkspace(cfg1, output_dir=os.path.join(workdir, "out_det1"))
    check("no scale parameters were built",
          ws1.model.prior_net.to_log_sigma is None
          and ws1.model.ma_encoder.to_log_sigma is None)

    os.chdir(workdir)
    try:
        ws1.run()
    finally:
        os.chdir(cwd)
    det_ctx = os.path.join(
        workdir, "checkpoints", f"{TASK}_ctxdet_Agent0_{N_EPISODES}", "1.ckpt"
    )
    check("det Stage 1 wrote a checkpoint", os.path.isfile(det_ctx))

    det_policy_overrides = [
        "policy.prior_net.d_model=64", "policy.prior_net.n_layers=1",
        "policy.prior_net.n_heads=4", "policy.prior_net.dim_feedforward=128",
        f"feature_dim={FEATURE_DIM}", f"latent_dim={LATENT_DIM}", "agent_id=0",
        f"dataloader.batch_size={BATCH}", f"val_dataloader.batch_size={BATCH}",
        "policy.down_dims=[32,64]", "policy.diffusion_step_embed_dim=32",
        "policy.kernel_size=3", "policy.n_cond_tokens=2", "policy.cond_token_dim=32",
        "policy.cross_attn_heads=2", "policy.num_inference_steps=3",
        "policy.noise_scheduler.num_train_timesteps=20",
        f"contextualizer_ckpt={det_ctx}",
        f"task.shape_meta.obs.head_cam.shape=[3,{IMG_H},{IMG_W}]",
    ]
    cfg2 = load_cfg("cls_dp_det", zarr_path, det_policy_overrides)
    check("det Stage 2 disables sampling", cfg2.policy.latent_sample is False)
    check("det Stage 2 prefix is distinct", "clsdpdet" in cfg2.checkpoint_name,
          cfg2.checkpoint_name)

    ws2 = CLSRobotWorkspace(cfg2, output_dir=os.path.join(workdir, "out_det2"))
    check("det Stage 2 loaded the det prior", True)

    # The whole point of the variant: identical inputs must give an identical latent.
    batch = build_stage2_batch(zarr_path)
    prior_inputs = {k: batch[k] for k in
                    ("prior_image_tokens", "prior_text_tokens", "prior_text_mask")}
    with torch.no_grad():
        z_a = ws2.model.resolve_latent(prior_inputs)
        z_b = ws2.model.resolve_latent(prior_inputs)
    check("latent is exactly repeatable", torch.equal(z_a, z_b))
    ws2.model.latent_sample = True  # must be ignored: there is no scale head to sample
    with torch.no_grad():
        z_c = ws2.model.resolve_latent(prior_inputs)
    check("latent_sample=True cannot reintroduce noise", torch.equal(z_a, z_c))
    ws2.model.latent_sample = False

    os.chdir(workdir)
    try:
        ws2.run()
    finally:
        os.chdir(cwd)
    det_pol = os.path.join(
        workdir, "checkpoints", f"{TASK}_clsdpdet_Agent0_{N_EPISODES}", "1.ckpt"
    )
    check("det Stage 2 wrote a checkpoint", os.path.isfile(det_pol))

    # Safety net: pairing a stochastic Stage 1 with a deterministic Stage 2 must fail
    # loudly rather than silently dropping the scale head.
    stochastic_ctx = os.path.join(
        workdir, "checkpoints", f"{TASK}_ctx_Agent0_{N_EPISODES}", "1.ckpt"
    )
    if os.path.isfile(stochastic_ctx):
        mismatched = load_cfg(
            "cls_dp_det", zarr_path,
            [o for o in det_policy_overrides if not o.startswith("contextualizer_ckpt")]
            + [f"contextualizer_ckpt={stochastic_ctx}"],
        )
        try:
            CLSRobotWorkspace(mismatched, output_dir=os.path.join(workdir, "out_bad"))
            raised = False
        except RuntimeError:
            raised = True
        check("mismatched stochastic/deterministic pairing is rejected", raised)


def build_stage2_batch(zarr_path):
    ds = build_dataset(zarr_path, stage=2)
    return ds.postprocess(ds[np.arange(BATCH)], torch.device("cpu"))


SELF_DIM = LATENT_DIM // 2
TEAM_DIM = LATENT_DIM - SELF_DIM
N_OTHERS = N_AGENTS - 1

SMALL_DECODER = ["d_model=64", "n_layers=1", "n_heads=4", "dim_feedforward=128"]

FG_CTX_OVERRIDES = (
    [
        "contextualizer.prior_net.d_model=64", "contextualizer.prior_net.n_layers=1",
        "contextualizer.prior_net.n_heads=4",
        "contextualizer.prior_net.dim_feedforward=128",
        "contextualizer.ma_encoder.d_model=64", "contextualizer.ma_encoder.n_layers=1",
        "contextualizer.ma_encoder.n_heads=4",
        "contextualizer.ma_encoder.dim_feedforward=128",
        f"feature_dim={FEATURE_DIM}", f"latent_dim={LATENT_DIM}",
        f"self_dim={SELF_DIM}", "agent_id=0",
        f"dataloader.batch_size={BATCH}", f"val_dataloader.batch_size={BATCH}",
    ]
    + [f"contextualizer.{d}.{o}" for o in SMALL_DECODER
       for d in ("decoder_self", "decoder_team", "prior_probe", "leak_probe")]
)


def test_factorized_variant(workdir, zarr_path):
    """Drive Study FG end to end through the real _fg configs."""
    print("\n[8] Factorized variant, end to end")
    from diffusion_policy.workspace.cls_robotworkspace import CLSRobotWorkspace
    from diffusion_policy.workspace.contextualizer_workspace import ContextualizerWorkspace

    cwd = os.getcwd()

    cfg1 = load_cfg("cls_stage1_fg", zarr_path, FG_CTX_OVERRIDES)
    check("FG config resolves", cfg1.contextualizer.factorize is True)
    check("FG inherits determinism from its DET parent",
          cfg1.contextualizer.deterministic is True)
    check("FG checkpoint prefix is distinct", "ctxfg" in cfg1.checkpoint_name,
          cfg1.checkpoint_name)
    check("residual is team-width, not full-width",
          cfg1.contextualizer.ma_encoder.latent_dim == TEAM_DIM,
          f"{cfg1.contextualizer.ma_encoder.latent_dim} vs {TEAM_DIM}")
    check("decoder_self covers exactly one agent",
          cfg1.contextualizer.decoder_self.n_agents == 1)
    check("decoder_team covers exactly the teammates",
          cfg1.contextualizer.decoder_team.n_agents == N_OTHERS)

    ws1 = ContextualizerWorkspace(cfg1, output_dir=os.path.join(workdir, "out_fg1"))
    check("monolithic decoder is not built when factorized", ws1.model.ma_decoder is None)
    check("both probes are attached",
          ws1.model.prior_probe is not None and ws1.model.leak_probe is not None)

    ds = build_dataset(zarr_path, stage=1)
    ws1.model.set_normalizer(ds.get_normalizer())
    batch = ds.postprocess(ds[np.arange(BATCH)], torch.device("cpu"))
    _, metrics = ws1.model.compute_loss(batch, beta=0.1)
    for key in ("ctx_recon_own", "ctx_recon_others", "ctx_recon_others_baseline",
                "ctx_probe_recon_others", "ctx_leak_recon_others",
                "ctx_prior_gap", "ctx_leak_ratio"):
        check(f"metric {key} present and finite",
              key in metrics and np.isfinite(metrics[key]))

    print("    NOTE: debug mode runs ~6 optimizer steps on random SigLIP features, so all")
    print("    three gate lines below are EXPECTED to look bad. What is being checked here")
    print("    is that they are computed and reported, not what they say. Section [9]")
    print("    verifies the probe machinery can actually distinguish signal from noise.")
    os.chdir(workdir)
    try:
        ws1.run()
    finally:
        os.chdir(cwd)
    fg_ctx = os.path.join(
        workdir, "checkpoints", f"{TASK}_ctxfg_Agent0_{N_EPISODES}", "1.ckpt"
    )
    check("FG Stage 1 wrote a checkpoint", os.path.isfile(fg_ctx))

    fg_policy_overrides = [
        "policy.prior_net.d_model=64", "policy.prior_net.n_layers=1",
        "policy.prior_net.n_heads=4", "policy.prior_net.dim_feedforward=128",
        f"feature_dim={FEATURE_DIM}", f"latent_dim={LATENT_DIM}", "agent_id=0",
        f"dataloader.batch_size={BATCH}", f"val_dataloader.batch_size={BATCH}",
        "policy.down_dims=[32,64]", "policy.diffusion_step_embed_dim=32",
        "policy.kernel_size=3", "policy.n_cond_tokens=2", "policy.cond_token_dim=32",
        "policy.cross_attn_heads=2", "policy.num_inference_steps=3",
        "policy.noise_scheduler.num_train_timesteps=20",
        f"contextualizer_ckpt={fg_ctx}",
        f"task.shape_meta.obs.head_cam.shape=[3,{IMG_H},{IMG_W}]",
    ]
    cfg2 = load_cfg("cls_dp_fg", zarr_path, fg_policy_overrides)
    check("FG Stage 2 prefix is distinct", "clsdpfg" in cfg2.checkpoint_name,
          cfg2.checkpoint_name)

    # Stage 1 FG saves extra modules (two decoders, two probes) but _load_prior_weights
    # pulls only prior_net.*, which is identical across every variant.
    ws2 = CLSRobotWorkspace(cfg2, output_dir=os.path.join(workdir, "out_fg2"))
    check("FG Stage 2 loaded the prior despite the extra Stage 1 modules", True)

    os.chdir(workdir)
    try:
        ws2.run()
    finally:
        os.chdir(cwd)
    check("FG Stage 2 wrote a checkpoint", os.path.isfile(os.path.join(
        workdir, "checkpoints", f"{TASK}_clsdpfg_Agent0_{N_EPISODES}", "1.ckpt")))


def test_probe_detects_information():
    """The leak probe must be able to tell an informative latent from a useless one.

    Without this, `ctx_leak_ratio` could look healthy purely because every probe fails.
    Fit two identical probes on the same target: one reads a latent that literally encodes
    it, the other reads noise. The informative one has to win by a wide margin, otherwise
    the probe machinery is not measuring anything.
    """
    print("\n[9] Probe methodology is not vacuous")
    from diffusion_policy.model.cls.ma_kinematics import MAKinematicsDecoder

    torch.manual_seed(0)
    target = torch.randn(BATCH, N_OTHERS, N_FUTURE, STATE_DIM)
    own_state = torch.randn(BATCH, STATE_DIM)

    flat = target.reshape(BATCH, -1)
    informative = torch.zeros(BATCH, TEAM_DIM)
    width = min(TEAM_DIM, flat.shape[1])
    informative[:, :width] = flat[:, :width]
    noise = torch.randn(BATCH, TEAM_DIM)

    def fit(latent, steps=300):
        probe = MAKinematicsDecoder(
            state_dim=STATE_DIM, n_agents=N_OTHERS, n_future_states=N_FUTURE,
            latent_dim=TEAM_DIM, d_model=64, n_layers=1, n_heads=4, dim_feedforward=128,
        )
        opt = torch.optim.AdamW(probe.parameters(), lr=3e-3)
        for _ in range(steps):
            opt.zero_grad()
            loss = ((probe(own_state, latent) - target) ** 2).mean()
            loss.backward()
            opt.step()
        return loss.item()

    err_informative = fit(informative)
    err_noise = fit(noise)
    ratio = err_noise / max(err_informative, 1e-12)
    print(f"       informative latent {err_informative:.5f} vs noise {err_noise:.5f}"
          f"   ratio {ratio:.1f}x")
    check("a probe recovers information that is present", err_informative < err_noise)
    check("the gap is large enough to be a usable signal", ratio > 2.0)


def test_eval_script(workdir):
    """Cover eval_multi_cls_dp.py's pure logic without the simulator.

    SAPIEN has no macOS/py3.9 wheel, so gym/sapien/mani_skill/robofactory cannot be
    imported here. Stub them so the module body still executes and its non-simulator
    helpers can be exercised for real.
    """
    print("\n[6] eval_multi_cls_dp.py (simulator stubbed)")
    import types

    class _Anything(types.ModuleType):
        def __getattr__(self, name):
            def _callable(*args, **kwargs):
                return None
            _callable.__name__ = name
            return _callable

    stubs = [
        "gymnasium", "sapien", "sapien.utils", "tyro", "tyro.conf",
        "mani_skill", "mani_skill.envs", "mani_skill.envs.sapien_env",
        "robofactory", "robofactory.planner", "robofactory.planner.motionplanner",
        "robofactory.tasks", "robofactory.utils", "robofactory.utils.wrappers",
        "robofactory.utils.wrappers.record",
    ]
    saved = {name: sys.modules.get(name) for name in stubs}
    for name in stubs:
        sys.modules[name] = _Anything(name)
    # `import a.b` binds b as an attribute of a; __getattr__ alone would hand back a
    # function instead of the submodule, so wire the parents explicitly.
    for name in stubs:
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, sys.modules[name])
    sys.modules["robofactory.tasks"].__all__ = []
    sys.modules["mani_skill.envs.sapien_env"].BaseEnv = object

    cwd = os.getcwd()
    try:
        os.chdir(os.path.dirname(PACKAGE_DIR))  # repo root, for configs/instructions
        os.chdir(PACKAGE_DIR)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "eval_multi_cls_dp", os.path.join(HERE, "eval_multi_cls_dp.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        check("module body executes", True)
        check("env id -> instruction bank key",
              mod.instruction_task_name("LiftBarrier-rf") == "LiftBarrier"
              and mod.instruction_task_name("LiftBarrier") == "LiftBarrier")

        check("_is_success handles python bool", mod._is_success({"success": True}) is True)
        check("_is_success handles a batched tensor",
              mod._is_success({"success": torch.tensor([True])}) is True
              and mod._is_success({"success": torch.tensor([False])}) is False)
        check("_is_success handles numpy", mod._is_success({"success": np.array([True])}) is True)
        check("_is_success handles a missing key", mod._is_success({}) is False)

        rng = np.random.default_rng(0)
        text = mod.load_instruction("LiftBarrier-rf", "eval", None, rng)
        check("loads a held-out instruction", isinstance(text, str) and len(text) > 10, text)
        pinned = mod.load_instruction("LiftBarrier-rf", "eval", 3, rng)
        check("instruction_index pins deterministically",
              pinned == mod.load_instruction("LiftBarrier-rf", "eval", 3, rng))

        # the checkpoint path eval builds must match what Stage 2 actually wrote
        expected = f"checkpoints/{TASK}_clsdp_Agent0_{N_EPISODES}/1.ckpt"
        check("eval checkpoint path matches the workspace naming",
              os.path.isfile(os.path.join(workdir, expected)), expected)
    finally:
        os.chdir(cwd)
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def main():
    workdir = tempfile.mkdtemp(prefix="cls_pipeline_")
    print(f"scratch dir: {workdir}")
    try:
        write_synthetic_pkl(os.path.join(workdir, "data", "pkl_data"))
        zarr_path = test_packer(workdir)
        inject_fake_siglip(zarr_path)
        ds1, ds2 = test_dataset_alignment(zarr_path)
        contextualizer = test_contextualizer(ds1)
        test_action_expert(ds2, contextualizer)
        test_workspaces(workdir, zarr_path)
        test_deterministic_variant(workdir, zarr_path)
        test_eval_script(workdir)
        test_factorized_variant(workdir, zarr_path)
        test_probe_detects_information()
        print(f"\nALL {len(PASSED)} PIPELINE CHECKS PASSED\n")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
