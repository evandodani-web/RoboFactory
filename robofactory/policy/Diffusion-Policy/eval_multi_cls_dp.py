import sys

sys.path.append("./")
sys.path.insert(0, "./policy/Diffusion-Policy")

import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager

import dill
import gymnasium as gym
import hydra
import numpy as np
import sapien
import torch
import tyro
import yaml
from dataclasses import dataclass
from typing import Annotated, List, Optional, Union

from mani_skill.envs.sapien_env import BaseEnv

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.dp_runner import DPRunner
from diffusion_policy.model.cls.siglip_encoder import (
    DEFAULT_MODEL_NAME,
    SigLIPFeatureExtractor,
)
from diffusion_policy.workspace.cls_robotworkspace import CLSRobotWorkspace  # noqa: F401
from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver
from robofactory.tasks import *  # noqa: F401,F403
from robofactory.utils.wrappers.record import RecordEpisodeMA

INSTRUCTION_DIR = os.path.join("configs", "instructions")


@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = ""
    """The environment ID of the task you want to simulate"""

    config: str = "${CONFIG_DIR}/table/lift_barrier.yaml"
    """Configuration to build scenes, assets and agents."""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "rgb"
    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = None
    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    reward_mode: Optional[str] = None
    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = "pd_joint_pos"
    render_mode: str = "rgb_array"
    shader: str = "default"
    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    quiet: bool = False
    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = 10000

    data_num: int = 150
    """The number of episode demonstrations used for training the policy"""

    checkpoint_num: int = 100
    """The training epoch of the checkpoint to load"""

    record_dir: Optional[str] = "./eval_video/{env_id}"
    max_steps: int = 250

    instruction_split: str = "eval"
    """Which instruction bank half to sample from. The paper evaluates on held-out ones."""

    instruction_index: Optional[int] = None
    """Pin a specific instruction instead of sampling one."""

    ckpt_prefix: str = "clsdp"
    """Checkpoint family to load: checkpoints/{task}_{ckpt_prefix}_Agent{i}_{data_num}/.
    Use 'clsdpdet' for the deterministic-latent variant, 'clsdpfm' for flow matching."""

    siglip_model: str = DEFAULT_MODEL_NAME
    siglip_pool_grid: int = 14

    latent_sample: Optional[bool] = None
    """Override the checkpoint's latent_sample. True marginalises over z (Eq. 12)."""

    num_inference_steps: Optional[int] = None
    """Override the checkpoint's sampler step count. The flow-matching expert is trained
    once and evaluated at several step counts, so this makes the sweep a pure
    inference-time experiment rather than a retraining one."""

    timing_json: Optional[str] = None
    """Write per-episode sampler latency here. eval_cls_sweep.sh aggregates these."""


def instruction_task_name(task_name):
    return task_name[:-3] if task_name.endswith("-rf") else task_name


def load_instruction(env_id, split, index, rng):
    path = os.path.join(INSTRUCTION_DIR, f"{instruction_task_name(env_id)}.json")
    with open(path) as f:
        bank = json.load(f)
    pool = bank[split]
    if index is None:
        index = int(rng.integers(len(pool)))
    return pool[index % len(pool)]


def get_policy(checkpoint, output_dir, device):
    payload = torch.load(open("./" + checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]

    # The Stage 1 prior weights live inside this checkpoint, so the workspace must not go
    # looking for the Stage 1 file (which may not exist on an eval-only machine).
    if "contextualizer_ckpt" in cfg:
        cfg.contextualizer_ckpt = None

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    policy.to(torch.device(device))
    policy.eval()
    return policy


class SamplerStats:
    """Wall-clock and denoiser-call accounting for one agent's policy.

    The whole point of the flow-matching variant is latency, and nothing in this repo
    measured it before. Two numbers are kept deliberately separate:

      * `ms_per_action` -- how long one policy call takes, which is what the sampler
        change actually moves.
      * `denoiser_calls_per_action` -- how many U-Net forwards that cost, which is the
        number the swap is supposed to cut from 100 to a handful.

    Episode wall-clock is a third, much less flattering number: `predict_action` runs once
    per macro-cycle and the following 6 executed steps are TOPP smoothing and simulator
    substeps that this change does not touch.
    """

    def __init__(self):
        self.actions = 0
        self.denoiser_calls = 0
        self.total_ms = 0.0

    def attach(self, policy):
        """Count U-Net forwards by wrapping the bound method, not the module."""
        inner = policy.model.forward

        def counting_forward(*args, **kwargs):
            self.denoiser_calls += 1
            return inner(*args, **kwargs)

        policy.model.forward = counting_forward

    @contextmanager
    def measure(self, device):
        # CUDA kernels are async, so an unsynchronized timer measures queueing, not work.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        try:
            yield
        finally:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            self.total_ms += (time.perf_counter() - start) * 1000.0
            self.actions += 1

    def summary(self):
        actions = max(self.actions, 1)
        return {
            "policy_calls": self.actions,
            "denoiser_calls": self.denoiser_calls,
            "ms_per_action": round(self.total_ms / actions, 3),
            "denoiser_calls_per_action": round(self.denoiser_calls / actions, 3),
            "total_ms": round(self.total_ms, 1),
        }


class CLSRunner(DPRunner):
    """DPRunner that also feeds the contextualizer its current-frame SigLIP tokens."""

    def __init__(self, output_dir, siglip, text_tokens, text_mask, **kwargs):
        super().__init__(output_dir=output_dir, **kwargs)
        self.siglip = siglip
        self.text_tokens = text_tokens
        self.text_mask = text_mask

    def get_action(self, policy, observaton=None):
        if observaton is not None:
            self.obs.append(observaton)
        obs = self.get_n_steps_obs()

        # Cast explicitly: get_model_input divides uint8 by 255 and yields float64, and
        # the prior inputs bypass the normalizer that would otherwise downcast them.
        obs_dict = dict_apply(
            dict(obs),
            lambda x: torch.from_numpy(x).to(device=policy.device, dtype=torch.float32),
        )

        with torch.no_grad():
            model_input = {
                "head_cam": obs_dict["head_cam"].unsqueeze(0),
                "agent_pos": obs_dict["agent_pos"].unsqueeze(0),
            }
            # The prior sees only the current frame, never the history.
            current_frame = obs_dict["head_cam"][-1].unsqueeze(0)
            model_input["prior_image_tokens"] = self.siglip.encode_image(
                current_frame
            ).to(device=policy.device, dtype=torch.float32)
            model_input["prior_text_tokens"] = self.text_tokens
            model_input["prior_text_mask"] = self.text_mask

            action_dict = policy.predict_action(model_input)

        action = action_dict["action"].detach().to("cpu").numpy()
        return action.squeeze(0)


class CLSDP:
    def __init__(
        self,
        task_name,
        checkpoint_num,
        data_num,
        agent_id,
        siglip,
        text_tokens,
        text_mask,
        latent_sample=None,
        ckpt_prefix="clsdp",
        num_inference_steps=None,
    ):
        checkpoint = (
            f"checkpoints/{task_name}_{ckpt_prefix}_Agent{agent_id}_{data_num}/"
            f"{checkpoint_num}.ckpt"
        )
        self.policy = get_policy(checkpoint, None, "cuda:0")
        if latent_sample is not None:
            self.policy.latent_sample = latent_sample
        if num_inference_steps is not None:
            self.policy.num_inference_steps = num_inference_steps
        self.stats = SamplerStats()
        self.stats.attach(self.policy)
        self.runner = CLSRunner(
            output_dir=None,
            siglip=siglip,
            text_tokens=text_tokens,
            text_mask=text_mask,
        )

    def update_obs(self, observation):
        self.runner.update_obs(observation)

    def get_action(self, observation=None):
        with self.stats.measure(self.policy.device):
            return self.runner.get_action(self.policy, observation)

    def get_last_obs(self):
        return self.runner.obs[-1]


def _is_success(info):
    """ManiSkill reports success as a batched tensor, not a bare bool."""
    success = info.get("success", False)
    if torch.is_tensor(success):
        return bool(success.reshape(-1)[0].item())
    if isinstance(success, np.ndarray):
        return bool(success.reshape(-1)[0])
    return bool(success)


def report_and_finish(cls_models, args, env_id, verdict, record_dir, episode_ms):
    """Emit timing, then the verdict.

    eval_cls_sweep.sh reads the *last* line of the log to decide success, so the verdict
    must stay last no matter what else gets printed here.
    """
    per_agent = [m.stats.summary() for m in cls_models]
    policy_ms = sum(a["total_ms"] for a in per_agent)
    payload = {
        "env_id": env_id,
        "seed": args.seed[0] if args.seed else None,
        "ckpt_prefix": args.ckpt_prefix,
        "checkpoint_num": args.checkpoint_num,
        "num_inference_steps": cls_models[0].policy.num_inference_steps,
        "solver": getattr(
            getattr(cls_models[0].policy, "transport", None), "solver", "ddpm"
        ),
        "success": verdict == "success",
        # Kept apart on purpose: the sampler swap moves policy_ms, while episode_ms is
        # dominated by TOPP smoothing and simulator substeps that it does not touch.
        "policy_ms_total": round(policy_ms, 1),
        "episode_ms_total": round(episode_ms, 1),
        "policy_fraction_of_episode": (
            round(policy_ms / episode_ms, 4) if episode_ms > 0 else None
        ),
        "per_agent": per_agent,
    }
    mean_ms = sum(a["ms_per_action"] for a in per_agent) / len(per_agent)
    mean_calls = sum(a["denoiser_calls_per_action"] for a in per_agent) / len(per_agent)
    print(
        f"timing: {mean_ms:.1f} ms/action, {mean_calls:.1f} denoiser calls/action, "
        f"policy {payload['policy_ms_total']:.0f} ms of {payload['episode_ms_total']:.0f} "
        f"ms episode"
    )
    if args.timing_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.timing_json)), exist_ok=True)
        with open(args.timing_json, "w") as f:
            json.dump(payload, f, indent=2)

    if record_dir:
        print(f"Saving video to {record_dir}")
    print(verdict)


def get_model_input(observation, agent_pos, agent_id):
    camera_name = "head_camera_agent" + str(agent_id)
    head_cam = (
        np.moveaxis(
            observation["sensor_data"][camera_name]["rgb"].squeeze(0).numpy(), -1, 0
        ).astype(np.float32)
        / 255.0
    )
    return dict(head_cam=head_cam, agent_pos=np.asarray(agent_pos, dtype=np.float32))


def main(args: Args):
    np.set_printoptions(suppress=True, precision=5)
    verbose = not args.quiet
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    if args.seed is not None:
        np.random.seed(args.seed[0])

    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in [
        "sensor_data",
        "rgb",
        "rgbd",
        "depth",
        "point_cloud",
    ]:
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False

    env_id = args.env_id
    if env_id == "":
        with open(args.config, "r") as f:
            env_id = yaml.safe_load(f)["task_name"] + "-rf"

    env_kwargs = dict(
        config=args.config,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        enable_shadow=True,
        parallel_in_single_scene=parallel_in_single_scene,
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))

    env: BaseEnv = gym.make(env_id, **env_kwargs)

    record_dir = (
        args.record_dir
        + "/"
        + str(args.seed)
        + "_"
        + str(args.data_num)
        + "_"
        + str(args.checkpoint_num)
    )
    if record_dir:
        record_dir = record_dir.format(env_id=env_id)
        env = RecordEpisodeMA(
            env,
            record_dir,
            info_on_video=False,
            save_trajectory=False,
            max_steps_per_video=30000,
        )

    raw_obs, _ = env.reset(seed=args.seed[0])
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=verbose,
        base_pose=[agent.robot.pose for agent in env.agent.agents],
        visualize_target_grasp_pose=verbose,
        print_env_info=False,
        is_multi_agent=True,
    )
    agent_num = planner.agent_num

    # One instruction per episode, shared by every agent (paper Sec. V-B.3).
    rng = np.random.default_rng(args.seed[0])
    instruction = load_instruction(
        env_id, args.instruction_split, args.instruction_index, rng
    )
    print(f'instruction: "{instruction}"')

    siglip = SigLIPFeatureExtractor(
        model_name=args.siglip_model,
        device="cuda:0",
        pool_grid=args.siglip_pool_grid,
    )
    text_tokens, text_mask = siglip.encode_text([instruction])
    text_tokens = text_tokens.to(dtype=torch.float32)
    text_mask = text_mask.to(dtype=torch.float32)

    cls_models = [
        CLSDP(
            env_id,
            args.checkpoint_num,
            args.data_num,
            agent_id,
            siglip,
            text_tokens,
            text_mask,
            latent_sample=args.latent_sample,
            ckpt_prefix=args.ckpt_prefix,
            num_inference_steps=args.num_inference_steps,
        )
        for agent_id in range(agent_num)
    ]
    head = cls_models[0].policy
    print(
        f"sampler: {type(head).__name__} steps={head.num_inference_steps} "
        f"solver={getattr(getattr(head, 'transport', None), 'solver', 'ddpm')}"
    )

    if args.seed is not None and env.action_space is not None:
        env.action_space.seed(args.seed[0])
    if args.render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        env.render()

    for agent_id in range(agent_num):
        initial_qpos = raw_obs["agent"][f"panda-{agent_id}"]["qpos"].squeeze(0)[:-2].numpy()
        initial_qpos = np.append(initial_qpos, planner.gripper_state[agent_id])
        cls_models[agent_id].update_obs(
            get_model_input(raw_obs, initial_qpos, agent_id)
        )

    info = {}
    step_count = 0
    episode_start = time.perf_counter()
    while True:
        if verbose:
            print("Iteration:", step_count)
        step_count += 1
        if step_count > args.max_steps:
            break

        action_dict = defaultdict(list)
        action_step_dict = defaultdict(list)

        # Each agent acts on its own local observation only: no shared views, no state
        # exchange, no communication. Coordination comes entirely through z.
        for agent_id in range(agent_num):
            action_list = cls_models[agent_id].get_action()
            for i in range(6):
                now_action = action_list[i]
                raw_obs = env.get_obs()
                if i == 0:
                    current_qpos = (
                        raw_obs["agent"][f"panda-{agent_id}"]["qpos"]
                        .squeeze(0)[:-2]
                        .numpy()
                    )
                else:
                    current_qpos = action_list[i - 1][:-1]
                path = np.vstack((current_qpos, now_action[:-1]))
                try:
                    times, position, right_vel, acc, duration = planner.planner[
                        agent_id
                    ].TOPP(path, 0.05, verbose=True)
                except Exception as e:
                    print(f"Error occurred: {e}")
                    action_dict[f"panda-{agent_id}"].append(
                        np.hstack([current_qpos, now_action[-1]])
                    )
                    action_step_dict[f"panda-{agent_id}"].append(1)
                    continue
                n_step = position.shape[0]
                action_step_dict[f"panda-{agent_id}"].append(n_step)
                gripper_state = now_action[-1]
                if n_step == 0:
                    action_dict[f"panda-{agent_id}"].append(
                        np.hstack([current_qpos, gripper_state])
                    )
                for j in range(n_step):
                    action_dict[f"panda-{agent_id}"].append(
                        np.hstack([position[j], gripper_state])
                    )

        start_idx = [0 for _ in range(agent_num)]
        for i in range(6):
            max_step = 0
            for agent_id in range(agent_num):
                max_step = max(max_step, action_step_dict[f"panda-{agent_id}"][i])
            for j in range(max_step):
                true_action = dict()
                for agent_id in range(agent_num):
                    now_step = min(j, action_step_dict[f"panda-{agent_id}"][i] - 1)
                    true_action[f"panda-{agent_id}"] = action_dict[
                        f"panda-{agent_id}"
                    ][start_idx[agent_id] + now_step]
                observation, reward, terminated, truncated, info = env.step(true_action)
                if verbose:
                    env.render_human()

            for agent_id in range(agent_num):
                start_idx[agent_id] += action_step_dict[f"panda-{agent_id}"][i]
                if action_step_dict[f"panda-{agent_id}"][i] == 0:
                    continue
                cls_models[agent_id].update_obs(
                    get_model_input(
                        observation, true_action[f"panda-{agent_id}"], agent_id
                    )
                )

        if verbose:
            print("info", info)
        if args.render_mode is not None:
            env.render()
        if _is_success(info):
            env.close()
            report_and_finish(
                cls_models,
                args,
                env_id,
                "success",
                record_dir,
                (time.perf_counter() - episode_start) * 1000.0,
            )
            return

    env.close()
    report_and_finish(
        cls_models,
        args,
        env_id,
        "failed",
        record_dir,
        (time.perf_counter() - episode_start) * 1000.0,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
