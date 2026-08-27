"""Re-package per-agent .pkl episodes into a single time-aligned multi-agent .zarr.

CLS-DP's posterior needs the privileged future trajectories of *all* agents,
s^{1:N}_{t+1:t+H}, which the existing per-agent zarr files cannot express. This script
produces one zarr per task holding every agent's camera, state and action on a shared
time axis, plus the per-episode instruction id.

The per-agent pkl directories written by parse_h5_to_pkl_multi.py are already step
aligned: that script loops over the same episode index and step index for every agent,
with the length taken once from panda-0. So this is a pure re-packaging job -- no new
data collection is required.

Note that `joint_action` and `endpose` are written from the same source array upstream,
so state == action == the 8-D commanded joint vector (7 arm joints + 1 gripper).

The `{task}_global` directory produced by parse_h5_to_pkl_multi.py is deliberately
ignored: CLS-DP forbids shared global views.

Usage:
    python script/parse_pkl_to_zarr_multi.py --task_name LiftBarrier --load_num 150 --agent_num 2
"""

import argparse
import json
import os
import pickle
import shutil

import numpy as np
import zarr

_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTRUCTION_DIR = os.path.join(_PACKAGE_DIR, "configs", "instructions")

IMAGE_CHUNK = 100
VECTOR_CHUNK = 1000


def instruction_task_name(task_name):
    """Env ids carry a `-rf` suffix; the instruction bank is keyed on the bare name."""
    return task_name[:-3] if task_name.endswith("-rf") else task_name


def load_instruction_bank(task_name, split):
    path = os.path.join(INSTRUCTION_DIR, f"{instruction_task_name(task_name)}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"instruction bank not found at {path}. Run: "
            f"python script/generate_instructions.py "
            f"--task_name {instruction_task_name(task_name)}"
        )
    with open(path) as f:
        bank = json.load(f)
    return bank[split]


def episode_length(load_dir, episode_idx):
    """Number of consecutive {j}.pkl files in an episode directory."""
    episode_dir = os.path.join(load_dir, f"episode{episode_idx}")
    if not os.path.isdir(episode_dir):
        return 0
    n = 0
    while os.path.exists(os.path.join(episode_dir, f"{n}.pkl")):
        n += 1
    return n


def read_step(load_dir, episode_idx, step_idx):
    path = os.path.join(load_dir, f"episode{episode_idx}", f"{step_idx}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--load_num", type=int, required=True, help="Number of episodes")
    parser.add_argument("--agent_num", type=int, required=True)
    parser.add_argument("--pkl_dir", type=str, default="data/pkl_data")
    parser.add_argument("--zarr_dir", type=str, default="data/zarr_data")
    parser.add_argument(
        "--instruction_split",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Which half of the instruction bank to sample from",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    task_name = args.task_name
    n_agents = args.agent_num
    n_episodes = args.load_num

    instructions = load_instruction_bank(task_name, args.instruction_split)
    rng = np.random.default_rng(args.seed)

    agent_dirs = [
        os.path.join(args.pkl_dir, f"{task_name}_Agent{i}") for i in range(n_agents)
    ]
    for d in agent_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(
                f"missing {d}. Run parse_h5_to_pkl_multi.py first."
            )

    save_dir = os.path.join(args.zarr_dir, f"{task_name}_multi_{n_episodes}.zarr")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    root = zarr.group(save_dir)
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    # Probe the first step to learn shapes without loading everything into memory.
    probe = read_step(agent_dirs[0], 0, 0)
    probe_rgb = probe["observation"]["head_camera"]["rgb"]
    img_h, img_w = probe_rgb.shape[0], probe_rgb.shape[1]
    state_dim = int(np.asarray(probe["joint_action"]).shape[-1])
    print(f"image {img_h}x{img_w}, state dim {state_dim}, {n_agents} agents")

    cam_arrays, state_arrays, action_arrays = [], [], []
    for i in range(n_agents):
        cam_arrays.append(
            data_group.zeros(
                f"head_camera_agent{i}",
                shape=(0, 3, img_h, img_w),
                chunks=(IMAGE_CHUNK, 3, img_h, img_w),
                dtype="uint8",
                compressor=compressor,
            )
        )
        state_arrays.append(
            data_group.zeros(
                f"state_agent{i}",
                shape=(0, state_dim),
                chunks=(VECTOR_CHUNK, state_dim),
                dtype="float32",
                compressor=compressor,
            )
        )
        action_arrays.append(
            data_group.zeros(
                f"action_agent{i}",
                shape=(0, state_dim),
                chunks=(VECTOR_CHUNK, state_dim),
                dtype="float32",
                compressor=compressor,
            )
        )
    instruction_array = data_group.zeros(
        "instruction_id",
        shape=(0,),
        chunks=(VECTOR_CHUNK,),
        dtype="int64",
        compressor=compressor,
    )

    episode_ends = []
    episode_instruction_ids = []
    total = 0

    for ep in range(n_episodes):
        lengths = [episode_length(d, ep) for d in agent_dirs]
        if min(lengths) == 0:
            print(f"episode {ep}: missing data, stopping early")
            break
        if len(set(lengths)) != 1:
            print(f"episode {ep}: agent lengths differ {lengths}, truncating to min")
        ep_len = min(lengths)

        # Buffer a single episode at a time so peak memory stays bounded.
        cam_block = [np.empty((ep_len, 3, img_h, img_w), dtype=np.uint8) for _ in range(n_agents)]
        state_block = [np.empty((ep_len, state_dim), dtype=np.float32) for _ in range(n_agents)]

        for i in range(n_agents):
            for j in range(ep_len):
                step = read_step(agent_dirs[i], ep, j)
                rgb = np.asarray(step["observation"]["head_camera"]["rgb"])
                cam_block[i][j] = np.moveaxis(rgb, -1, 0)
                state_block[i][j] = np.asarray(step["joint_action"], dtype=np.float32)

        instruction_id = int(rng.integers(len(instructions)))
        for i in range(n_agents):
            cam_arrays[i].append(cam_block[i])
            state_arrays[i].append(state_block[i])
            # state and action are the same commanded joint vector upstream; both are
            # stored so downstream code can stay explicit about which one it means.
            action_arrays[i].append(state_block[i])
        instruction_array.append(np.full((ep_len,), instruction_id, dtype=np.int64))

        total += ep_len
        episode_ends.append(total)
        episode_instruction_ids.append(instruction_id)
        print(f"episode {ep + 1}/{n_episodes}: {ep_len} steps (total {total})", end="\r")

    print()
    meta_group.create_dataset(
        "episode_ends",
        data=np.array(episode_ends, dtype=np.int64),
        chunks=(VECTOR_CHUNK,),
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )

    root.attrs["task_name"] = task_name
    root.attrs["instruction_task"] = instruction_task_name(task_name)
    root.attrs["n_agents"] = n_agents
    root.attrs["state_dim"] = state_dim
    root.attrs["instruction_split"] = args.instruction_split
    root.attrs["n_instructions"] = len(instructions)
    root.attrs["episode_instruction_ids"] = episode_instruction_ids

    print(f"wrote {len(episode_ends)} episodes / {total} steps -> {save_dir}")
    print("next: python script/precompute_siglip_features.py --zarr_path " + save_dir)


if __name__ == "__main__":
    main()
