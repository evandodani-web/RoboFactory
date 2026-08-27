"""Dataset over the time-aligned multi-agent zarr produced by parse_pkl_to_zarr_multi.py.

Serves both CLS-DP stages from one window definition. With n_obs_steps=3 and
n_future_states=8 the sampled window has length 11 and time t sits at index n_obs_steps-1:

    window index:   0     1     2     3     4    ...    10
    absolute time: t-2   t-1    t    t+1   t+2   ...   t+8

    obs history        [0 : 3]     O_t^i, S_t^i
    action target      [0 : 8]     UNet horizon (executed slice is [2:8])
    privileged future  [3 : 11]    s^{1:N}_{t+1:t+8}
    prior frame          [2]       o_t^i, current frame only

Stage 1 loads every agent's states plus this agent's cached SigLIP features; Stage 2 loads
this agent's camera, states and actions. Keeping the key set stage-dependent matters: Stage
1 runs at batch size 512, and pulling raw images it never uses would cost gigabytes of
buffer for nothing.

Structure deliberately mirrors RobotImageDataset (preallocated buffers plus the numba batch
sampler) so it plugs into the same create_dataloader path.
"""

import copy
import os
from typing import Dict

import numpy as np
import torch
import zarr

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.sampler import (
    SequenceSampler,
    downsample_mask,
    get_val_mask,
)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.dataset.robot_image_dataset import batch_sample_sequence
from diffusion_policy.model.common.normalizer import LinearNormalizer

STAGE_CONTEXTUALIZER = 1
STAGE_ACTION_EXPERT = 2


def default_text_cache_path(zarr_path):
    return zarr_path.rstrip("/") + "_text.npz"


class MultiAgentImageDataset(BaseImageDataset):
    def __init__(
        self,
        zarr_path: str,
        agent_id: int = 0,
        stage: int = STAGE_ACTION_EXPERT,
        n_agents: int = None,
        horizon: int = 8,
        n_obs_steps: int = 3,
        n_future_states: int = 8,
        seed: int = 42,
        val_ratio: float = 0.02,
        batch_size: int = 64,
        max_train_episodes: int = None,
        text_cache_path: str = None,
    ):
        super().__init__()

        root = zarr.open(zarr_path, mode="r")
        if n_agents is None:
            n_agents = int(root.attrs["n_agents"])
        instruction_split = root.attrs.get("instruction_split", "train")

        if stage not in (STAGE_CONTEXTUALIZER, STAGE_ACTION_EXPERT):
            raise ValueError(f"stage must be 1 or 2, got {stage}")
        if agent_id >= n_agents:
            raise ValueError(f"agent_id {agent_id} out of range for {n_agents} agents")

        sequence_length = n_obs_steps + n_future_states
        if horizon > sequence_length:
            raise ValueError(
                f"horizon {horizon} exceeds the sampled window {sequence_length}"
            )

        self.agent_id = agent_id
        self.n_agents = n_agents
        self.stage = stage
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_future_states = n_future_states
        self.batch_size = batch_size

        # ------------------------------------------------------------ keys
        siglip_key = f"siglip_img_agent{agent_id}"
        if siglip_key not in root["data"]:
            raise KeyError(
                f"{siglip_key} missing from {zarr_path}. Run: "
                f"python script/precompute_siglip_features.py --zarr_path {zarr_path}"
            )

        keys = [siglip_key, "instruction_id"]
        if stage == STAGE_CONTEXTUALIZER:
            keys += [f"state_agent{j}" for j in range(n_agents)]
        else:
            keys += [
                f"head_camera_agent{agent_id}",
                f"state_agent{agent_id}",
                f"action_agent{agent_id}",
            ]

        self.siglip_key = siglip_key
        self.camera_key = f"head_camera_agent{agent_id}"
        self.state_key = f"state_agent{agent_id}"
        self.action_key = f"action_agent{agent_id}"

        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=keys)

        # The SigLIP cache is stored as float16 to keep it off disk cheaply, but the numba
        # batch sampler compiles in nopython mode and has no float16 data model
        # (`NotImplementedError: float16`). Widen to float32 once, here, rather than paying
        # for it on disk.
        for key, array in list(self.replay_buffer.data.items()):
            if array.dtype == np.float16:
                self.replay_buffer.data[key] = array.astype(np.float32)

        # ------------------------------------------------------------ splits
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = downsample_mask(
            mask=~val_mask, max_n=max_train_episodes, seed=seed
        )

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=sequence_length,
            pad_before=n_obs_steps - 1,
            pad_after=sequence_length - n_obs_steps,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.sequence_length = sequence_length

        # ------------------------------------------------------------ text
        if text_cache_path is None:
            text_cache_path = default_text_cache_path(zarr_path)
        if not os.path.exists(text_cache_path):
            raise FileNotFoundError(
                f"text feature cache not found at {text_cache_path}. Run: "
                f"python script/precompute_siglip_features.py --zarr_path {zarr_path}"
            )
        cache = np.load(text_cache_path)
        self.text_tokens = torch.from_numpy(
            cache[f"{instruction_split}_tokens"].astype(np.float32)
        )
        self.text_mask = torch.from_numpy(
            cache[f"{instruction_split}_mask"].astype(np.float32)
        )

        # ------------------------------------------------------------ buffers
        # The torch views must share storage with the numpy buffers that the numba batch
        # sampler writes into, so these cannot be pinned: Tensor.pin_memory() returns a
        # *copy* in pinned memory and would silently break that aliasing. (RobotImageDataset
        # calls it and discards the result, which is a no-op.)
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.replay_buffer.items()
        }
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}

    # ---------------------------------------------------------------- plumbing

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.sequence_length,
            pad_before=self.n_obs_steps - 1,
            pad_after=self.sequence_length - self.n_obs_steps,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            raise NotImplementedError
        if isinstance(idx, (int, np.integer)):
            sample = self.sampler.sample_sequence(int(idx))
            return dict_apply(sample, torch.from_numpy)
        if isinstance(idx, np.ndarray):
            assert len(idx) == self.batch_size
            for k, v in self.replay_buffer.items():
                batch_sample_sequence(
                    self.buffers[k],
                    v,
                    self.sampler.indices,
                    idx,
                    self.sampler.sequence_length,
                )
            return self.buffers_torch
        raise ValueError(idx)

    # -------------------------------------------------------------- normalizer

    def get_normalizer(self, mode="limits", **kwargs):
        normalizer = LinearNormalizer()

        if self.stage == STAGE_CONTEXTUALIZER:
            # One shared state normalizer across agents: the decoder reconstructs every
            # agent into a single tensor, so they must live on a common scale.
            all_states = np.concatenate(
                [
                    self.replay_buffer[f"state_agent{j}"][:]
                    for j in range(self.n_agents)
                ],
                axis=0,
            )
            normalizer.fit(
                data={"state": all_states}, last_n_dims=1, mode=mode, **kwargs
            )
        else:
            normalizer.fit(
                data={
                    "action": self.replay_buffer[self.action_key],
                    "agent_pos": self.replay_buffer[self.state_key],
                },
                last_n_dims=1,
                mode=mode,
                **kwargs,
            )
            normalizer["head_cam"] = get_image_range_normalizer()

        return normalizer

    # ------------------------------------------------------------- postprocess

    def _prior_inputs(self, samples, device):
        """Extract the current-frame image tokens and the episode's text tokens."""
        current = self.n_obs_steps - 1

        image_tokens = (
            samples[self.siglip_key][:, current].to(device, non_blocking=True).float()
        )
        instruction_id = samples["instruction_id"][:, current].long()

        text_tokens = self.text_tokens[instruction_id].to(device, non_blocking=True)
        text_mask = self.text_mask[instruction_id].to(device, non_blocking=True)

        return {
            "prior_image_tokens": image_tokens,
            "prior_text_tokens": text_tokens,
            "prior_text_mask": text_mask,
        }

    def postprocess(self, samples, device):
        current = self.n_obs_steps - 1
        batch = self._prior_inputs(samples, device)

        if self.stage == STAGE_CONTEXTUALIZER:
            batch["own_state"] = (
                samples[self.state_key][:, current].to(device, non_blocking=True)
            )
            future = [
                samples[f"state_agent{j}"][
                    :, self.n_obs_steps : self.n_obs_steps + self.n_future_states
                ]
                for j in range(self.n_agents)
            ]
            batch["future_states"] = torch.stack(future, dim=1).to(
                device, non_blocking=True
            )
            return batch

        head_cam = (
            samples[self.camera_key][:, : self.n_obs_steps].to(
                device, non_blocking=True
            )
            / 255.0
        )
        batch["obs"] = {
            "head_cam": head_cam,
            "agent_pos": samples[self.state_key][:, : self.n_obs_steps].to(
                device, non_blocking=True
            ),
        }
        batch["action"] = samples[self.action_key][:, : self.horizon].to(
            device, non_blocking=True
        )
        return batch
