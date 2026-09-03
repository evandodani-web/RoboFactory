"""Stage 2a: action chunk autoencoder training.

This workspace trains `ActionChunkAutoencoder` to reconstruct normalized action
chunks of shape (B, horizon, action_dim) from their learned latent tokens
(B, n_tokens, latent_dim).

It emits a small reconstruction gate:
  * recon_mse vs batch-mean baseline
  * prints PASS/FAIL so the experiment can be stopped early if the autoencoder
    fails to capture the action manifold
"""

from __future__ import annotations

import os
import random
import pathlib
from dataclasses import dataclass
from typing import Dict

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf

from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import optimizer_to
from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.sampler import SequenceSampler

from diffusion_policy.model.cls.action_autoencoder import ActionChunkAutoencoder
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


class ActionAEWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = int(cfg.training.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.action_autoencoder: ActionChunkAutoencoder = hydra.utils.instantiate(
            cfg.action_autoencoder
        )
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.action_autoencoder.parameters()
        )

        self.global_step = 0
        self.epoch = 0

    def _load_actions(self):
        action_key = f"action_agent{self.cfg.agent_id}"
        state_key = f"state_agent{self.cfg.agent_id}"
        # Load only what we need:
        #   * actions for training data
        #   * states (which match actions in this repo) only for normalizer stats
        replay = ReplayBuffer.copy_from_path(
            self.cfg.zarr_path,
            keys=[action_key, state_key],
            backend="numpy",
        )
        return replay

    def _make_sampler(self, replay, *, episode_mask: np.ndarray):
        horizon = int(self.cfg.horizon)
        pad = horizon - 1
        return SequenceSampler(
            replay_buffer=replay,
            sequence_length=horizon,
            pad_before=pad,
            pad_after=pad,
            episode_mask=episode_mask,
        )

    def _fit_normalizer(self, replay):
        normalizer = LinearNormalizer()
        action_key = f"action_agent{self.cfg.agent_id}"
        state_key = f"state_agent{self.cfg.agent_id}"
        normalizer.fit(
            data={
                "action": replay[action_key],
                "agent_pos": replay[state_key],
            },
            last_n_dims=1,
            mode=self.cfg.normalizer.mode,
        )
        return normalizer

    def _make_dataloader(self, sampler, *, batch_size: int, device: torch.device):
        # Minimal loader: rely on SequenceSampler's fast padding logic.
        # We sample random indices from the precomputed sampler index list.
        return _RandomSequenceLoader(
            sampler=sampler, batch_size=batch_size, device=device, action_key=None
        )

    def run(self):
        cfg = OmegaConf.to_container(self.cfg, resolve=True)
        cfg = OmegaConf.create(cfg)

        if cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {latest_ckpt_path}")
                self.load_checkpoint(path=latest_ckpt_path)

        device = torch.device(cfg.training.device)

        replay = self._load_actions()
        n_episodes = int(replay.episode_ends.shape[0])
        val_ratio = float(cfg.training.val_ratio)
        val_mask = get_val_mask(n_episodes=n_episodes, val_ratio=val_ratio, seed=cfg.training.seed)
        train_mask = ~val_mask

        train_sampler = self._make_sampler(replay, episode_mask=train_mask)
        val_sampler = self._make_sampler(replay, episode_mask=val_mask)

        self.normalizer = self._fit_normalizer(replay)
        optimizer_to(self.optimizer, device)
        self.action_autoencoder.to(device)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 5

        batch_size = int(cfg.dataloader.batch_size)
        lr = float(cfg.optimizer.lr)
        # lr is part of cfg.optimizer instantiation already; keep it here for logging.
        _ = lr

        train_loader = _RandomSequenceLoader(
            sampler=train_sampler,
            batch_size=batch_size,
            device=device,
            action_key=f"action_agent{cfg.agent_id}",
        )
        val_loader = _RandomSequenceLoader(
            sampler=val_sampler,
            batch_size=batch_size,
            device=device,
            action_key=f"action_agent{cfg.agent_id}",
        )

        if len(train_loader) == 0:
            raise RuntimeError(
                "no training batches: check zarr_path and episode counts"
            )

        total_steps = 0
        os.makedirs(self.output_dir, exist_ok=True)
        log_path = os.path.join(self.output_dir, "logs.json.txt")

        with JsonLogger(log_path) as json_logger:
            for epoch in range(int(cfg.training.num_epochs)):
                self.epoch = epoch
                self.action_autoencoder.train()

                train_losses = []
                for step_idx, batch in enumerate(train_loader):
                    if cfg.training.max_train_steps is not None and step_idx >= cfg.training.max_train_steps:
                        break

                    actions = batch["action"]  # (B, horizon, action_dim), already on device
                    # Train in normalized space to match what Stage 2 expects.
                    actions_norm = self.normalizer["action"].normalize(actions)

                    recon_norm = self.action_autoencoder(actions_norm)
                    recon_loss = torch.mean((recon_norm - actions_norm) ** 2)

                    self.optimizer.zero_grad()
                    recon_loss.backward()
                    self.optimizer.step()

                    train_losses.append(float(recon_loss.detach().cpu()))
                    self.global_step += 1
                    json_logger.log(
                        {
                            "train_loss": float(recon_loss.detach().cpu()),
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                        }
                    )
                    total_steps += 1

                # Validation + gate
                self.action_autoencoder.eval()
                with torch.no_grad():
                    recon_mses = []
                    baseline_mses = []
                    for val_step_idx, batch in enumerate(val_loader):
                        if cfg.training.max_val_steps is not None and val_step_idx >= cfg.training.max_val_steps:
                            break
                        actions = batch["action"]
                        actions_norm = self.normalizer["action"].normalize(actions)

                        recon = self.action_autoencoder(actions_norm)
                        recon_mse = torch.mean((recon - actions_norm) ** 2).item()
                        baseline = actions_norm.mean(dim=0, keepdim=True)
                        baseline_mse = torch.mean((baseline - actions_norm) ** 2).item()
                        recon_mses.append(recon_mse)
                        baseline_mses.append(baseline_mse)

                recon_mean = float(np.mean(recon_mses))
                baseline_mean = float(np.mean(baseline_mses))

                gate_pass = recon_mean < baseline_mean
                print(
                    f"[Action AE gate] recon_mse={recon_mean:.6f} baseline_mse={baseline_mean:.6f} -> "
                    + ("PASS" if gate_pass else "FAIL")
                )

                if (self.epoch + 1) % int(cfg.training.checkpoint_every) == 0:
                    self.save_checkpoint(
                        f"checkpoints/{cfg.checkpoint_name}/{self.epoch + 1}.ckpt"
                    )


class _RandomSequenceLoader:
    """Small iterator for SequenceSampler windows (actions only)."""

    def __init__(self, *, sampler: SequenceSampler, batch_size: int, device: torch.device, action_key: str):
        self.sampler = sampler
        self.batch_size = batch_size
        self.device = device
        self.action_key = action_key if action_key is not None else self.sampler.keys[0]

        self._indices = np.arange(len(self.sampler.indices), dtype=np.int64)
        self._n = len(self._indices)

    def __len__(self):
        # Approximate "num batches" for the iterator contract.
        return max(1, self._n // self.batch_size)

    def __iter__(self):
        # Shuffle each epoch by permuting sampler indices.
        perm = np.random.permutation(self._indices)
        n_batches = len(self)
        for bi in range(n_batches):
            batch_indices = perm[bi * self.batch_size : (bi + 1) * self.batch_size]
            if len(batch_indices) == 0:
                continue

            # Collect samples by materializing each sequence.
            # SequenceSampler is already optimized for padding; this keeps the code simple.
            actions = []
            for idx in batch_indices:
                sample = self.sampler.sample_sequence(int(idx))
                actions.append(sample[self.action_key])  # numpy (horizon, action_dim)
            actions = np.stack(actions, axis=0)
            actions_t = torch.from_numpy(actions).to(device=self.device, dtype=torch.float32)
            yield {"action": actions_t}

