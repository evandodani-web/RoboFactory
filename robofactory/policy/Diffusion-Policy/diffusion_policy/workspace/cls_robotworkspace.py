"""CLS-DP Stage 2 training loop: the latent-conditioned action-expert.

Mirrors RobotWorkspace, with three differences:

  1. The frozen contextualizer prior is loaded from the Stage 1 checkpoint *before* the EMA
     copy is taken, so both models start from identical prior weights.
  2. The optimizer only ever sees parameters with requires_grad=True, so the frozen prior
     never reaches AdamW.
  3. Batches carry the prior's SigLIP token inputs alongside obs/action.
"""

import copy
import os
import pathlib
import random

import dill
import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf

from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import optimizer_to
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.policy.cls_diffusion_unet_image_policy import (
    CLSDiffusionUnetImagePolicy,
    PRIOR_KEYS,
)
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.workspace.robotworkspace import create_dataloader

OmegaConf.register_new_resolver("eval", eval, replace=True)

PRIOR_PREFIX = "prior_net."


class CLSRobotWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: CLSDiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        contextualizer_ckpt = cfg.get("contextualizer_ckpt", None)
        if contextualizer_ckpt:
            self._load_prior_weights(contextualizer_ckpt)
        else:
            # Expected when reconstructing a workspace from a trained Stage 2 checkpoint,
            # where load_payload supplies prior_net. During training it means the prior is
            # random, which would make this an expensive w/o-CLS ablation.
            print(
                "contextualizer_ckpt is null: prior_net keeps its random init unless a "
                "checkpoint payload is loaded on top."
            )

        # Freeze after loading; the EMA copy below then inherits frozen weights.
        self.model.prior_net.requires_grad_(False)
        self.model.prior_net.eval()

        self.ema_model: CLSDiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=trainable)

        self.global_step = 0
        self.epoch = 0

    def _load_prior_weights(self, checkpoint_path):
        """Pull `prior_net.*` out of a Stage 1 Contextualizer checkpoint."""
        path = pathlib.Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"contextualizer checkpoint not found: {path}. Train Stage 1 first with "
                "bash policy/Diffusion-Policy/train_cls_stage1.sh ..."
            )
        payload = torch.load(path.open("rb"), pickle_module=dill, map_location="cpu")
        contextualizer_state = payload["state_dicts"]["model"]
        prior_state = {
            key[len(PRIOR_PREFIX) :]: value
            for key, value in contextualizer_state.items()
            if key.startswith(PRIOR_PREFIX)
        }
        if not prior_state:
            raise RuntimeError(
                f"no prior_net.* weights found in {path}; is this a Stage 1 checkpoint?"
            )
        missing, unexpected = self.model.prior_net.load_state_dict(
            prior_state, strict=False
        )
        if missing or unexpected:
            raise RuntimeError(
                f"prior_net weights do not match the configured PriorNet.\n"
                f"missing={sorted(missing)}\nunexpected={sorted(unexpected)}\n"
                "The Stage 1 and Stage 2 prior_net configs must be identical."
            )
        print(f"loaded frozen contextualizer prior from {path}")

    @staticmethod
    def _predict_inputs(batch):
        """Observation dict plus the prior inputs, for the sampling diagnostic."""
        obs_dict = dict(batch["obs"])
        for key in PRIOR_KEYS:
            if key in batch:
                obs_dict[key] = batch[key]
        return obs_dict

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {latest_ckpt_path}")
                self.load_checkpoint(path=latest_ckpt_path)

        dataset = hydra.utils.instantiate(cfg.task.dataset)
        train_dataloader = create_dataloader(dataset, **cfg.dataloader)
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = create_dataloader(val_dataset, **cfg.val_dataloader)

        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for _ in range(cfg.training.num_epochs):
                step_log = dict()

                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dataset.postprocess(batch, device)
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        if cfg.training.use_ema:
                            ema.step(self.model)

                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) and batch_idx >= (
                            cfg.training.max_train_steps - 1
                        ):
                            break

                step_log["train_loss"] = float(np.mean(train_losses))

                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                if (self.epoch % cfg.training.val_every) == 0 and len(val_dataloader) > 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dataset.postprocess(batch, device)
                                val_losses.append(
                                    self.model.compute_loss(batch).item()
                                )
                                if (
                                    cfg.training.max_val_steps is not None
                                ) and batch_idx >= (cfg.training.max_val_steps - 1):
                                    break
                        if val_losses:
                            step_log["val_loss"] = float(np.mean(val_losses))

                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        batch = train_sampling_batch
                        gt_action = batch["action"]
                        result = policy.predict_action(self._predict_inputs(batch))
                        mse = torch.nn.functional.mse_loss(
                            result["action_pred"], gt_action
                        )
                        step_log["train_action_mse_error"] = mse.item()
                        del result, mse

                if ((self.epoch + 1) % cfg.training.checkpoint_every) == 0:
                    self.save_checkpoint(
                        f"checkpoints/{cfg.checkpoint_name}/{self.epoch + 1}.ckpt"
                    )

                policy.train()
                # compute_loss must never train the frozen prior back into train mode.
                self.model.prior_net.eval()

                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = CLSRobotWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
