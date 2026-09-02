"""CLS-DP Stage 1 training loop: distill privileged multi-agent dynamics into the prior.

One contextualizer is trained per agent (Eq. 5 sums over per-agent parameters), so this
workspace is launched once per agent id.

The go/no-go signal for this stage is `val_ctx_recon_others` versus
`val_ctx_recon_others_baseline`. The decoder only ever sees the agent's own current state
plus z, so if it cannot reconstruct *teammates'* futures better than a batch-mean
predictor, then z is not carrying coordination information and Stage 2 has nothing to
condition on. Watch that pair before spending GPU hours on Stage 2.
"""

import copy
import os
import pathlib
import random

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf

from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import optimizer_to
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.policy.contextualizer import Contextualizer
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.workspace.robotworkspace import create_dataloader

OmegaConf.register_new_resolver("eval", eval, replace=True)


class ContextualizerWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: Contextualizer = hydra.utils.instantiate(cfg.contextualizer)
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters()
        )

        self.global_step = 0
        self.epoch = 0

    def _beta(self, total_steps):
        """Linear warm-up of the KL weight over the first `beta_warmup_ratio` of training."""
        beta_max = self.cfg.training.beta
        ratio = self.cfg.training.beta_warmup_ratio
        if ratio <= 0:
            return beta_max
        warmup_steps = max(int(total_steps * ratio), 1)
        return beta_max * min(1.0, self.global_step / warmup_steps)

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

        self.model.set_normalizer(dataset.get_normalizer())

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        device = torch.device(cfg.training.device)
        self.model.to(device)
        optimizer_to(self.optimizer, device)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.val_every = 1
            cfg.training.checkpoint_every = 1

        total_steps = len(train_dataloader) * cfg.training.num_epochs

        if len(train_dataloader) == 0:
            raise RuntimeError(
                f"no training batches: {len(dataset)} windows at batch size "
                f"{cfg.dataloader.batch_size}. Lower the batch size or collect more data."
            )

        step_log = dict()
        # Hydra normally creates the run dir, but JsonLogger will not; create it so the
        # workspace can also be driven programmatically (mirrors save_checkpoint).
        os.makedirs(self.output_dir, exist_ok=True)
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for _ in range(cfg.training.num_epochs):
                train_losses = list()
                epoch_metrics = list()

                self.model.train()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Contextualizer epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dataset.postprocess(batch, device)
                        beta = self._beta(total_steps)

                        raw_loss, metrics = self.model.compute_loss(batch, beta=beta)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            if cfg.training.grad_clip_norm is not None:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(),
                                    cfg.training.grad_clip_norm,
                                )
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(
                            loss=raw_loss_cpu,
                            others=metrics["ctx_recon_others"],
                            refresh=False,
                        )
                        train_losses.append(raw_loss_cpu)
                        epoch_metrics.append(metrics)

                        step_log = dict(metrics)
                        step_log.update(
                            {
                                "train_loss": raw_loss_cpu,
                                "global_step": self.global_step,
                                "epoch": self.epoch,
                                "lr": lr_scheduler.get_last_lr()[0],
                            }
                        )

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) and batch_idx >= (
                            cfg.training.max_train_steps - 1
                        ):
                            break

                step_log["train_loss"] = float(np.mean(train_losses))
                for key in epoch_metrics[0] if epoch_metrics else ():
                    step_log[key] = float(np.mean([m[key] for m in epoch_metrics]))

                # ------------------------------------------------------- validation
                if (self.epoch % cfg.training.val_every) == 0 and len(val_dataloader) > 0:
                    self.model.eval()
                    val_metrics = list()
                    with torch.no_grad():
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dataset.postprocess(batch, device)
                                _, metrics = self.model.compute_loss(
                                    batch, beta=self._beta(total_steps)
                                )
                                val_metrics.append(metrics)
                                if (
                                    cfg.training.max_val_steps is not None
                                ) and batch_idx >= (cfg.training.max_val_steps - 1):
                                    break
                    if val_metrics:
                        for key in val_metrics[0]:
                            step_log[f"val_{key}"] = float(
                                np.mean([m[key] for m in val_metrics])
                            )

                # ------------------------------------------------------- checkpoint
                if ((self.epoch + 1) % cfg.training.checkpoint_every) == 0:
                    self.save_checkpoint(
                        f"checkpoints/{cfg.checkpoint_name}/{self.epoch + 1}.ckpt"
                    )

                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        self._report(step_log)

    def _report(self, step_log):
        others = step_log.get("val_ctx_recon_others", step_log.get("ctx_recon_others"))
        baseline = step_log.get(
            "val_ctx_recon_others_baseline", step_log.get("ctx_recon_others_baseline")
        )
        if others is None or baseline is None:
            return
        verdict = "PASS" if others < baseline else "FAIL"
        print(
            f"\n[Stage 1 gate] teammate reconstruction {others:.5f} vs "
            f"batch-mean baseline {baseline:.5f} -> {verdict}"
        )
        if verdict == "FAIL":
            print(
                "  The latent is not encoding teammate dynamics. Stage 2 will not "
                "outperform the w/o-CLS ablation. Check beta, the warm-up ratio, and "
                "that future_states really contains all agents."
            )

        # The number above is measured on the *combined* latent (prior + privileged
        # residual), but deployment only has the prior. When the probe is attached, report
        # both so a passing gate cannot hide an insufficient prior.
        prior_only = step_log.get(
            "val_ctx_probe_recon_others", step_log.get("ctx_probe_recon_others")
        )
        if prior_only is not None:
            gap = step_log.get("val_ctx_prior_gap", step_log.get("ctx_prior_gap"))
            prior_verdict = "PASS" if prior_only < baseline else "FAIL"
            print(
                f"[Stage 1 gate] prior-only reconstruction {prior_only:.5f} "
                f"(gap vs combined {gap:+.5f}) -> {prior_verdict}"
            )
            if prior_verdict == "FAIL":
                print(
                    "  Only the combined latent works. Stage 2 and deployment see the "
                    "prior alone, so the distillation has not transferred."
                )

        leak = step_log.get(
            "val_ctx_leak_recon_others", step_log.get("ctx_leak_recon_others")
        )
        if leak is not None:
            ratio = step_log.get("val_ctx_leak_ratio", step_log.get("ctx_leak_ratio"))
            held = ratio is None or ratio > 1.5
            print(
                f"[Stage 1 gate] leak probe (z_self -> teammates) {leak:.5f}"
                + (f", ratio vs prior-only {ratio:.2f}x" if ratio is not None else "")
                + f" -> {'SPLIT HELD' if held else 'NOT SEPARATED'}"
            )
            if not held:
                print(
                    "  z_self predicts teammates at least as well as z_team does, so the "
                    "two halves are not cleanly separated and the split may be cosmetic."
                )


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = ContextualizerWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
