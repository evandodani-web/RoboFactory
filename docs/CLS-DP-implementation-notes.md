# CLS-DP Implementation Notes

Running log of decisions, rationale, and open questions while implementing CLS-DP in this repo.
Companion to [CLS-DP-replication-spec.md](CLS-DP-replication-spec.md), which is the paper extraction.

**Everything marked** `[REVIEW]` **is a judgement call I made where the paper is silent or where the
repo forced my hand. Those are the things worth your attention.** They are collected in
section 14 so you can skim them in one pass.

Status: implementation complete. Both verification suites pass (section 12): module math plus a
69-check end-to-end run over synthetic data covering the packer, dataset alignment, both losses,
both training loops and checkpoint round-trip. Not yet run against real demonstrations or the
simulator — see section 11 for why the simulator cannot be installed on this machine.

---



## 0. Where the code lives

**Decision:** implement inside the existing `diffusion_policy` package rather than creating a
sibling `policy/CLS-DP/` directory.

**Rationale.** The repo's README says "we plan to provide more policies in the future", which hints
at sibling directories under `policy/`. But `diffusion_policy` is not really "the DP policy" — it is
the shared training library (normalizer, replay buffer, sequence sampler, UNet components,
checkpoint utils, Hydra config root). A sibling directory would either duplicate ~60 files or
require fragile cross-package `sys.path` surgery.

Keeping CLS-DP inside the package means three things work with **zero** plumbing changes:

1. `train.py` is already generic — it just does `hydra.utils.get_class(cfg._target_)` and calls
  `workspace.run()`. Both CLS-DP stages reuse it as-is; only `--config-name` differs.
2. Hydra's `config_path` already points at `diffusion_policy/config`.
3. `get_policy()` at eval reconstructs a workspace from `cfg._target_` stored inside the
  checkpoint, so CLS-DP checkpoints load through the same path as DP checkpoints.

New files are namespaced with a `cls_` prefix or live under `model/cls/`.

### File inventory


| File                                                                        | Purpose                                                      |
| --------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `script/generate_instructions.py`                                           | Builds the per-task instruction bank                         |
| `script/parse_pkl_to_zarr_multi.py`                                         | Per-agent pkl -> time-aligned multi-agent zarr               |
| `script/precompute_siglip_features.py`                                      | Caches frozen SigLIP features into that zarr                 |
| `configs/instructions/*.json`                                               | 100 train + 100 held-out eval instructions per task          |
| `model/cls/siglip_encoder.py`                                               | Frozen SigLIP wrapper (never part of a saved model)          |
| `model/cls/prior_net.py`                                                    | Eq. 7 observation-conditioned prior + Fig. 4 attention split |
| `model/cls/ma_kinematics.py`                                                | Eq. 8 privileged encoder and the reconstruction decoder      |
| `model/cls/cross_attention.py`                                              | `CrossAttention1d` + `LatentTokenizer`                       |
| `model/diffusion/cls_conditional_unet1d.py`                                 | U-Net with z cross-attention in down + mid                   |
| `policy/contextualizer.py`                                                  | Stage 1 CVAE and the residual KL                             |
| `policy/cls_diffusion_unet_image_policy.py`                                 | Stage 2 action-expert                                        |
| `dataset/multi_agent_image_dataset.py`                                      | One dataset serving both stages                              |
| `workspace/contextualizer_workspace.py`                                     | Stage 1 loop + the go/no-go gate                             |
| `workspace/cls_robotworkspace.py`                                           | Stage 2 loop                                                 |
| `config/cls_stage1.yaml`, `config/cls_dp.yaml`, `config/task/cls_task.yaml` | Hydra                                                        |
| `train_cls_stage1.sh`, `train_cls_dp.sh`, `eval_cls_multi.sh`               | Entry points                                                 |
| `eval_multi_cls_dp.py`                                                      | Decentralized rollout                                        |
| `verify_cls_dp.py`                                                          | Module math and shapes; needs only torch                     |
| `verify_cls_pipeline.py`                                                    | End-to-end on synthetic data; 69 checks                      |


---



## 1. Timestep indexing — the single most important convention

This tripped me up, so it is worth writing down precisely.

The paper defines `A_t := a_{t:t+H-1}` (H future actions, execute the first 6). This repo's
`DiffusionUnetImagePolicy` does something subtly different:

```python
start = To - 1          # To = n_obs_steps = 3  ->  start = 2
end = start + self.n_action_steps
action = action_pred[:, start:end]
```

The sampled window has length `horizon=8` with `pad_before = n_obs_steps-1 = 2`. Window index
`To-1 = 2` **is** time `t` — the timestep of the most recent observation. So the U-Net predicts 8
actions spanning `a_{t-2} .. a_{t+5}`, and slicing `[2:10]` on a length-8 tensor yields **6** actions
`a_t .. a_{t+5}`. That is exactly the paper's "Execution steps: 6", and it confirms the authors ran
against this codebase.

**Decision** `[REVIEW]`**:** keep the repo convention for the action-expert rather than switching to the
paper's literal `a_{t:t+H-1}`.

Why: the `Ours w/o CLS` ablation is supposed to be "this policy minus `z`". If I change the action
indexing, the ablation is no longer the repo's DP and the headline 38-vs-9 comparison stops being
apples-to-apples. Keeping the convention also means the trained policy drops straight into the
existing eval loop, which hardcodes `for i in range(6)`.

The privileged target **does** follow the paper literally: `s^{1:N}_{t+1:t+H}`, i.e. 8 steps
starting one step after `t`.

### Window layout

`sequence_length = n_obs_steps + n_future_states = 3 + 8 = 11`, `pad_before = 2`, `pad_after = 8`.

```
window index:   0     1     2     3     4    ...    10
absolute time: t-2   t-1    t    t+1   t+2   ...   t+8

obs history        [0 : 3]        -> O_t^i, S_t^i          (L = 3)
action target      [0 : 8]        -> UNet horizon          (H = 8)
  executed slice     [2 : 8]      -> a_t .. a_{t+5}        (6 steps)
privileged future  [3 : 11]       -> s^{1:N}_{t+1:t+8}     (F = 8)
prior frame          [2]          -> o_t^i (current only)
```

I verified against `create_indices` in `common/sampler.py` that this covers every timestep
`t` in `[0, episode_len-1]` exactly once: `min_start = -pad_before = -2` gives `t=0`, and
`max_start = ep_len - 11 + 8 = ep_len - 3` gives `t = ep_len-1`. Edge padding repeats the first/last
frame, which is the standard DP behaviour.

Config resolution confirms the executed slice is 6 steps; see section 11.

---



## 2. Data pipeline



### 2.1 What the existing pipeline already gives us

Confirmed by reading `script/parse_h5_to_pkl_multi.py`:

```python
joint_action=res["action"][f'panda-{agent_id}'][j],
endpose=res["action"][f'panda-{agent_id}'][j],
```

Both fields come from the same array, and `parse_pkl_to_zarr_dp.py` then writes that same array into
`state`, `action`, **and** `tcp_action`. So `d_s = d_a = 8` and `s_t^i` is the *commanded joint
target* (7 arm joints + 1 gripper), not measured qpos. This is consistent with eval, where
`agent_pos` is fed from the previously executed action.

Useful consequence: the privileged target `s^{1:N}_{t+1:t+H}` is literally **all agents' future
action trajectories**. No new simulation or re-collection is needed.

Critically, the pkl writer loops `for agent_id in range(agent_num + 1)` over the *same* episode index
`i` and step index `j`, with `min_len` computed once from `panda-0`. So per-agent pkl directories are
already step-aligned. Building a multi-agent zarr is a pure re-packaging job.

### 2.2 New multi-agent zarr

`script/parse_pkl_to_zarr_multi.py` produces `data/zarr_data/{task}_multi_{num}.zarr`:

```
data/
  head_camera_agent{i}   (T, 3, 240, 320) uint8
  state_agent{i}         (T, 8) float32
  action_agent{i}        (T, 8) float32
  instruction_id         (T,)   int64
  siglip_img_agent{i}    (T, 17, 768) float16   <- added by the precompute script
meta/
  episode_ends           (E,)   int64
```

`episode_ends` is shared across agents because steps are aligned. This layout is exactly what
`ReplayBuffer.copy_from_store` expects (a `data/` group plus `meta/episode_ends`), so the existing
`SequenceSampler` works unmodified. `instruction_id` is constant within an episode.

The script buffers **one episode at a time** and uses `zarr.Array.append`, so peak memory stays
bounded. The single-agent script accumulates everything in a Python list first, which would be
roughly `N x 7GB` here.

`[REVIEW]` **The** `{task}_global` **pkl directory is deliberately ignored.** It contains
`head_camera_global`, and CLS-DP explicitly forbids shared global views. Using it would invalidate
the whole premise.

### 2.3 Memory note

`ReplayBuffer.copy_from_path(..., store=None)` loads every requested array fully into RAM. That is
why the dataset's key set is stage-dependent (section 5.5): Stage 1 runs at batch size 512 and does
not touch images at all, so it never pays the ~7GB per-agent camera cost.

---



## 3. Instructions

The paper generates instructions with an LLM, then diversifies them following RoboTwin 2.0:
100 for training, 100 held out for evaluation, one sampled per episode and shared by all agents.

**Decision** `[REVIEW]`**:** ship a deterministic template-based generator
(`script/generate_instructions.py`) instead of calling an LLM. Reproducible without an API key and
no network dependency in the data pipeline. The seed phrasings are taken verbatim from the paper's
Table IV so the distribution is anchored to the original. It produces 240-756 unique phrasings per
task, then splits 100/100 with a fixed seed. If you want true LLM instructions, replace the JSON
files — nothing else depends on how they were produced.

The generator hard-fails if a task cannot produce 200 unique phrasings, which caught three tasks
that were initially too thin.

**Two bugs caught while generating:**

1. **Colour mismatch.** The paper's stacking instructions say "blue / red / green", but the configs
  define `cubeA` blue, `cubeB` **green**, `cubeC` **red**. The generator follows the configs so the
   language matches what the agent actually sees.
2. **Grammar leaks.** The first version produced "Both arms Hoist the barrier" (capitalised verb
  mid-sentence) and "Bring the item **at** its target position" (wrong preposition). Fixed with a
   slot naming convention: capitalised keys are sentence-initial verbs, `*_l` keys are the
   mid-sentence forms, and `to_target` / `at_target` are separate so motion verbs and placement
   verbs get the right preposition. Camera Alignment and Take Photo also get target wording that
   does not mention a "goal region", since those tasks mark the target with a static cube.

---



## 4. SigLIP handling

Both towers are frozen and the prior consumes only the **current** frame, so features can be
precomputed once. Caching all 196 patch tokens at 768-d is ~300 KB/frame, larger than the source
image (~230 KB), and is ~9 GB per agent at 150 episodes. That is fine on this machine.

**Decision** `[REVIEW]`**, Study A:** average-pool the 14x14 patch grid down to `4x4 = 16` tokens
and prepend the pooled embedding (`M = 17`). That was a cache-size call, not a paper claim.

**Study B (current):** keep the native 14x14 grid (`M = 197` = pooled embedding + 196 patches).
The paper never said to downsample, and Study A's 37% LiftBarrier result is 24 points below the
published 61%, so this run removes that deviation. Revert with `--pool_grid 4` on precompute and
eval if you need the cheap cache again.

Text is trivial to cache (200 instructions per task), so it is stored at full token resolution in a
sidecar `.npz`. It cannot live inside the zarr because its leading dimension is the instruction
count, not the timestep count, and `ReplayBuffer` expects every array under `data/` to share a time
axis.

**Decision:** SigLIP lives **outside** any saved model. `SigLIPFeatureExtractor` is instantiated by
the precompute script and by the eval script, never by a policy. This keeps checkpoints free of
~800 MB of frozen weights and means dataloader workers never need a GPU.

Consequence: `precompute_siglip_features.py` **is a required pipeline step**, not an optimisation.
The eval script runs SigLIP live because there is no cache at rollout time.

`transformers==4.49.0` added to `robofactory/requirements.txt`.

---



## 5. The CVAE



### 5.1 Residual KL in closed form

With `mu_q = mu_rho + mu_E`, `sigma_q = sigma_E`, `mu_p = mu_rho`, `sigma_p = sigma_rho`, the prior
mean cancels out of the mean-difference term:

```
KL = sum_d [ log(sigma_rho/sigma_E) + (sigma_E^2 + mu_E^2) / (2 * sigma_rho^2) - 0.5 ]
```

Implemented over log-sigmas in `_residual_kl`. This is not just an optimisation — it is the
mechanism:

- `mu_rho` gets **no gradient from the KL at all**. It is trained purely by reconstruction, flowing
through `z`. That is what forces the prior mean to encode teammate dynamics.
- `sigma_rho` gets gradient **only** from the KL, aligning it to `sigma_E`.

If you implement the posterior as an independent Gaussian, the first property disappears and the
distillation stops working. `verify_cls_dp.py` guards this by drawing `mu_rho` at scale 5.0 and
asserting the closed form still matches `torch.distributions.kl_divergence` exactly — if the prior
mean leaked in, that test would blow up. Measured error: **0.000e+00**.

### 5.2 Zero-init trick `[REVIEW]`

`to_mu_residual` and both `to_log_sigma` heads are zero-initialised. At step 0 this gives
`mu_E = 0` and `sigma_E = sigma_rho = 1`, so **the posterior starts exactly equal to the prior and
the KL starts at exactly 0** (verified). The residual only grows as far as reconstruction demands.

Not stated in the paper, but it follows naturally from the residual parameterisation and makes the
beta warm-up better behaved. Cheap to revert.

### 5.3 Beta warm-up

`beta = 1e-1` with linear warm-up over the first 40% of training (Table I), implemented as a ramp
`0 -> beta` across the first `0.4 * total_steps` optimizer steps, then constant.

### 5.4 The go/no-go gate

Stage 1 logs reconstruction error split into **own-agent** and **teammate** components:

- `ctx_recon_own` — how well `D(s_t^i, z)` reconstructs agent `i`'s own future
- `ctx_recon_others` — how well it reconstructs everyone else's
- `ctx_recon_others_baseline` — a batch-mean predictor on the same targets

`recon_others` is the real signal. `s_t^i` alone explains `recon_own` reasonably well, so only the
teammate term proves that `z` carries coordination information. `ContextualizerWorkspace` prints an
explicit PASS/FAIL at the end of training. **If it FAILs, do not bother running Stage 2** — the
latent is empty and you will just reproduce the `w/o CLS` ablation at higher cost.

### 5.5 One dataset, two stages

`MultiAgentImageDataset` takes a `stage` flag that controls which zarr keys get loaded:

- Stage 1: every agent's `state_agent{j}`, this agent's `siglip_img_agent{i}`, `instruction_id`
- Stage 2: this agent's camera, state, action, SigLIP features, `instruction_id`

This matters for more than tidiness. Stage 1 runs at batch size 512; if it pulled raw images it
never uses, the preallocated buffers alone would be `512 x 11 x 230KB` = 1.3 GB, on top of ~7 GB of
resident replay buffer per agent.

---



## 6. Architecture sizing

The paper gives no layer counts. I sized against the parameter budget derived from Table III in the
spec (~2.3 M for the MA-kinematics pair, ~95 M marginal per agent). Measured values from
`verify_cls_dp.py`:


| Module                 | Config                                     | Params (measured)                  |
| ---------------------- | ------------------------------------------ | ---------------------------------- |
| `MAKinematicsEncoder`  | `d_model=256`, 2 layers, 4 heads, ff 512   | 1.19 M                             |
| `MAKinematicsDecoder`  | `d_model=256`, 2 layers, 4 heads, ff 512   | 1.66 M                             |
| encoder + decoder      |                                            | **2.85 M** (derived target ~2.3 M) |
| `PriorNet`             | `d_model=768`, 2 layers, 8 heads, ff 2048  | 17.3 M                             |
| `CLSConditionalUnet1D` | `[256,512,1024]`, incl. 5 cross-attn sites | 78.1 M                             |


Per-agent deployed total is roughly `78.1 + 17.3 + 11` (ResNet-18) = **~106 M**, against the ~95 M
marginal cost implied by Table III. Same order, slightly heavy — the encoder/decoder pair is about
0.5 M over budget and `PriorNet` is a guess. All Hydra-configurable if you want to trim.

---



## 7. Cross-attention injection

Per the paper, `z` enters **only** the downsampling and bottleneck stages, never the upsampling path.
With `down_dims: [256, 512, 1024]` that is 3 down levels + 2 mid blocks = **5 injection sites**
(asserted in `verify_cls_dp.py`).

`CLSConditionalUnet1D` is a new file rather than a modification of `ConditionalUnet1D`, so the
existing DP and the `w/o CLS` ablation stay byte-identical.

**Decision** `[REVIEW]`**:** the cross-attention output projection is **zero-initialised**, so every
block is an exact identity at step 0 and the network starts as vanilla DP, then learns to use `z`.
Standard practice for adding a conditioning branch (ControlNet-style) and it noticeably stabilises
early training. Verified both directions: identical output with and without `z` at init, and a
non-zero difference once the projections have moved.

**Decision** `[REVIEW]`**:** `z (256,)` is projected to `n_cond_tokens = 4` tokens of width 256 for
keys/values. The paper does not say how a single latent vector becomes an attention context. One
token would work; 4 gives slightly more capacity at negligible cost.

---



## 8. Stage 2 details

- `z` is sampled from the **prior** during Stage 2 training, never the posterior — the paper is
explicit, and it is what makes train and deploy consistent.
- `sg(z)` is enforced with `torch.no_grad()` around the prior forward plus `.detach()`.
- `PriorNet` is frozen with `requires_grad_(False)` and forced back to `.eval()` on every `train()`
call, so its norm/dropout statistics cannot drift from what Stage 1 produced.
- The optimizer is built from `filter(requires_grad)`, so frozen params never reach Adam.
- Prior weights are loaded **before** the EMA deepcopy, so both models start identical. `EMAModel`
copies `requires_grad=False` params verbatim rather than averaging them (checked in
`ema_model.py`), so the frozen prior stays exactly frozen in the EMA copy too.
- `PriorNet` **is** part of the policy's `state_dict`, so Stage 2 checkpoints are self-contained
and eval needs only that one file plus SigLIP from HuggingFace.

**Decision** `[REVIEW]`**:** `latent_sample: true` by default at inference — actually sample
`z ~ N(mu_rho, sigma_rho)` rather than taking the mean. Faithful to Eq. 12, which defines the
deployed policy as a marginal over `z`. Set `latent_sample: false` (or pass `--latent-sample False`
to the eval script) for lower-variance, more repeatable rollouts; worth trying if results are noisy.

**Checkpoint size.** Model + EMA + optimizer for ~95 M trainable params lands around 1.1 GB, versus
roughly 0.9 GB for the repo's DP. `checkpoint_every` is set to 25 rather than the repo's 150 because
training is only 100 epochs.

---



## 9. Deviations from the repo's DP defaults

Per Table I:


| Setting            | Repo DP | CLS-DP                              |
| ------------------ | ------- | ----------------------------------- |
| `num_epochs`       | 300     | **100**                             |
| `batch_size`       | 64      | **32** (Stage 2), **512** (Stage 1) |
| `checkpoint_every` | 150     | **25**                              |
| everything else    |         | unchanged                           |


`horizon=8`, `n_obs_steps=3`, `K=100`, ResNet-18, FiLM and `lr=1e-4` already match the paper exactly.

`[REVIEW]` **Optimizer.** Table I / §V-A say "Adam". Study A used `AdamW` (`weight_decay=1e-6`)
so CLS-DP and the `w/o CLS` ablation would differ only in `z`. **Study B (current) uses
`torch.optim.Adam`** in `cls_stage1.yaml` and `cls_dp.yaml` to match the paper. Same `lr`,
`betas`, `eps`, and `weight_decay=1e-6`. Swap `_target_` back to `torch.optim.AdamW` if you
want the ablation-fair optimizer again.

---



## 10. Bugs and rough edges found in the existing code

Not blockers, but worth knowing. I did not change existing files except to add `transformers` to
`requirements.txt`.

1. `pin_memory()` **is a no-op** in `RobotImageDataset`. `Tensor.pin_memory()` returns a *copy* in
  pinned memory; the return value is discarded. It cannot work as written anyway — the torch views
   must alias the numpy buffers that the numba sampler writes into, and a pinned copy would not.
   I omitted it from the new dataset and left a comment saying why.
2. `dataset.batch_size` **must equal** `dataloader.batch_size`**.** `default_task.yaml` never sets
  `dataset.batch_size`, so it silently relies on the class default (64) matching `robot_dp.yaml`'s
   dataloader (64). Change one and you get an assertion failure deep in `__getitem__`. The CLS task
   config wires it explicitly via `${dataloader.batch_size}`.
3. `get_model_input` **yields float64.** Dividing a uint8 array by the Python int `255` promotes to
  float64. DP gets away with it because `_normalize` casts to `scale.dtype`. The prior inputs bypass
   the normalizer, so the CLS eval script casts to float32 explicitly.
4. `info['success']` **is a batched tensor**, not a bool. The existing `if info['success'] == True:`
  works by accident. The CLS eval script uses an explicit `_is_success` helper.
5. `torch.mean(torch.tensor(val_losses))` on a list of 0-dim tensors works only because they are
  scalars. The CLS workspace accumulates `.item()` floats instead.

---



## 11. Environment

**Official replication path (Linux x86_64):** `pyproject.toml` + `uv.lock` + `setup_uv.sh`.
That creates a uv-managed Python 3.9 `.venv`. Recreate with `bash setup_uv.sh --force`
or `uv sync --python 3.9`. `robofactory/requirements.txt` is a pip-readable mirror of the
direct pins, not the source of truth.

The first pass on macOS had **no conda** — the README's old `conda create -n RoboFactory python=3.9`
had never been run there. A local `/Users/evan.dodani/dev/RoboFactory/.venv` was built from
system Python 3.9.6 as a fallback and `.venv/` was gitignored. A later Linux pass used
Miniforge; that conda env is **not** the replication recipe either.

`pip install -r robofactory/requirements.txt` **fails on macOS**, at `mani_skill` ->
`sapien==3.0.0.b1`:

```
ERROR: Could not find a version that satisfies the requirement sapien==3.0.0.b1 (from versions: none)
```

This is pre-existing and is exactly what `setup.py` warns about ("until sapien is uploaded to pypi
with mac support, users need to install manually"). The workaround URL in that comment is now a
404, and SAPIEN's current nightly release publishes macOS wheels only for **cp310-cp314** — there is
no cp39 macOS build. So under the README's Python 3.9, the simulator cannot be installed on macOS at
all. Even with cp310+ it would still need Vulkan/MoltenVK.

Everything else installs at the pinned versions:


| Package             | Version                    |
| ------------------- | -------------------------- |
| torch / torchvision | 2.6.0 / 0.21.0 (as pinned) |
| zarr                | 2.18.2                     |
| hydra-core          | 1.3.2                      |
| diffusers           | 0.32.2                     |
| numba               | 0.60.0                     |
| transformers        | 4.49.0                     |
| numpy               | 2.0.2                      |


Two environment notes:

- **numpy resolves to 2.0.2, not 1.26.4.** `mani_skill` pins `numpy<2.0.0`; without it, pip picks
2.0.2. Worth knowing because `numpy==1.26.4` **segfaults on import** under this machine's Python
3.9.6, so a full install including `mani_skill` may be unusable here regardless of SAPIEN.
- `sentencepiece` **was missing.** `SiglipTokenizer` requires it and `transformers` does not pull it
in. Found by running the real encoder; now pinned in `requirements.txt`.

**Consequence on macOS:** everything except `eval_multi_cls_dp.py`'s simulator loop is testable.
Actual training and rollouts need Linux x86_64.

**Linux x86_64 uv env (this recipe):** `mani_skill` pulls `numpy==1.26.4`, sapien 3.0.0b1
imports, and `setuptools` stays at 80.x. That is the stack to replicate. `uv.lock` pins
the full transitive graph.

---



## 12. Verification performed

Two suites, both runnable from `robofactory/`:

```bash
../.venv/bin/python policy/Diffusion-Policy/verify_cls_dp.py        # module math, ~2s
../.venv/bin/python policy/Diffusion-Policy/verify_cls_pipeline.py  # end to end, ~7s
```

`verify_cls_dp.py` (needs only torch) covers module shapes, the residual-KL identity against
`torch.distributions.kl_divergence` (error **0.000e+00** with `mu_rho` drawn at scale 5.0, proving
the prior mean cancels), KL exactly 0 at init, cross-attention identity at init, and z influencing
the U-Net once trained.

`verify_cls_pipeline.py` builds a synthetic dataset on disk and drives the real code end to end:
**69 checks**, covering the pkl->zarr packer, dataset timestep alignment, both loss functions, both
training workspaces including checkpoint round-trip, and the eval script's pure logic with the
simulator stubbed out.

The centrepiece is the alignment check. Synthetic states encode their own index as
`state[t] = [t, agent_id, ...]`, so the test asserts *exactly* which absolute timesteps land where:
observation history is `o_{t-2..t}`, the action target is `a_{t-2..t+5}`, the executed slice is
`a_{t..t+5}` (6 steps), and the privileged window is `s^{1:N}_{t+1:t+8}` with the agent axis in
order. A shape-only test would pass while being off by one; this would not.

Also verified: the Stage 1 gate discriminates in **both** directions — it passes after 60 real
training steps (section [3], `others=0.0047` vs `baseline=0.0073`) and correctly fails after 6 debug
steps on random features (section [5]). That FAIL in the test output is expected and annotated.

The real SigLIP encoder was exercised separately against the downloaded
`google/siglip-base-patch16-224`: `feature_dim=768`, `image_size=224`, 17 tokens per frame, correct
handling of both float and uint8 input, 26 KB/frame in fp16 versus 230 KB for the raw image.

**Still not covered — needs your environment:**

- Real demonstration data through `parse_h5_to_pkl_multi.py`.
- `precompute_siglip_features.py` against a real multi-agent zarr (its components are tested, the
script itself is not).
- Any simulator rollout: `eval_multi_cls_dp.py`'s env loop, TOPP smoothing, and success detection.
- GPU/CUDA paths and real training dynamics.

Start with `--load_num 5` and `training.debug=True` to shake out the plumbing cheaply.

---



## 13. Bugs the tests caught

All four were real and are fixed. Three would have surfaced only after you had already collected
data and started a run.

1. **numba cannot compile float16.** `batch_sample_sequence` runs in nopython mode and raised
  `NotImplementedError: float16` on the SigLIP cache. This would have crashed on the *first
   training batch of every run*. Fixed by widening float16 arrays to float32 once, right after
   `copy_from_path`, so the fp16-on-disk saving is kept and numba gets a type it supports.
2. **SigLIP text padding was left unmasked.** SigLIP's tokenizer returns only `input_ids` with no
  attention mask, and pads with the EOS id out to `max_length`: "Open the lid." is 3 real tokens
   followed by 61 pads. My fallback all-ones mask meant the prior's cross-attention spent its text
   budget on padding. Measured at init: unmasked gives a text share of **0.801**, masked gives
   **0.284** (which matches the uniform-attention baseline for ~6.5 text vs 17 image tokens). Left
   unfixed, the Fig. 4 analysis would have measured padding count rather than grounding. Now the
   mask is derived from `input_ids != pad_token_id`, keeping the first pad as the terminator.
3. `sentencepiece` **missing from requirements.** `SiglipTokenizer` needs it; `transformers` does
  not depend on it. Precompute would have failed immediately in a fresh environment.
4. `JsonLogger` **does not create its output directory.** Hydra normally creates the run dir, so
  this only bites when a workspace is driven programmatically. Both workspaces now `makedirs`
   first, matching what `save_checkpoint` already does.

One finding that was **not** a bug, and is worth remembering because it looks like one:

- With a zero-initialised `to_out`, the cross-attention `to_q/to_k/to_v` receive **exactly zero**
gradient on step 0 — their gradient routes through `to_out.weight`, which is still zero. Only
`to_out` trains on the first step; the rest of the branch starts learning immediately after. This
is the standard ControlNet zero-conv behaviour. My first version of the test asserted q/k/v
gradient at step 0 and failed; the test now asserts the correct invariant and checks that the
branch trains end to end after one optimizer step.

---



## 14. All `[REVIEW]` decisions in one place


| #   | Decision                                                                | Where       | Revert cost                 |
| --- | ----------------------------------------------------------------------- | ----------- | --------------------------- |
| 1   | Code lives inside `diffusion_policy`, not a sibling package             | section 0   | high                        |
| 2   | Repo action-indexing convention over the paper's literal `a_{t:t+H-1}`  | section 1   | medium                      |
| 3   | `{task}_global` camera ignored entirely                                 | section 2.2 | n/a, required by the method |
| 4   | Template instruction generator instead of an LLM                        | section 3   | low, swap the JSON          |
| 5   | Cube colours from configs, not from the paper                           | section 3   | low                         |
| 6   | SigLIP: Study A pooled 14x14 -> 4x4; Study B keeps the full 14x14 grid  | section 4   | low, re-run precompute      |
| 7   | Zero-init posterior heads so KL starts at 0                             | section 5.2 | low                         |
| 8   | Zero-init cross-attention output projections                            | section 7   | low                         |
| 9   | `n_cond_tokens = 4`                                                     | section 7   | low                         |
| 10  | `latent_sample: true` at inference                                      | section 8   | low, config flag            |
| 11  | Optimizer: Study A used AdamW; Study B uses paper Adam                  | section 9   | low                         |
| 12  | One contextualizer per agent (not weight-shared)                        | below       | high                        |
| 13  | Stage 1 also runs 100 epochs                                            | below       | low                         |


**12** follows Eq. 5, which sums over per-agent `(theta_i, psi_i)`. It is the expensive reading;
sharing weights across agents would cut Stage 1 cost by `N` if you need to economise.

**13**: the paper says "all methods are trained for 100 epochs" but is ambiguous about whether that
covers the contextualizer. I defaulted Stage 1 to 100 as well.

---



## 15. Runbook

All commands run from the `robofactory/` directory, matching the existing README.

```bash
# 0. one-time: instruction banks for all six tasks
python script/generate_instructions.py --all

# 1. collect demonstrations (unchanged from the existing pipeline)
python script/generate_data.py --config configs/table/lift_barrier.yaml --num 150
mv <traj>.h5   data/h5_data/LiftBarrier-rf.h5
mv <traj>.json data/h5_data/LiftBarrier-rf.json

# 2. h5 -> per-agent pkl (unchanged)
python script/parse_h5_to_pkl_multi.py --task_name LiftBarrier-rf --load_num 150 --agent_num 2

# 3. per-agent pkl -> time-aligned multi-agent zarr  [NEW]
python script/parse_pkl_to_zarr_multi.py --task_name LiftBarrier-rf --load_num 150 --agent_num 2

# 4. cache frozen SigLIP features into that zarr  [NEW, required]
python script/precompute_siglip_features.py \
    --zarr_path data/zarr_data/LiftBarrier-rf_multi_150.zarr \
    --pool_grid 14 --overwrite

# 5. Stage 1: contextualizer, once per agent
#    args: task load_num agent_id n_agents seed gpu
bash policy/Diffusion-Policy/train_cls_stage1.sh LiftBarrier-rf 150 0 2 42 0
bash policy/Diffusion-Policy/train_cls_stage1.sh LiftBarrier-rf 150 1 2 42 0
#    -> CHECK THE PASS/FAIL GATE PRINTED AT THE END BEFORE CONTINUING

# 6. Stage 2: action-expert, once per agent
bash policy/Diffusion-Policy/train_cls_dp.sh LiftBarrier-rf 150 0 2 42 0
bash policy/Diffusion-Policy/train_cls_dp.sh LiftBarrier-rf 150 1 2 42 0

# 7. decentralized evaluation
#    args: config data_num ckpt debug task [seed]
bash policy/Diffusion-Policy/eval_cls_multi.sh \
    configs/table/lift_barrier.yaml 150 100 1 LiftBarrier-rf 10000

# anytime: CPU checks, no data, no GPU, no simulator needed
python policy/Diffusion-Policy/verify_cls_dp.py        # module math, ~2s
python policy/Diffusion-Policy/verify_cls_pipeline.py  # end to end, ~7s
```

Agent counts per task: LiftBarrier 2, PlaceFood 2, TwoRobotsStackCube 2, CameraAlignment 3,
ThreeRobotsStackCube 3, TakePhoto 4.

### The `w/o CLS` ablation

The paper's decentralized baseline is this policy minus `z`. The cheapest faithful way to get it is
to train the repo's existing per-agent DP (`train.sh`) on the same demonstrations, since the CLS
action-expert is deliberately identical to it apart from the cross-attention branch. Reproducing the
38-vs-9 gap is the primary correctness check on the whole implementation.

---



## 16. Change log

- Extracted the paper into `CLS-DP-replication-spec.md`, including the derived parameter budget from
Table III and the residual-KL derivation.
- Built the data path: instruction generator, multi-agent zarr packer, SigLIP precompute.
- Built the model: `PriorNet`, `MAKinematicsEncoder` / `Decoder`, `CrossAttention1d`,
`LatentTokenizer`, `CLSConditionalUnet1D`, `Contextualizer`, `CLSDiffusionUnetImagePolicy`.
- Built training: `MultiAgentImageDataset` (both stages), `ContextualizerWorkspace` with the
teammate-reconstruction gate, `CLSRobotWorkspace` with frozen-prior loading.
- Built configs, three shell entry points, and the decentralized eval script.
- Added `verify_cls_dp.py` and fixed everything it caught.
- Expanded instruction templates twice: once for the 200-phrasing floor, once for grammar.
- Added `transformers==4.49.0` to requirements.
- Built `.venv` from system Python 3.9.6 and installed everything except the simulator, which has
no macOS/cp39 wheel (section 11). Added `.venv/` to `.gitignore`.
- Added `verify_cls_pipeline.py` (69 end-to-end checks) and fixed the four bugs it and the real
SigLIP run surfaced: numba/float16, unmasked SigLIP text padding, missing `sentencepiece`, and
`JsonLogger`'s missing output directory (section 13).
- Pinned `sentencepiece==0.2.2`.
- Pinned `setuptools>=70,<81` so SAPIEN can still import `pkg_resources`.
- Replaced the conda recipe with `pyproject.toml` + `uv.lock` + `setup_uv.sh`.
- Started **Study B** (section 17): 150 demos, Adam, full 14x14 SigLIP. Study A (100 / AdamW /
  4x4, 37% LiftBarrier) is left in place.

---

## 17. Study log

The 100-demo LiftBarrier run is **Study A**. This section records it and the paper-matching
follow-up so the two are not mixed.

### Study A — 100 demos, AdamW, 4x4 SigLIP (done)

| Knob | Value |
|---|---|
| Demos | 100 |
| Optimizer | `AdamW`, `weight_decay=1e-6` |
| SigLIP tokens | 17 (pooled embedding + 4x4 grid) |
| Checkpoints | `checkpoints/LiftBarrier-rf_{ctx,clsdp}_Agent{0,1}_100/` |
| Eval | `eval_results/LiftBarrier-rf_100_100_20260829_024814/` |

Result: **37 / 100** on LiftBarrier (paper CLS-DP: **61%**; paper `w/o CLS`: 14%). The 100-demo
count was a leftover from an earlier smoke-test rebuild, not a paper setting. Do not overwrite
these artifacts.

### Study B — 150 demos, Adam, 14x14 SigLIP (current)

A new study, not a continuation of Study A. The three paper-matching changes are applied
together, so this run cannot isolate which one closed (or failed to close) the 24-point gap.

| Knob | Value | Why |
|---|---|---|
| Demos | **150** | paper / repo runbook; H5 already has 150 trajectories |
| Optimizer | **`Adam`** | Table I / §V-A |
| SigLIP tokens | **197** (pooled embedding + native 14x14) | paper never said to downsample |
| Checkpoints | `checkpoints/LiftBarrier-rf_{ctx,clsdp}_Agent{0,1}_150/` | distinct from Study A |

Same task (LiftBarrier), same seeds, same 100-epoch schedule, same instruction bank. Stage 1
must still print PASS on the teammate-reconstruction gate before Stage 2 starts.

LiftBarrier Study B eval: **61 / 100**, matching Table II.

### Study DET — deterministic latent (built, not yet trained)

Ablates the stochasticity of the coordination latent, holding everything else fixed against
Study B. Not a new objective: setting `sigma_prior = sigma_posterior = 1` in the Gaussian KL
collapses it to `0.5*||z_E||^2`, so this is the same loss with the scale parameters frozen
out. `beta` keeps its meaning and warm-up, and the distillation geometry is unchanged — the
alignment term still depends only on the residual, so the prior is still trained purely by
reconstruction.

| Knob | Value |
|---|---|
| Latent | deterministic; encoders emit `z` directly, no scale head is constructed |
| Sampling | none, anywhere. `latent_sample` cannot reintroduce it |
| Configs | `cls_stage1_det.yaml`, `cls_dp_det.yaml` (inherit their baselines via Hydra defaults) |
| Checkpoints | `checkpoints/LiftBarrier-rf_{ctxdet,clsdpdet}_Agent{0,1}_150/` |
| Pipeline | `policy/Diffusion-Policy/train_study_det.sh` |
| Eval | `eval_cls_sweep.sh ... clsdpdet` (9th arg selects the checkpoint family) |

Config inheritance was verified by composition: the resolved configs differ from their Study B
parents **only** in the deterministic flags, `latent_sample`, the run name and the checkpoint
prefix. Adam, `beta` 0.1 / 0.4 warm-up, 100 epochs, K=100, horizon 8, `d_model` 768 and latent
256 are all inherited, so any delta is attributable to the stochasticity.

Reuses Study B's 14x14 SigLIP cache — the frozen encoders are untouched. The pipeline script
asserts the cache has 197 tokens rather than silently training on a stale 4x4 one.

**What this tests.** Whether the learned per-dimension `sigma_prior` was load-bearing. Under
partial observability some coordination dimensions are genuinely unknowable from one frame, and
a deterministic alignment has no way to say so — it forces the prior to match the posterior even
there, and the usual result is mode-averaging. Watch `ctx_recon_others` against Study B: if it
degrades, that is the mechanism showing up, not a bug.

In deterministic mode the `ctx_kl` metric key holds the L2 alignment term. The key name is
shared so the gate and epoch-averaging code stay common.

### Study B — ThreeRobotsStackCube (in progress)

Same recipe, new task. Paper Table II is **20%** on this 3-agent stack. Official HuggingFace
demos (`FACEONG/RoboFactory_Dataset/ThreeRobotsStackCube`, 150 trajectories). Checkpoints
`checkpoints/ThreeRobotsStackCube-rf_{ctx,clsdp}_Agent{0,1,2}_150/`. Eval uses
`max_steps=800` (the env horizon) and the same 100 unseen seeds / held-out instructions.
Pipeline: `policy/Diffusion-Policy/train_eval_three_robots_stack_cube.sh`.

