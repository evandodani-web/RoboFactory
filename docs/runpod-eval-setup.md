# RunPod / headless Linux: RoboFactory + CLS-DP eval setup

This is a field guide for getting **ManiSkill / SAPIEN rendering** and **CLS-DP evaluation** working on a headless NVIDIA RunPod (or similar Docker GPU box). It records exactly what was required on a working pod (Ubuntu 24.04, NVIDIA L40S, driver 580.x, CUDA 12/13 host) after several earlier pods failed on Vulkan.

The short README Quick Start is necessary but **not sufficient** on many RunPods. The usual failure mode is:

```text
vkCreateInstance: Found no drivers!
ERROR_INCOMPATIBLE_DRIVER
```

or SAPIEN/ManiSkill segfaulting / refusing to render even though `nvidia-smi` looks fine.

---

## 0. What “done” looks like

Before you pull checkpoints, you want all of these green:

| Check | Command / expectation |
| --- | --- |
| GPU visible | `nvidia-smi` shows your GPU |
| Vulkan sees the GPU | `vulkaninfo --summary` lists `deviceName = NVIDIA …` |
| Python 3.9 venv | `source .venv/bin/activate` → `python -c "import torch; print(torch.cuda.is_available())"` → `True` |
| RoboFactory assets | `robofactory/assets/scenes/table/table.glb` exists |
| ManiSkill render | offscreen `env.render()` returns an RGB array |
| RoboFactory task | `LiftBarrier-rf` resets and steps with `render_mode="rgb_array"` |

---

## 1. Machine assumptions

Validated on:

- **Host OS in container:** Ubuntu 24.04 LTS
- **GPU:** NVIDIA L40S (any recent NVIDIA discrete GPU should be similar)
- **Driver:** 580.159.04 (ICD `api_version` 1.4.312)
- **Repo root:** `/workspace/RoboFactory` (adjust paths if your pod mounts elsewhere)

`nvidia-smi` working does **not** mean Vulkan works. RunPod images often ship:

- `NVIDIA_DRIVER_CAPABILITIES=compute,utility` (**missing `graphics`**)
- `NVIDIA_VISIBLE_DEVICES=void` (odd but common; CUDA can still work)
- A stock `/etc/vulkan/icd.d/nvidia_icd.json` whose `library_path` is the **basename** `libGLX_nvidia.so.0`

On this pod, that relative ICD path made the Vulkan loader print:

```text
loader_scanned_icd_add: Could not get 'vkCreateInstance' via 'vk_icdGetInstanceProcAddr'
for ICD libGLX_nvidia.so.0
```

even though `libGLX_nvidia.so.0` was on disk and exporting the ICD symbols. Pointing the ICD at the **absolute** library path fixed it.

---

## 2. System packages (EGL / OpenGL / Vulkan tools)

### 2.1 Install

On Ubuntu 24.04 the README package names `libegl1-mesa` / `libgles2-mesa` are transitional or renamed. Use:

```bash
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y \
  libgl1 \
  libglvnd0 \
  libegl1 \
  libgles2 \
  libopengl0 \
  libegl1-mesa-dev \
  libglvnd-dev \
  libvulkan1 \
  vulkan-tools \
  mesa-utils
```

Older Debian/Ubuntu may still accept the README line:

```bash
sudo apt install libgl1 libglvnd0 libegl1-mesa libgles2-mesa libopengl0
```

### 2.2 Confirm NVIDIA GL libs are present

```bash
ldconfig -p | grep -E 'libGLX_nvidia|libEGL_nvidia|libvulkan'
ls -la /lib/x86_64-linux-gnu/libGLX_nvidia.so.0 \
       /lib/x86_64-linux-gnu/libEGL_nvidia.so.0
```

You want real files (or symlinks into `*.580.*` / your driver version), not missing libs. Also confirm these vendor JSON files exist somewhere (locations vary by image):

- Vulkan ICD: `/etc/vulkan/icd.d/nvidia_icd.json` (and/or `/usr/share/vulkan/icd.d/nvidia_icd.json`)
- EGL vendor: `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`
- Optional Optimus layer: `/etc/vulkan/implicit_layer.d/nvidia_layers.json`

ManiSkill’s docs also describe creating these files if they are missing:
https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html

---

## 3. The Vulkan ICD fix (the important part)

### 3.1 Diagnose

```bash
vulkaninfo --summary
```

**Bad (what we saw with the stock ICD):**

```text
ERROR: ... Could not get 'vkCreateInstance' via 'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0
ERROR: ... vkCreateInstance: Found no drivers!
ERROR_INCOMPATIBLE_DRIVER
```

**Interesting detail:** loading the same `.so` from Python/`ctypes` and calling `vk_icdGetInstanceProcAddr(NULL, "vkCreateInstance")` returned a valid pointer. So the library was fine; the **loader + relative ICD path** combination was not.

### 3.2 Fix without writing under `/etc` (preferred on shared pods)

Create a **workspace-local** ICD that uses an absolute `library_path`, and force the loader to use only that file via `VK_ICD_FILENAMES`.

From the repo root (`$ROOT`, e.g. `/workspace/RoboFactory`):

```bash
ROOT=/workspace/RoboFactory   # change if needed
mkdir -p "$ROOT/.vulkan"

# Resolve the real library once (survives minor path differences)
LIBGLX="$(readlink -f /lib/x86_64-linux-gnu/libGLX_nvidia.so.0)"

cat > "$ROOT/.vulkan/nvidia_icd.json" <<EOF
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "${LIBGLX}",
        "api_version" : "1.4.312"
    }
}
EOF
```

If `readlink -f` is unavailable, hardcode:

```json
"library_path": "/lib/x86_64-linux-gnu/libGLX_nvidia.so.0"
```

Match `api_version` to whatever your stock ICD had (or use a conservative `"1.2.155"` as in ManiSkill’s docs). The absolute path is the critical bit.

### 3.3 Env helper script

Create `$ROOT/.vulkan/env.sh` and **source it in every shell** that runs sim / eval:

```bash
# Source before ManiSkill / SAPIEN eval on headless RunPod:
#   source /workspace/RoboFactory/.vulkan/env.sh

export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility,graphics}"

# Prefer all if void/empty (RunPod sometimes sets void)
if [ -z "${NVIDIA_VISIBLE_DEVICES:-}" ] || [ "${NVIDIA_VISIBLE_DEVICES}" = "void" ]; then
  export NVIDIA_VISIBLE_DEVICES=all
fi

export VK_ICD_FILENAMES="/workspace/RoboFactory/.vulkan/nvidia_icd.json"

# Headless: no X display needed for offscreen EGL/Vulkan rendering
unset DISPLAY || true

# Some pods set this; without the hf_transfer package, HF downloads hard-fail
unset HF_HUB_ENABLE_HF_TRANSFER || true

# Repo root must be on PYTHONPATH so `import robofactory` works from robofactory/
export PYTHONPATH="/workspace/RoboFactory${PYTHONPATH:+:$PYTHONPATH}"
```

Hook new shells (optional but convenient):

```bash
echo 'source /workspace/RoboFactory/.vulkan/env.sh' >> ~/.bashrc
```

### 3.4 Re-test Vulkan

```bash
source /workspace/RoboFactory/.vulkan/env.sh
vulkaninfo --summary | head -60
```

You should see something like:

```text
Devices:
========
GPU0:
        deviceName         = NVIDIA L40S
        driverID           = DRIVER_ID_NVIDIA_PROPRIETARY
```

Warnings about `DISPLAY` / `XDG_RUNTIME_DIR` are normal on headless pods when you are not using a window surface. Offscreen `rgb_array` rendering does not need them.

### 3.5 Alternative: patch the system ICD

If you control the image and prefer a global fix, rewrite `/etc/vulkan/icd.d/nvidia_icd.json` (and optionally `/usr/share/vulkan/icd.d/nvidia_icd.json`) to use the same absolute `library_path`. On shared / auto-reviewed environments, prefer the workspace-local `VK_ICD_FILENAMES` approach above.

---

## 4. Python environment (uv + lockfile)

RoboFactory pins **Python 3.9** and SAPIEN manylinux wheels. Do not use the system 3.12.

From the repo root:

```bash
cd /workspace/RoboFactory
bash setup_uv.sh
# equivalent: uv sync --python 3.9
source .venv/bin/activate
```

Notes from this pod:

- First sync downloads a large torch/CUDA stack (~10GB `.venv`).
- `setuptools` must stay `<81` (SAPIEN still imports `pkg_resources`).
- Recreate with `bash setup_uv.sh --force` if the venv is corrupted.

Activate every time:

```bash
source /workspace/RoboFactory/.venv/bin/activate
source /workspace/RoboFactory/.vulkan/env.sh
cd /workspace/RoboFactory/robofactory
```

Quick CUDA check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected: 2.6.0+cu124 True NVIDIA L40S
```

### Why `PYTHONPATH` matters

`uv` is configured with `package = false`, so the `robofactory` package is **not** editable-installed into the venv. Eval scripts do `sys.path.append("./")` while cwd is `robofactory/`, which is not enough for `from robofactory.tasks import *`. Putting the **repo root** on `PYTHONPATH` (as in `env.sh`) fixes imports.

---

## 5. Assets

### 5.1 Intended path

```bash
cd /workspace/RoboFactory/robofactory
python script/download_assets.py
```

That calls Hugging Face `snapshot_download` into `./assets`.

### 5.2 Pitfalls we hit

1. **`HF_HUB_ENABLE_HF_TRANSFER=1` without `hf_transfer` installed**  
   Instant failure: `ValueError: Fast download using 'hf_transfer' is enabled ...`.  
   Fix: `unset HF_HUB_ENABLE_HF_TRANSFER` (already in `env.sh`), or `pip install hf_transfer`.

2. **Anonymous HF rate limit (HTTP 429)**  
   After most files downloaded: `We had to rate limit your IP ... create a HF account ... pass a HF_TOKEN`.  
   Fix options:
   - `huggingface-cli login` / export `HF_TOKEN=…` and re-run (resumes), or
   - use the bundled zip (below).

### 5.3 Reliable fallback on this repo layout

`download_assets.py` also pulls `assets.zip` into `robofactory/assets/`. That zip is the complete object/scene set (~12MB compressed, ~1179 files). After a partial or rate-limited snapshot:

```bash
cd /workspace/RoboFactory/robofactory/assets
unzip -o assets.zip
# expect: scenes/table/table.glb, objects/steel_barrier_annotated/, etc.
```

You do **not** need RoboCasa assets for table tasks (`configs/table/*.yaml`). Only run `python -m mani_skill.utils.download_asset RoboCasa` if you evaluate kitchen configs.

ManiSkill robot / base assets download on first env creation into `~/.maniskill` (or `$MS_ASSET_DIR`).

---

## 6. Smoke tests (run these before overnight eval)

Always:

```bash
source /workspace/RoboFactory/.venv/bin/activate
source /workspace/RoboFactory/.vulkan/env.sh
cd /workspace/RoboFactory/robofactory
```

### 6.1 CLS-DP unit / pipeline (CPU, no sim required)

```bash
python policy/Diffusion-Policy/verify_cls_dp.py
python policy/Diffusion-Policy/verify_cls_pipeline.py
```

### 6.2 ManiSkill offscreen render

```bash
python - <<'PY'
import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np

env = gym.make(
    "PickCube-v1",
    obs_mode="rgb",
    control_mode="pd_ee_delta_pose",
    render_mode="rgb_array",
    num_envs=1,
)
obs, info = env.reset(seed=0)
for _ in range(5):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
frame = np.asarray(env.render())
print("render", frame.shape, frame.dtype)
env.close()
print("MANISKILL RENDER SMOKE OK")
PY
```

### 6.3 RoboFactory task used by CLS-DP eval

```bash
python - <<'PY'
import gymnasium as gym
import numpy as np
from robofactory.tasks import *  # noqa: F401,F403

env = gym.make(
    "LiftBarrier-rf",
    obs_mode="rgb",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
    reward_mode="dense",
    num_envs=1,
    sim_backend="auto",
    config="configs/table/lift_barrier.yaml",
)
obs, info = env.reset(seed=0)
print("render", np.asarray(env.render()).shape)
obs, reward, term, trunc, info = env.step(env.action_space.sample())
env.close()
print("ROBOFACTORY LIFTBARRIER SMOKE OK")
PY
```

If Vulkan is wrong, these usually fail at env construction or first `render()` with ICD / `enumeratePhysicalDevices` errors, not later in the policy code.

---

## 7. Running CLS-DP evaluation

Instruction banks already live under `robofactory/configs/instructions/`. Place checkpoints under:

```text
robofactory/checkpoints/{Task}_{prefix}_Agent{i}_{data_num}/{epoch}.ckpt
```

Example Study B prefix: `clsdp` (also `clsdpdet`, `clsdpfm`, `clsdpfg`, … depending on variant).

```bash
source /workspace/RoboFactory/.venv/bin/activate
source /workspace/RoboFactory/.vulkan/env.sh
cd /workspace/RoboFactory/robofactory

# DEBUG_MODE=0 → render_mode=rgb_array (correct for headless)
# DEBUG_MODE=1 → human viewer (needs a display; usually wrong on RunPod)
bash policy/Diffusion-Policy/eval_cls_multi.sh \
  configs/table/lift_barrier.yaml 150 100 0 LiftBarrier-rf 10000
```

First run may download SigLIP weights via Hugging Face (`transformers`). If HF rate-limits again, set `HF_TOKEN`.

Agent counts (for sanity-checking checkpoint folders): LiftBarrier/PlaceFood/TwoRobotsStackCube = 2; CameraAlignment/ThreeRobotsStackCube = 3; TakePhoto = 4.

---

## 8. Troubleshooting cheat sheet

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `nvidia-smi` OK, `vulkaninfo` “Found no drivers” | Relative ICD `library_path` or missing graphics capability | Absolute-path ICD + `VK_ICD_FILENAMES`; export `NVIDIA_DRIVER_CAPABILITIES=…,graphics` |
| `Could not get 'vkCreateInstance' … libGLX_nvidia.so.0` | Same ICD issue | §3 |
| `NVIDIA_VISIBLE_DEVICES=void` | RunPod default | `export NVIDIA_VISIBLE_DEVICES=all` (in `env.sh`) |
| SAPIEN / ManiSkill import OK, render black / crash | Vulkan still broken | Fix ICD before debugging policy |
| `ModuleNotFoundError: robofactory` | `package=false` + cwd | `PYTHONPATH=$REPO_ROOT` |
| `hf_transfer` ValueError | `HF_HUB_ENABLE_HF_TRANSFER=1` | `unset` it or install `hf_transfer` |
| HF `429 Too Many Requests` | Anonymous IP limit | `HF_TOKEN` / login, or unzip `assets.zip` |
| `pkg_resources` / setuptools errors | setuptools ≥ 81 | Keep `setuptools>=70,<81` (lockfile already does) |
| macOS / non-x86_64 | No SAPIEN wheels | Use Linux x86_64 |
| Eval opens viewer / hangs | `DEBUG_MODE=1` | Use `0` on headless |

Useful debug env:

```bash
VK_LOADER_DEBUG=error,warn,info vulkaninfo --summary 2>&1 | less
```

---

## 9. Minimal “new pod” checklist (copy-paste)

```bash
# 0) clone / cd to repo
cd /workspace/RoboFactory   # or your mount

# 1) apt deps
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y libgl1 libglvnd0 libegl1 libgles2 libopengl0 \
  libegl1-mesa-dev libglvnd-dev libvulkan1 vulkan-tools mesa-utils

# 2) workspace Vulkan ICD + env (see §3 for full files)
mkdir -p .vulkan
# write .vulkan/nvidia_icd.json with absolute libGLX_nvidia path
# write .vulkan/env.sh as in §3.3
source .vulkan/env.sh
vulkaninfo --summary | head -40   # must list NVIDIA GPU

# 3) Python
bash setup_uv.sh
source .venv/bin/activate
source .vulkan/env.sh

# 4) assets
cd robofactory
unset HF_HUB_ENABLE_HF_TRANSFER
python script/download_assets.py || unzip -o assets/assets.zip -d assets

# 5) smoke
python policy/Diffusion-Policy/verify_cls_dp.py
# then ManiSkill / LiftBarrier-rf render smokes from §6

# 6) drop checkpoints → eval_cls_multi.sh with DEBUG_MODE=0
```

---

## 10. Files this pod left behind for reuse

| Path | Purpose |
| --- | --- |
| `.vulkan/nvidia_icd.json` | Absolute-path NVIDIA Vulkan ICD |
| `.vulkan/env.sh` | Caps, `VK_ICD_FILENAMES`, `PYTHONPATH`, HF transfer unset |
| `.venv/` | uv-managed Python 3.9 env from `uv.lock` |
| `robofactory/assets/` | Table/object meshes (from zip / HF) |
| `~/.bashrc` hook | Auto-sources `.vulkan/env.sh` |

`.vulkan/` is machine-local glue; keep it on the pod or recreate with §3 when you spin a new one. Re-check `library_path` if the driver major version changes and the `.so` path moves.

---

## References

- Repo Quick Start / EGL blurb: [`README.md`](../README.md)
- Env bootstrap: [`setup_uv.sh`](../setup_uv.sh), [`pyproject.toml`](../pyproject.toml)
- ManiSkill Vulkan notes: https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html
- CLS-DP eval entry: `robofactory/policy/Diffusion-Policy/eval_cls_multi.sh`
