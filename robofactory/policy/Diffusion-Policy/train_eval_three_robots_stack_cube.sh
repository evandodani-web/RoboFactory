#!/usr/bin/env bash
# Study B recipe on ThreeRobotsStackCube (3 agents), then a 100-seed eval.
# Same knobs that reproduced LiftBarrier at 61%: 150 demos, Adam, 14x14 SigLIP.
set -euo pipefail

REPO_ROOT=/workspace/RoboFactory
cd "${REPO_ROOT}/robofactory"
source "${REPO_ROOT}/.venv/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export HF_HUB_ENABLE_HF_TRANSFER=0

TASK="ThreeRobotsStackCube-rf"
CONFIG="configs/table/three_robots_stack_cube.yaml"
N_AGENTS=3
LOAD_NUM=150
SEED=42
GPU=0
# Env is registered with 800 steps (LiftBarrier was 500, eval still used 250
# policy cycles). Use the env horizon so a long sequential stack is not cut short.
MAX_STEPS=800
JOBS=10

ZARR="data/zarr_data/${TASK}_multi_${LOAD_NUM}.zarr"
H5="data/h5_data/${TASK}.h5"
JSON="data/h5_data/${TASK}.json"
LOG_DIR="data/outputs/study_b_threerobots"
mkdir -p "${LOG_DIR}" data/h5_data data/pkl_data data/zarr_data

echo "=== Study B / ThreeRobotsStackCube: download official 150 demos ==="
python - <<'PY'
import os
import shutil
from huggingface_hub import hf_hub_download

dest = "/workspace/RoboFactory/robofactory/data/h5_data"
pairs = [
    ("ThreeRobotsStackCube/ThreeRobotsStackCube.h5", "ThreeRobotsStackCube-rf.h5"),
    ("ThreeRobotsStackCube/ThreeRobotsStackCube.json", "ThreeRobotsStackCube-rf.json"),
]
for remote, local in pairs:
    target = os.path.join(dest, local)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        print(f"already have {target} ({os.path.getsize(target)} bytes)")
        continue
    path = hf_hub_download(
        repo_id="FACEONG/RoboFactory_Dataset",
        repo_type="dataset",
        filename=remote,
        local_dir="/tmp/rf_ds",
    )
    shutil.copy2(path, target)
    print(f"wrote {target} ({os.path.getsize(target)} bytes)")
PY

python - <<'PY'
import json
p = "/workspace/RoboFactory/robofactory/data/h5_data/ThreeRobotsStackCube-rf.json"
data = json.load(open(p))
n = len(data["episodes"])
print(f"official json episodes: {n}")
if n < 150:
    raise SystemExit(f"need 150 demos, json only has {n}")
PY

if [ ! -d "data/pkl_data/${TASK}_Agent0/episode149" ]; then
    echo "=== parse h5 -> pkl (3 agents) ==="
    python script/parse_h5_to_pkl_multi.py \
        --task_name "${TASK}" --load_num "${LOAD_NUM}" --agent_num "${N_AGENTS}" \
        | tee "${LOG_DIR}/parse_h5.log"
else
    echo "=== skip h5 -> pkl (episode149 already present) ==="
fi

if [ ! -d "${ZARR}" ]; then
    echo "=== pack pkl -> multi-agent zarr ==="
    python script/parse_pkl_to_zarr_multi.py \
        --task_name "${TASK}" --load_num "${LOAD_NUM}" --agent_num "${N_AGENTS}" \
        | tee "${LOG_DIR}/parse_zarr.log"
else
    echo "=== skip zarr pack (${ZARR} exists) ==="
fi

python - <<PY
import zarr, sys
r = zarr.open("${ZARR}", "r")
n = int(r.attrs.get("siglip_n_image_tokens", 0))
print(f"cached SigLIP tokens: {n}")
sys.exit(0 if n == 197 else 1)
PY
siglip_ok=$?
if [ "${siglip_ok}" -ne 0 ]; then
    echo "=== cache SigLIP at 14x14 ==="
    python script/precompute_siglip_features.py \
        --zarr_path "${ZARR}" \
        --pool_grid 14 \
        --overwrite \
        --device cuda \
        --batch_size 64 | tee "${LOG_DIR}/precompute_siglip.log"
else
    echo "=== skip SigLIP cache (197 tokens already present) ==="
fi

for agent in 0 1 2; do
    ckpt="checkpoints/${TASK}_ctx_Agent${agent}_${LOAD_NUM}/100.ckpt"
    if [ -f "${ckpt}" ]; then
        echo "=== skip Stage 1 agent ${agent} (${ckpt} exists) ==="
        continue
    fi
    echo "=== Stage 1 agent ${agent} ==="
    bash policy/Diffusion-Policy/train_cls_stage1.sh \
        "${TASK}" "${LOAD_NUM}" "${agent}" "${N_AGENTS}" "${SEED}" "${GPU}" \
        | tee "${LOG_DIR}/stage1_agent${agent}.log"
    if [ ! -f "${ckpt}" ]; then
        echo "MISSING ${ckpt}"
        exit 1
    fi
    if ! grep -q -- "-> PASS" "${LOG_DIR}/stage1_agent${agent}.log"; then
        echo "Stage 1 gate FAILED for agent ${agent}; not starting Stage 2"
        exit 1
    fi
done

for agent in 0 1 2; do
    ckpt="checkpoints/${TASK}_clsdp_Agent${agent}_${LOAD_NUM}/100.ckpt"
    if [ -f "${ckpt}" ]; then
        echo "=== skip Stage 2 agent ${agent} (${ckpt} exists) ==="
        continue
    fi
    echo "=== Stage 2 agent ${agent} ==="
    bash policy/Diffusion-Policy/train_cls_dp.sh \
        "${TASK}" "${LOAD_NUM}" "${agent}" "${N_AGENTS}" "${SEED}" "${GPU}" \
        | tee "${LOG_DIR}/stage2_agent${agent}.log"
    if [ ! -f "${ckpt}" ]; then
        echo "MISSING ${ckpt}"
        exit 1
    fi
done

echo "=== Study B / ThreeRobotsStackCube training finished; starting eval ==="
bash policy/Diffusion-Policy/eval_cls_sweep.sh \
    "${TASK}" "${CONFIG}" "${LOAD_NUM}" 100 1000 1099 "${JOBS}" "${MAX_STEPS}" \
    | tee "${LOG_DIR}/eval_sweep.log"

echo "=== Study B / ThreeRobotsStackCube eval finished ==="
cat "${LOG_DIR}/eval_sweep.log" | tail -10
