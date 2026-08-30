#!/usr/bin/env bash
# Study B: LiftBarrier, 150 demos, Adam, full 14x14 SigLIP.
# Recaches SigLIP if needed, then Stage 1 (both agents) then Stage 2 (both agents).
# Study A checkpoints (*_100) are left untouched.
set -euo pipefail

REPO_ROOT=/workspace/RoboFactory
cd "${REPO_ROOT}/robofactory"
# train.py lives under policy/Diffusion-Policy and expects that as cwd
# (Hydra config_path, relative checkpoint paths). The existing train_*.sh
# scripts are meant to be launched from robofactory/.
source "${REPO_ROOT}/.venv/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

ZARR="data/zarr_data/LiftBarrier-rf_multi_150.zarr"
LOG_DIR="data/outputs/study_b"
mkdir -p "${LOG_DIR}"

echo "=== Study B: recache SigLIP at 14x14 ==="
python script/precompute_siglip_features.py \
    --zarr_path "${ZARR}" \
    --pool_grid 14 \
    --overwrite \
    --device cuda \
    --batch_size 64 | tee "${LOG_DIR}/precompute_siglip.log"

echo "=== Study B: Stage 1 agent 0 ==="
bash policy/Diffusion-Policy/train_cls_stage1.sh LiftBarrier-rf 150 0 2 42 0 \
    | tee "${LOG_DIR}/stage1_agent0.log"

echo "=== Study B: Stage 1 agent 1 ==="
bash policy/Diffusion-Policy/train_cls_stage1.sh LiftBarrier-rf 150 1 2 42 0 \
    | tee "${LOG_DIR}/stage1_agent1.log"

for agent in 0 1; do
    ckpt="checkpoints/LiftBarrier-rf_ctx_Agent${agent}_150/100.ckpt"
    if [ ! -f "${ckpt}" ]; then
        echo "MISSING ${ckpt} — Stage 1 agent ${agent} did not finish"
        exit 1
    fi
done

echo "=== Study B: Stage 2 agent 0 ==="
bash policy/Diffusion-Policy/train_cls_dp.sh LiftBarrier-rf 150 0 2 42 0 \
    | tee "${LOG_DIR}/stage2_agent0.log"

echo "=== Study B: Stage 2 agent 1 ==="
bash policy/Diffusion-Policy/train_cls_dp.sh LiftBarrier-rf 150 1 2 42 0 \
    | tee "${LOG_DIR}/stage2_agent1.log"

echo "=== Study B training finished ==="
ls -l checkpoints/LiftBarrier-rf_ctx_Agent0_150/100.ckpt \
      checkpoints/LiftBarrier-rf_ctx_Agent1_150/100.ckpt \
      checkpoints/LiftBarrier-rf_clsdp_Agent0_150/100.ckpt \
      checkpoints/LiftBarrier-rf_clsdp_Agent1_150/100.ckpt
