#!/usr/bin/env bash
# CLS-DP decentralized evaluation.
#
# Usage:
#   bash policy/Diffusion-Policy/eval_cls_multi.sh ${config_name} ${DATA_NUM} ${CHECKPOINT_NUM} ${DEBUG_MODE} ${TASK_NAME} [${SEED}]
# Example:
#   bash policy/Diffusion-Policy/eval_cls_multi.sh configs/table/lift_barrier.yaml 150 100 1 LiftBarrier-rf 10000
#
# DEBUG_MODE=1 opens the viewer and prints per-step info.
# Each agent runs on its own camera only; the shared task instruction is sampled from the
# held-out half of the instruction bank.
set -euo pipefail

config_name=${1}
DATA_NUM=${2}
CHECKPOINT_NUM=${3}
DEBUG_MODE=${4:-0}
TASK_NAME=${5}
SEED=${6:-10000}

export HYDRA_FULL_ERROR=1

quiet_flag="--quiet"
render_mode="rgb_array"
if [ "${DEBUG_MODE}" = "1" ]; then
    quiet_flag=""
    render_mode="human"
fi

python ./policy/Diffusion-Policy/eval_multi_cls_dp.py \
    --env-id "${TASK_NAME}" \
    --config "${config_name}" \
    --data-num "${DATA_NUM}" \
    --checkpoint-num "${CHECKPOINT_NUM}" \
    --seed "${SEED}" \
    --render-mode "${render_mode}" \
    --instruction-split eval \
    ${quiet_flag}
