#!/usr/bin/env bash
# CLS-DP Stage 2, FLOW-MATCHING variant (Study FM): train the action-expert for one agent.
#
# This reuses Study B's contextualizer (*_ctx_*) exactly and changes only the Stage 2
# transport between noise and actions from a 100-step DDPM reverse chain to a handful of
# flow-matching Euler steps. It never trains Stage 1 — that would clobber Study B.
#
# Usage:
#   bash policy/Diffusion-Policy/train_cls_dp_fm.sh ${task_name} ${load_num} ${agent_id} ${n_agents} ${seed} ${gpu_id} [${ctx_epoch}]
# Example:
#   bash policy/Diffusion-Policy/train_cls_dp_fm.sh LiftBarrier-rf 150 0 2 42 0
#   bash policy/Diffusion-Policy/train_cls_dp_fm.sh LiftBarrier-rf 150 1 2 42 0
#
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib_ckpt_guard.sh
source "${SCRIPT_DIR}/lib_ckpt_guard.sh"

task_name=${1}
load_num=${2}
agent_id=${3}
n_agents=${4}
seed=${5:-42}
gpu_id=${6:-0}
ctx_epoch=${7:-100}

DEBUG=False
config_name=cls_dp
exp_name=${task_name}-cls-dp-fm

zarr_path="data/zarr_data/${task_name}_multi_${load_num}.zarr"
ctx_ckpt="checkpoints/${task_name}_ctx_Agent${agent_id}_${load_num}/${ctx_epoch}.ckpt"

require_ckpt "${ctx_ckpt}" \
    "Reuse Study B Stage 1 (*_ctx_*). Run train_study_b.sh / train_cls_stage1.sh first — do not retrain Stage 1 from Study FM."

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mStage 2 action-expert [FLOW-MATCHING] | ${task_name} agent ${agent_id}/${n_agents}\033[0m"
echo -e "\033[33mfrozen prior: ${ctx_ckpt}\033[0m"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

python ./policy/Diffusion-Policy/train.py --config-name=${config_name}.yaml \
    sampler=flow \
    action_space=raw \
    task_name=${task_name} \
    agent_id=${agent_id} \
    n_agents=${n_agents} \
    data_num=${load_num} \
    task.dataset.zarr_path="${zarr_path}" \
    contextualizer_ckpt="${ctx_ckpt}" \
    training.debug=$DEBUG \
    training.seed=${seed} \
    training.device="cuda:0" \
    exp_name=${exp_name}

