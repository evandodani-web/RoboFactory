#!/usr/bin/env bash
# CLS-DP Stage 2, DETERMINISTIC variant: train the action-expert for one agent.
#
# Identical to train_cls_dp.sh except it uses cls_dp_det.yaml and pairs with the
# deterministic Stage 1 checkpoint. Checkpoints land under
# checkpoints/{task}_clsdpdet_Agent{i}_{n}/.
#
# Usage:
#   bash policy/Diffusion-Policy/train_cls_dp_det.sh ${task_name} ${load_num} ${agent_id} ${n_agents} ${seed} ${gpu_id} [${ctx_epoch}]
# Example:
#   bash policy/Diffusion-Policy/train_cls_dp_det.sh LiftBarrier-rf 150 0 2 42 0
set -euo pipefail

task_name=${1}
load_num=${2}
agent_id=${3}
n_agents=${4}
seed=${5:-42}
gpu_id=${6:-0}
ctx_epoch=${7:-100}

DEBUG=False
config_name=cls_dp_det
exp_name=${task_name}-cls-dp-det

zarr_path="data/zarr_data/${task_name}_multi_${load_num}.zarr"
ctx_ckpt="checkpoints/${task_name}_ctxdet_Agent${agent_id}_${load_num}/${ctx_epoch}.ckpt"

if [ ! -f "${ctx_ckpt}" ]; then
    echo -e "\033[31mmissing deterministic contextualizer checkpoint ${ctx_ckpt}\033[0m"
    echo "Run train_cls_stage1_det.sh for this agent first, or pass a different ctx_epoch."
    exit 1
fi

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mStage 2 action-expert [DETERMINISTIC] | ${task_name} agent ${agent_id}/${n_agents}\033[0m"
echo -e "\033[33mfrozen prior: ${ctx_ckpt}\033[0m"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

python ./policy/Diffusion-Policy/train.py --config-name=${config_name}.yaml \
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
