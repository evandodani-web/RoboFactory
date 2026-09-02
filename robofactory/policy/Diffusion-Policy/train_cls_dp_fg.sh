#!/usr/bin/env bash
# CLS-DP Stage 2, FACTORIZED variant (Study FG): train the action-expert for one agent.
#
# Factorization is internal to Stage 1 -- the prior still emits one 256-d latent and the
# action expert still consumes all of it -- so this differs from train_cls_dp_det.sh only
# in which Stage 1 checkpoint it pairs with and where it writes.
#
# Usage:
#   bash policy/Diffusion-Policy/train_cls_dp_fg.sh ${task_name} ${load_num} ${agent_id} ${n_agents} ${seed} ${gpu_id} [${ctx_epoch}]
# Example:
#   bash policy/Diffusion-Policy/train_cls_dp_fg.sh LiftBarrier-rf 150 0 2 42 0
set -euo pipefail

task_name=${1}
load_num=${2}
agent_id=${3}
n_agents=${4}
seed=${5:-42}
gpu_id=${6:-0}
ctx_epoch=${7:-100}

DEBUG=False
config_name=cls_dp_fg
exp_name=${task_name}-cls-dp-fg

zarr_path="data/zarr_data/${task_name}_multi_${load_num}.zarr"
ctx_ckpt="checkpoints/${task_name}_ctxfg_Agent${agent_id}_${load_num}/${ctx_epoch}.ckpt"

if [ ! -f "${ctx_ckpt}" ]; then
    echo -e "\033[31mmissing factorized contextualizer checkpoint ${ctx_ckpt}\033[0m"
    echo "Run train_cls_stage1_fg.sh for this agent first, or pass a different ctx_epoch."
    exit 1
fi

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mStage 2 action-expert [FACTORIZED] | ${task_name} agent ${agent_id}/${n_agents}\033[0m"
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
