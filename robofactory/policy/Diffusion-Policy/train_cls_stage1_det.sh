#!/usr/bin/env bash
# CLS-DP Stage 1, DETERMINISTIC variant: train the contextualizer for one agent.
#
# Identical to train_cls_stage1.sh except it uses cls_stage1_det.yaml, which inherits
# that config and overrides only the stochasticity. Checkpoints land under
# checkpoints/{task}_ctxdet_Agent{i}_{n}/ so the reproduced *_ctx_* runs are untouched.
#
# Usage:
#   bash policy/Diffusion-Policy/train_cls_stage1_det.sh ${task_name} ${load_num} ${agent_id} ${n_agents} ${seed} ${gpu_id}
# Example:
#   bash policy/Diffusion-Policy/train_cls_stage1_det.sh LiftBarrier-rf 150 0 2 42 0
set -euo pipefail

task_name=${1}
load_num=${2}
agent_id=${3}
n_agents=${4}
seed=${5:-42}
gpu_id=${6:-0}

DEBUG=False
config_name=cls_stage1_det
exp_name=${task_name}-cls-stage1-det

zarr_path="data/zarr_data/${task_name}_multi_${load_num}.zarr"
if [ ! -d "${zarr_path}" ]; then
    echo -e "\033[31mmissing ${zarr_path}\033[0m"
    echo "Run script/parse_pkl_to_zarr_multi.py and script/precompute_siglip_features.py first."
    exit 1
fi

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mStage 1 contextualizer [DETERMINISTIC] | ${task_name} agent ${agent_id}/${n_agents}\033[0m"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

python ./policy/Diffusion-Policy/train.py --config-name=${config_name}.yaml \
    task_name=${task_name} \
    agent_id=${agent_id} \
    n_agents=${n_agents} \
    data_num=${load_num} \
    task.dataset.zarr_path="${zarr_path}" \
    training.debug=$DEBUG \
    training.seed=${seed} \
    training.device="cuda:0" \
    exp_name=${exp_name}
