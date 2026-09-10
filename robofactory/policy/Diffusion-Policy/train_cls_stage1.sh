#!/usr/bin/env bash
# CLS-DP Stage 1: train the contextualizer for one agent.
#
# Usage:
#   bash policy/Diffusion-Policy/train_cls_stage1.sh ${task_name} ${load_num} ${agent_id} ${n_agents} ${seed} ${gpu_id} [hydra overrides...]
# Example (LiftBarrier has 2 agents, so run it twice):
#   bash policy/Diffusion-Policy/train_cls_stage1.sh LiftBarrier-rf 150 0 2 42 0
#   bash policy/Diffusion-Policy/train_cls_stage1.sh LiftBarrier-rf 150 1 2 42 0
# Longer horizon (Study B + h25):
#   CONFIG_NAME=cls_stage1_h25 bash policy/Diffusion-Policy/train_cls_stage1.sh LiftBarrier-rf 150 0 2 42 0
#   bash policy/Diffusion-Policy/train_cls_stage1.sh LiftBarrier-rf 150 0 2 42 0 horizon=h25
#
# CONFIG_NAME selects a convenience yaml (default cls_stage1).
#
# Prerequisites:
#   python script/generate_instructions.py --all
#   python script/parse_h5_to_pkl_multi.py --task_name ${task_name} --load_num N --agent_num N_AGENTS
#   python script/parse_pkl_to_zarr_multi.py --task_name ${task_name} --load_num N --agent_num N_AGENTS
#   python script/precompute_siglip_features.py --zarr_path data/zarr_data/${task_name}_multi_N.zarr --pool_grid 14
#
# Owns Stage 1 prefixes launched through this entrypoint:
#   cls_stage1      -> checkpoints/{task}_ctx_Agent{i}_{n}/
#   cls_stage1_h25  -> checkpoints/{task}_ctxh25_Agent{i}_{n}/
#   cls_stage1_bfg  -> checkpoints/{task}_ctxbfg_Agent{i}_{n}/   (Study B-FG)
#   + horizon=h25   -> the same with h25 appended, e.g. *_ctxbfgh25_*
# The guarded path is resolved through Hydra, so overrides are always accounted for.
# Study FM reuses *_ctx_* and must never retrain them. Set FORCE_OVERWRITE_CKPT=1 only
# when intentionally regenerating a Stage 1 final.
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

DEBUG=False
config_name=${CONFIG_NAME:-cls_stage1}
exp_name=${task_name}-cls-stage1

zarr_path="data/zarr_data/${task_name}_multi_${load_num}.zarr"
if [ ! -d "${zarr_path}" ]; then
    echo -e "\033[31mmissing ${zarr_path}\033[0m"
    echo "Run script/parse_pkl_to_zarr_multi.py and script/precompute_siglip_features.py first."
    exit 1
fi

# Guard the path this run will actually write. Resolved through Hydra rather than a tag
# table so group overrides (horizon=h25, ...) are accounted for; the old table guarded
# *_ctx_* while a bare `horizon=h25` wrote *_ctxh25_*.
ckpt_name=$(resolve_checkpoint_name \
    "${SCRIPT_DIR}/diffusion_policy/config" "${config_name}" \
    "task_name=${task_name}" "agent_id=${agent_id}" \
    "n_agents=${n_agents}" "data_num=${load_num}" "${@:7}")
final_ckpt="checkpoints/${ckpt_name}/100.ckpt"
refuse_overwrite_ckpt "${final_ckpt}" "Stage 1 (${ckpt_name})"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mStage 1 contextualizer | ${task_name} agent ${agent_id}/${n_agents} | config=${config_name}\033[0m"

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
    exp_name=${exp_name} \
    "${@:7}"
