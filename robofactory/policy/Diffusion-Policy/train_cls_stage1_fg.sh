#!/usr/bin/env bash
# CLS-DP Stage 1, FACTORIZED variant (Study FG): train the contextualizer for one agent.
#
# Same as train_cls_stage1_det.sh but with cls_stage1_fg.yaml, which inherits the
# deterministic config and adds the z_self / z_team split plus both measurement probes.
# Checkpoints land under checkpoints/{task}_ctxfg_Agent{i}_{n}/ so Study B and Study DET
# artifacts are untouched.
#
# Usage:
#   bash policy/Diffusion-Policy/train_cls_stage1_fg.sh ${task_name} ${load_num} ${agent_id} ${n_agents} ${seed} ${gpu_id} [hydra overrides...]
# Example:
#   bash policy/Diffusion-Policy/train_cls_stage1_fg.sh LiftBarrier-rf 150 0 2 42 0
#   bash policy/Diffusion-Policy/train_cls_stage1_fg.sh LiftBarrier-rf 150 0 2 42 0 horizon=h25
# CONFIG_NAME selects a convenience yaml (default cls_stage1_fg).
#
# Owns checkpoints/{task}_ctxfg*_Agent{i}_{n}/. FORCE_OVERWRITE_CTX=1 to regenerate.
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
config_name=${CONFIG_NAME:-cls_stage1_fg}
exp_name=${task_name}-cls-stage1-fg

if [ "${n_agents}" -lt 2 ]; then
    echo -e "\033[31mfactorization needs at least 2 agents, got ${n_agents}\033[0m"
    exit 1
fi

zarr_path="data/zarr_data/${task_name}_multi_${load_num}.zarr"
if [ ! -d "${zarr_path}" ]; then
    echo -e "\033[31mmissing ${zarr_path}\033[0m"
    echo "Run script/parse_pkl_to_zarr_multi.py and script/precompute_siglip_features.py first."
    exit 1
fi

# Default FG prefix; CONFIG_NAME=cls_stage1_fg_h25 writes *_ctxfgh25_* instead.
case "${config_name}" in
    cls_stage1_fg_h25) ctx_tag=ctxfgh25 ;;
    *) ctx_tag=ctxfg ;;
esac
final_ckpt="checkpoints/${task_name}_${ctx_tag}_Agent${agent_id}_${load_num}/100.ckpt"
refuse_overwrite_ckpt "${final_ckpt}" "Study FG Stage 1 (*_${ctx_tag}_*)"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mStage 1 contextualizer [FACTORIZED] | ${task_name} agent ${agent_id}/${n_agents}\033[0m"

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
