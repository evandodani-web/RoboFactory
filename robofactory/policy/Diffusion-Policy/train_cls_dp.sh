#!/usr/bin/env bash
# CLS-DP Stage 2: train the latent-conditioned action-expert for one agent.
#
# Usage:
#   bash policy/Diffusion-Policy/train_cls_dp.sh ${task_name} ${load_num} ${agent_id} ${n_agents} ${seed} ${gpu_id} [${ctx_epoch}] [hydra overrides...]
# Example:
#   bash policy/Diffusion-Policy/train_cls_dp.sh LiftBarrier-rf 150 0 2 42 0
#   bash policy/Diffusion-Policy/train_cls_dp.sh LiftBarrier-rf 150 1 2 42 0
# Study B + h25 (DDPM):
#   CONFIG_NAME=cls_dp_h25 CTX_TAG=ctxh25 bash policy/Diffusion-Policy/train_cls_dp.sh LiftBarrier-rf 150 0 2 42 0
# Study B + h25 + flow matching (same Stage 1 prior):
#   CONFIG_NAME=cls_dp_h25 CTX_TAG=ctxh25 SAMPLER=flow bash policy/Diffusion-Policy/train_cls_dp.sh LiftBarrier-rf 150 0 2 42 0
#   # or: CONFIG_NAME=cls_dp_fm_h25 CTX_TAG=ctxh25 ...
#
# CONFIG_NAME selects a convenience yaml (default cls_dp).
# CTX_TAG is the Stage 1 checkpoint family (default ctx; use ctxh25 after Stage 1 at h25).
# SAMPLER=ddpm|flow optionally overrides the Hydra sampler group (default: leave config alone).
#
# Requires Stage 1 to have been trained for this same agent first.
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
config_name=${CONFIG_NAME:-cls_dp}
exp_name=${task_name}-cls-dp
ctx_tag=${CTX_TAG:-ctx}

# Optional modular head swap. Empty keeps whatever the config's defaults say
# (DDPM for cls_dp / cls_dp_h25; flow for cls_dp_fm_h25).
sampler_args=()
if [ -n "${SAMPLER:-}" ]; then
    case "${SAMPLER}" in
        ddpm|flow) sampler_args=(sampler="${SAMPLER}") ;;
        *)
            echo -e "\033[31mSAMPLER must be ddpm or flow, got: ${SAMPLER}\033[0m"
            exit 1
            ;;
    esac
fi

zarr_path="data/zarr_data/${task_name}_multi_${load_num}.zarr"
ctx_ckpt="checkpoints/${task_name}_${ctx_tag}_Agent${agent_id}_${load_num}/${ctx_epoch}.ckpt"

if [ ! -f "${ctx_ckpt}" ]; then
    echo -e "\033[31mmissing contextualizer checkpoint ${ctx_ckpt}\033[0m"
    echo "Run train_cls_stage1.sh for this agent first, or pass a different ctx_epoch / CTX_TAG."
    exit 1
fi

# Stage 2 finals share the repo-level checkpoints/ tree, so a re-run that composes to an
# existing tag overwrites that study's result in place. Study FM's *_clsdpfm_* is the live
# example: re-running it with today's flow defaults would destroy the checkpoint behind the
# 67% number. Give the new run its own tag instead, e.g. run_tag=v2 -> *_clsdpfmv2_*.
stage2_ckpt_name=$(resolve_checkpoint_name \
    "${SCRIPT_DIR}/diffusion_policy/config" "${config_name}" \
    "${sampler_args[@]}" \
    "task_name=${task_name}" "agent_id=${agent_id}" \
    "n_agents=${n_agents}" "data_num=${load_num}" "${@:8}")
refuse_overwrite_ckpt \
    "checkpoints/${stage2_ckpt_name}/100.ckpt" "Stage 2 (${stage2_ckpt_name})"

head_label="${SAMPLER:-config-default}"
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mStage 2 action-expert | ${task_name} agent ${agent_id}/${n_agents} | config=${config_name} sampler=${head_label}\033[0m"
echo -e "\033[33mwrites: checkpoints/${stage2_ckpt_name}/\033[0m"
echo -e "\033[33mfrozen prior: ${ctx_ckpt}\033[0m"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

python ./policy/Diffusion-Policy/train.py --config-name=${config_name}.yaml \
    "${sampler_args[@]}" \
    task_name=${task_name} \
    agent_id=${agent_id} \
    n_agents=${n_agents} \
    data_num=${load_num} \
    task.dataset.zarr_path="${zarr_path}" \
    contextualizer_ckpt="${ctx_ckpt}" \
    training.debug=$DEBUG \
    training.seed=${seed} \
    training.device="cuda:0" \
    exp_name=${exp_name} \
    "${@:8}"
