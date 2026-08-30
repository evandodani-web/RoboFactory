#!/usr/bin/env bash
# CLS-DP evaluation sweep: 100 unseen seeds, matching the paper's protocol of
# "100 episodes with unseen seeds that introduce varied target object placements".
#
# Seeds are independent episodes, so they run in parallel. The simulator and the
# TOPP smoother are CPU-bound (~8 cores per episode) and each episode holds only
# ~2GB of GPU, so throughput is set by cores, not by the GPU.
#
# Usage:
#   bash policy/Diffusion-Policy/eval_cls_sweep.sh TASK_NAME CONFIG DATA_NUM CKPT [SEED_START] [SEED_END] [JOBS]
# Example:
#   bash policy/Diffusion-Policy/eval_cls_sweep.sh LiftBarrier-rf configs/table/lift_barrier.yaml 100 100
set -euo pipefail

TASK_NAME=${1}
CONFIG=${2}
DATA_NUM=${3}
CKPT=${4}
SEED_START=${5:-1000}
SEED_END=${6:-1099}
JOBS=${7:-10}

REPO_ROOT=/workspace/RoboFactory
cd "${REPO_ROOT}/robofactory"

export PYTHONPATH=${REPO_ROOT}
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export HYDRA_FULL_ERROR=1
# Each episode gets a slice of the machine; without this every worker tries to
# grab all 128 cores and they thrash.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

STAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="eval_results/${TASK_NAME}_${DATA_NUM}_${CKPT}_${STAMP}"
mkdir -p "${RUN_DIR}/logs"
RESULTS="${RUN_DIR}/results.csv"
echo "seed,success" > "${RESULTS}"

echo "task=${TASK_NAME} data_num=${DATA_NUM} ckpt=${CKPT}"
echo "seeds ${SEED_START}..${SEED_END} across ${JOBS} parallel workers"
echo "results -> ${RUN_DIR}"

run_seed() {
    local seed=$1
    local log="${RUN_DIR}/logs/seed_${seed}.log"
    set +e
    "${REPO_ROOT}/.venv/bin/python" ./policy/Diffusion-Policy/eval_multi_cls_dp.py \
        --env-id "${TASK_NAME}" \
        --config "${CONFIG}" \
        --data-num "${DATA_NUM}" \
        --checkpoint-num "${CKPT}" \
        --seed "${seed}" \
        --render-mode sensors \
        --obs-mode rgb \
        --sim-backend cpu \
        --num-envs 1 \
        --instruction-split eval \
        --quiet \
        --record-dir "./eval_video/{env_id}" > "${log}" 2>&1
    set -e

    local last
    last=$(tail -n 1 "${log}")
    local ok=0
    if [[ "${last}" == *"success"* ]]; then
        ok=1
    fi
    # flock keeps concurrent writers from interleaving inside a line
    flock "${RESULTS}" -c "echo '${seed},${ok}' >> '${RESULTS}'"
    echo "seed ${seed} -> ${ok}"
}
export -f run_seed
export TASK_NAME CONFIG DATA_NUM CKPT REPO_ROOT RUN_DIR RESULTS

seq "${SEED_START}" "${SEED_END}" | xargs -P "${JOBS}" -I{} bash -c 'run_seed {}'

TOTAL=$(( $(wc -l < "${RESULTS}") - 1 ))
SUCCESS=$(awk -F, 'NR>1 {s+=$2} END {print s+0}' "${RESULTS}")
RATE=$(awk -v s="${SUCCESS}" -v t="${TOTAL}" 'BEGIN {if (t>0) printf "%.1f", 100*s/t; else print "0.0"}')

{
    echo "task: ${TASK_NAME}"
    echo "demos: ${DATA_NUM}  checkpoint: ${CKPT}"
    echo "seeds: ${SEED_START}..${SEED_END}"
    echo "total: ${TOTAL}  success: ${SUCCESS}  success_rate: ${RATE}%"
} | tee "${RUN_DIR}/summary.txt"
