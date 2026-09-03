#!/usr/bin/env bash
# CLS-DP evaluation sweep: 100 unseen seeds, matching the paper's protocol of
# "100 episodes with unseen seeds that introduce varied target object placements".
#
# Seeds are independent episodes, so they run in parallel. The simulator and the
# TOPP smoother are CPU-bound (~8 cores per episode) and each episode holds only
# ~2GB of GPU, so throughput is set by cores, not by the GPU.
#
# Usage:
#   bash policy/Diffusion-Policy/eval_cls_sweep.sh TASK_NAME CONFIG DATA_NUM CKPT [SEED_START] [SEED_END] [JOBS] [MAX_STEPS] [CKPT_PREFIX] [STEPS]
# Example (reproduced baseline):
#   bash policy/Diffusion-Policy/eval_cls_sweep.sh LiftBarrier-rf configs/table/lift_barrier.yaml 150 100
# Example (deterministic-latent variant):
#   bash policy/Diffusion-Policy/eval_cls_sweep.sh LiftBarrier-rf configs/table/lift_barrier.yaml 150 100 1000 1099 10 250 clsdpdet
# Example (flow matching, re-evaluating one trained checkpoint at 8 sampler steps):
#   bash policy/Diffusion-Policy/eval_cls_sweep.sh LiftBarrier-rf configs/table/lift_barrier.yaml 150 100 1000 1099 10 250 clsdpfm 8
set -euo pipefail

TASK_NAME=${1}
CONFIG=${2}
DATA_NUM=${3}
CKPT=${4}
SEED_START=${5:-1000}
SEED_END=${6:-1099}
JOBS=${7:-10}
MAX_STEPS=${8:-250}
# Which checkpoint family to evaluate: clsdp (baseline), clsdpdet (deterministic),
# clsdpfm (flow matching), and any tag combination the config groups produce.
CKPT_PREFIX=${9:-clsdp}
# Sampler steps. Empty keeps whatever the checkpoint was trained with; setting it lets one
# flow-matching checkpoint be evaluated across step counts without retraining.
STEPS=${10:-}

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
# Prefix and step count are part of the run dir so no two configurations overwrite.
RUN_DIR="eval_results/${TASK_NAME}_${CKPT_PREFIX}${STEPS:+_s${STEPS}}_${DATA_NUM}_${CKPT}_${STAMP}"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/timing"
RESULTS="${RUN_DIR}/results.csv"
echo "seed,success" > "${RESULTS}"

echo "task=${TASK_NAME} variant=${CKPT_PREFIX} data_num=${DATA_NUM} ckpt=${CKPT} max_steps=${MAX_STEPS} steps=${STEPS:-<from ckpt>}"
echo "seeds ${SEED_START}..${SEED_END} across ${JOBS} parallel workers"
echo "results -> ${RUN_DIR}"

run_seed() {
    local seed=$1
    local log="${RUN_DIR}/logs/seed_${seed}.log"
    local steps_arg=()
    if [[ -n "${STEPS}" ]]; then
        steps_arg=(--num-inference-steps "${STEPS}")
    fi
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
        --ckpt-prefix "${CKPT_PREFIX}" \
        --siglip-pool-grid 14 \
        --max-steps "${MAX_STEPS}" \
        --timing-json "${RUN_DIR}/timing/seed_${seed}.json" \
        "${steps_arg[@]}" \
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
export TASK_NAME CONFIG DATA_NUM CKPT REPO_ROOT RUN_DIR RESULTS MAX_STEPS CKPT_PREFIX STEPS

seq "${SEED_START}" "${SEED_END}" | xargs -P "${JOBS}" -I{} bash -c 'run_seed {}'

TOTAL=$(( $(wc -l < "${RESULTS}") - 1 ))
SUCCESS=$(awk -F, 'NR>1 {s+=$2} END {print s+0}' "${RESULTS}")
RATE=$(awk -v s="${SUCCESS}" -v t="${TOTAL}" 'BEGIN {if (t>0) printf "%.1f", 100*s/t; else print "0.0"}')

# Aggregate the per-episode timing sidecars. Sampler latency and episode wall-clock are
# reported separately: the sampler runs once per macro-cycle and the 6 executed steps that
# follow are TOPP smoothing plus simulator substeps, which no sampler change touches.
TIMING=$("${REPO_ROOT}/.venv/bin/python" - "${RUN_DIR}/timing" <<'PY'
import glob, json, os, sys
files = sorted(glob.glob(os.path.join(sys.argv[1], "*.json")))
if not files:
    print("timing: none collected")
    raise SystemExit
runs = [json.load(open(f)) for f in files]
def mean(key):
    vals = [r[key] for r in runs if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else float("nan")
per = [a for r in runs for a in r["per_agent"]]
print(f"episodes_timed: {len(runs)}")
print(f"sampler_steps: {runs[0]['num_inference_steps']}  solver: {runs[0]['solver']}")
print(f"ms_per_action: {sum(a['ms_per_action'] for a in per) / len(per):.1f}")
print(f"denoiser_calls_per_action: {sum(a['denoiser_calls_per_action'] for a in per) / len(per):.1f}")
print(f"episode_ms: {mean('episode_ms_total'):.0f}")
print(f"policy_fraction_of_episode: {mean('policy_fraction_of_episode'):.3f}")
PY
)

{
    echo "task: ${TASK_NAME}"
    echo "variant: ${CKPT_PREFIX}"
    echo "demos: ${DATA_NUM}  checkpoint: ${CKPT}"
    echo "seeds: ${SEED_START}..${SEED_END}"
    echo "total: ${TOTAL}  success: ${SUCCESS}  success_rate: ${RATE}%"
    echo "${TIMING}"
} | tee "${RUN_DIR}/summary.txt"
