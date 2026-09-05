#!/usr/bin/env bash
# Resume the interrupted ThreeRobotsStackCube Study B eval.
# Appends only missing seeds into the existing run directory.
set -euo pipefail

REPO_ROOT=/workspace/RoboFactory
cd "${REPO_ROOT}/robofactory"
PYTHON="${REPO_ROOT}/.venv/bin/python"
if [ ! -x "${PYTHON}" ]; then
    echo "FATAL: ${PYTHON} missing or not executable (venv python symlink broken?)"
    exit 1
fi
"${PYTHON}" -c "import torch" >/dev/null

TASK_NAME="ThreeRobotsStackCube-rf"
CONFIG="configs/table/three_robots_stack_cube.yaml"
DATA_NUM=150
CKPT=100
CKPT_PREFIX=clsdp
MAX_STEPS=800
JOBS=10
SEED_START=1000
SEED_END=1099

RUN_DIR="eval_results/ThreeRobotsStackCube-rf_clsdp_150_100_20260904_175651"
RESULTS="${RUN_DIR}/results.csv"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/timing"

if [ ! -f "${RESULTS}" ]; then
    echo "seed,success" > "${RESULTS}"
fi

export PYTHONPATH=${REPO_ROOT}
# Graphics must be visible for SAPIEN/Vulkan sensor rendering. The container default
# NVIDIA_VISIBLE_DEVICES=void / compute-only caps is what broke the mid-eval resume.
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility,graphics}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Fail fast if NVIDIA Vulkan cannot initialize (otherwise workers segfault as rc=139).
"${PYTHON}" - <<'PY'
import ctypes, sys
lib = ctypes.CDLL("libGLX_nvidia.so.0")
negotiate = lib.vk_icdNegotiateLoaderICDInterfaceVersion
negotiate.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
negotiate.restype = ctypes.c_int
ver = ctypes.c_uint32(5)
rc = negotiate(ctypes.byref(ver))
if rc != 0:
    print(
        "FATAL: NVIDIA Vulkan ICD failed to initialize "
        f"(vk_icdNegotiateLoaderICDInterfaceVersion -> {rc}).\n"
        "CUDA still works, but SAPIEN needs graphics. Restart this pod/container with\n"
        "  NVIDIA_VISIBLE_DEVICES=all\n"
        "  NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics\n"
        "then re-run resume_three_robots_eval.sh. results.csv still has the 11 real seeds."
    )
    sys.exit(2)
print(f"Vulkan ICD ok (interface version {ver.value})")
PY


DONE_FILE=$(mktemp)
awk -F, 'NR>1 {print $1}' "${RESULTS}" | sort -n > "${DONE_FILE}"
MISSING=$(comm -23 <(seq "${SEED_START}" "${SEED_END}") "${DONE_FILE}")
N_MISSING=$(echo "${MISSING}" | grep -c . || true)
rm -f "${DONE_FILE}"

echo "resuming ${RUN_DIR}"
echo "already scored: $(( $(wc -l < "${RESULTS}") - 1 ))"
echo "missing seeds: ${N_MISSING}"
echo "max_steps=${MAX_STEPS} jobs=${JOBS}"

if [ "${N_MISSING}" -eq 0 ]; then
    echo "nothing to resume"
else
    run_seed() {
        local seed=$1
        local log="${RUN_DIR}/logs/seed_${seed}.log"
        set +e
        "${PYTHON}" ./policy/Diffusion-Policy/eval_multi_cls_dp.py \
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
            --quiet \
            --record-dir "./eval_video/{env_id}" > "${log}" 2>&1
        local rc=$?
        set -e

        local last
        last=$(tail -n 1 "${log}" 2>/dev/null || true)
        local ok=0
        if [[ "${last}" == *"success"* ]]; then
            ok=1
        fi
        if [ "${rc}" -ne 0 ] && [ "${ok}" -eq 0 ]; then
            echo "seed ${seed} exited rc=${rc} (counting as failure)"
        fi
        flock "${RESULTS}" -c "echo '${seed},${ok}' >> '${RESULTS}'"
        echo "seed ${seed} -> ${ok}"
    }
    export -f run_seed
    export TASK_NAME CONFIG DATA_NUM CKPT REPO_ROOT RUN_DIR RESULTS MAX_STEPS CKPT_PREFIX PYTHON

    echo "${MISSING}" | xargs -P "${JOBS}" -I{} bash -c 'run_seed {}'
fi

TOTAL=$(( $(wc -l < "${RESULTS}") - 1 ))
SUCCESS=$(awk -F, 'NR>1 {s+=$2} END {print s+0}' "${RESULTS}")
RATE=$(awk -v s="${SUCCESS}" -v t="${TOTAL}" 'BEGIN {if (t>0) printf "%.1f", 100*s/t; else print "0.0"}')

TIMING=$("${PYTHON}" - "${RUN_DIR}/timing" <<'PY'
import glob, json, os, sys
files = sorted(glob.glob(os.path.join(sys.argv[1], "*.json")))
if not files:
    print("timing: none collected")
    raise SystemExit
runs = [json.load(open(f)) for f in files]
def mean(key):
    vals = [r[key] for r in runs if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else float("nan")
per = [a for r in runs for a in r.get("per_agent", [])]
print(f"episodes_timed: {len(runs)}")
if runs and "num_inference_steps" in runs[0]:
    print(f"sampler_steps: {runs[0]['num_inference_steps']}  solver: {runs[0].get('solver')}")
if per:
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
