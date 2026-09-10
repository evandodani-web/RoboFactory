#!/usr/bin/env bash
# Study FM-v2: Study FM re-run under the current flow-matching defaults.
#
# Stage 2 ONLY. The flow head is a Stage 2 object, so this reuses Study B's existing
# *_ctx_* priors untouched and changes nothing else — same prior, same seed, same demos,
# same U-Net. The single difference against the original Study FM is the transport config:
#
#              sigma_dist        clamp_x1   default steps
#   Study FM   uniform           none       4
#   Study FM-v2 Beta(1.5, 1.0)   1.0        30
#
# Of those, only sigma_dist requires retraining; clamp and step count are inference knobs
# that were already re-measurable on the old checkpoint. So this run isolates exactly one
# thing: does pi0's high-noise-biased timestep distribution produce a field that integrates
# better, especially at low step counts?
#
#   Stage 1  (none — reuses *_ctx_* from Study B)
#   Stage 2  cls_dp.yaml sampler=flow run_tag=v2 -> *_clsdpfmv2_*
#
#   bash policy/Diffusion-Policy/train_study_fm_v2.sh
#
# run_tag=v2 is load-bearing. Today's defaults still compose to the tag `clsdpfm`, so
# without it this run would overwrite the checkpoint behind Study FM's 67% in place.
# train_cls_dp.sh now refuses that outright, but the tag is what makes the run correct
# rather than merely blocked.
set -euo pipefail

REPO_ROOT=/workspace/RoboFactory
cd "${REPO_ROOT}/robofactory"
source "${REPO_ROOT}/.venv/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

TASK=LiftBarrier-rf
DEMOS=150
AGENTS=2
SEED=42
GPU=0

STAGE2_TAG=clsdpfmv2
LOG_DIR="data/outputs/study_fm_v2"
mkdir -p "${LOG_DIR}"

echo "=== Study FM-v2: Study B priors + flow head at Beta(1.5,1) / clamp / 30 steps -> *_${STAGE2_TAG}_* ==="

# Stage 1 is Study B's and is never retrained here. Fail loudly rather than silently
# training an untrained prior, which the Stage 2 config would otherwise permit.
for agent in $(seq 0 $((AGENTS - 1))); do
    ckpt="checkpoints/${TASK}_ctx_Agent${agent}_${DEMOS}/100.ckpt"
    if [ ! -f "${ckpt}" ]; then
        echo "MISSING ${ckpt} — Study FM-v2 reuses Study B's Stage 1 and cannot train it."
        echo "Run train_study_b.sh first, or point CTX_TAG elsewhere by hand."
        exit 1
    fi
done

echo "Reusing Study B Stage 1 priors (unmodified):"
# shellcheck source=lib_ckpt_guard.sh
source policy/Diffusion-Policy/lib_ckpt_guard.sh
for agent in $(seq 0 $((AGENTS - 1))); do
    print_ckpt_fingerprint "checkpoints/${TASK}_ctx_Agent${agent}_${DEMOS}/100.ckpt"
done
echo

export CONFIG_NAME=cls_dp
export CTX_TAG=ctx
export SAMPLER=flow
# shellcheck disable=SC2206
extra_overrides=( run_tag=v2 ${EXTRA_OVERRIDES:-} )
for agent in $(seq 0 $((AGENTS - 1))); do
    ckpt="checkpoints/${TASK}_${STAGE2_TAG}_Agent${agent}_${DEMOS}/100.ckpt"
    if [ -f "${ckpt}" ]; then
        echo "=== skip Study FM-v2 Stage 2 agent ${agent} (${ckpt} exists) ==="
        continue
    fi
    echo "=== Study FM-v2: Stage 2 agent ${agent} (cls_dp sampler=flow run_tag=v2) ==="
    bash policy/Diffusion-Policy/train_cls_dp.sh \
        "${TASK}" "${DEMOS}" "${agent}" "${AGENTS}" "${SEED}" "${GPU}" 100 \
        "${extra_overrides[@]}" \
        | tee "${LOG_DIR}/stage2_agent${agent}.log"
    if [ ! -f "${ckpt}" ]; then
        echo "MISSING ${ckpt} — Stage 2 agent ${agent} did not finish"
        exit 1
    fi
done

echo "=== Study FM-v2 training finished ==="
ls -l checkpoints/${TASK}_${STAGE2_TAG}_Agent*_${DEMOS}/100.ckpt

cat <<EOF

Verify Study FM's original checkpoints are untouched (they should be, nothing wrote them):

  ls -l checkpoints/${TASK}_clsdpfm_Agent*_${DEMOS}/100.ckpt

Evaluate on the same 100 unseen seeds every other study used:

  bash policy/Diffusion-Policy/eval_cls_sweep.sh \\
      ${TASK} configs/table/lift_barrier.yaml ${DEMOS} 100 1000 1099 10 250 ${STAGE2_TAG}

30 steps is baked into the checkpoint config, so no trailing step argument is needed.

The comparison that makes this run worth its GPU time is the step curve, not the single
number. Study FM at matched step counts was:

    steps   4     8     12    24    30
    SR      20%   28%   40%   60%   67%

Re-sweeping FM-v2 across the same counts says whether Beta(1.5,1) bought low-step accuracy,
which is the entire hypothesis:

  for s in 4 8 12 24 30; do
    bash policy/Diffusion-Policy/eval_cls_sweep.sh \\
        ${TASK} configs/table/lift_barrier.yaml ${DEMOS} 100 1000 1099 10 250 ${STAGE2_TAG} \$s
  done

A flat curve that only matches at 30 means the sigma change did nothing for integrability
and the next lever is the solver (heun/midpoint) or reflow, not the schedule.
EOF
