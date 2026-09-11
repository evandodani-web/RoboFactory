#!/usr/bin/env bash
# Study B-FG-FM-H25, uniform-sigma re-run.
#
# Stage 2 ONLY. The existing *_clsdpbfgfmh25_* checkpoints trained under sigma_dist=beta,
# which was the flow default at the time and has since been reverted. Beta lost on
# LiftBarrier by 21 points (46/100 vs uniform's 67/100 on Study FM's monolithic prior), so
# the combined stack is currently carrying a known handicap that nothing at eval can undo:
#
#   sample_sigma() is reached from compute_loss() and from nowhere else. The sampler's grid
#   comes from sigma_schedule(), a linspace that never consults sigma_dist. The training
#   distribution is baked into the weights, so a uniform version is a new Stage 2 run.
#
# Everything else is held: same *_ctxbfgh25_* priors, same seed, same demos, same U-Net,
# same 25-step chunks, same stochastic factorized latent. Only sigma_dist moves.
#
#   Stage 1  (none — reuses *_ctxbfgh25_* from train_study_bfg.sh)
#   Stage 2  cls_dp_bfg.yaml sampler=flow horizon=h25 run_tag=uni -> *_clsdpbfgfmh25uni_*
#
#   bash policy/Diffusion-Policy/train_study_bfg_fm_h25_uniform.sh
#
# run_tag=uni is load-bearing. Without it this composes to `clsdpbfgfmh25` and would
# overwrite the Beta run in place, destroying the only paired comparison this study has.
# train_cls_dp.sh refuses that outright, but the tag is what makes the run correct rather
# than merely blocked.
#
# Roughly 14 GPU-hours for both agents. Does not start until you run this script.
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

CTX_TAG_NAME=ctxbfgh25
STAGE2_TAG=clsdpbfgfmh25uni
BETA_TAG=clsdpbfgfmh25
MAX_STEPS=65

LOG_DIR="data/outputs/study_bfg_h25_uniform"
mkdir -p "${LOG_DIR}"

echo "=== Study B-FG-FM-H25 (uniform sigma): reuses *_${CTX_TAG_NAME}_* -> *_${STAGE2_TAG}_* ==="

# Stage 1 belongs to train_study_bfg.sh and is never retrained here. It is a CVAE prior with
# no flow head in it at all, so the sigma change does not touch it. Fail loudly rather than
# silently training an untrained prior, which the Stage 2 config would otherwise permit.
for agent in $(seq 0 $((AGENTS - 1))); do
    ckpt="checkpoints/${TASK}_${CTX_TAG_NAME}_Agent${agent}_${DEMOS}/100.ckpt"
    if [ ! -f "${ckpt}" ]; then
        echo "MISSING ${ckpt} — this run reuses the B-FG h25 Stage 1 and cannot train it."
        echo "Run: SAMPLER=flow HORIZON=h25 bash policy/Diffusion-Policy/train_study_bfg.sh"
        exit 1
    fi
done

echo "Reusing B-FG h25 Stage 1 priors (unmodified):"
# shellcheck source=lib_ckpt_guard.sh
source policy/Diffusion-Policy/lib_ckpt_guard.sh
for agent in $(seq 0 $((AGENTS - 1))); do
    print_ckpt_fingerprint "checkpoints/${TASK}_${CTX_TAG_NAME}_Agent${agent}_${DEMOS}/100.ckpt"
done
echo

# sigma_dist=uniform is the default again, so these overrides are redundant today. They are
# spelled out anyway: this run's whole identity is its transport config, and a future
# default change should not silently retag itself as the uniform arm.
export CONFIG_NAME=cls_dp_bfg
export CTX_TAG=${CTX_TAG_NAME}
export SAMPLER=flow
# shellcheck disable=SC2206
extra_overrides=(
    horizon=h25
    run_tag=uni
    policy.sigma_dist=uniform
    policy.sigma_dist_scale=1.0
    policy.clamp_x1=null
    ${EXTRA_OVERRIDES:-}
)

for agent in $(seq 0 $((AGENTS - 1))); do
    ckpt="checkpoints/${TASK}_${STAGE2_TAG}_Agent${agent}_${DEMOS}/100.ckpt"
    if [ -f "${ckpt}" ]; then
        echo "=== skip Stage 2 agent ${agent} (${ckpt} exists) ==="
        continue
    fi
    echo "=== Study B-FG-FM-H25-uniform: Stage 2 agent ${agent} ==="
    bash policy/Diffusion-Policy/train_cls_dp.sh \
        "${TASK}" "${DEMOS}" "${agent}" "${AGENTS}" "${SEED}" "${GPU}" 100 \
        "${extra_overrides[@]}" \
        | tee "${LOG_DIR}/stage2_agent${agent}.log"
    if [ ! -f "${ckpt}" ]; then
        echo "MISSING ${ckpt} — Stage 2 agent ${agent} did not finish"
        exit 1
    fi
done

echo "=== Study B-FG-FM-H25 (uniform) training finished ==="
ls -l checkpoints/${TASK}_${STAGE2_TAG}_Agent*_${DEMOS}/100.ckpt

cat <<EOF

The Beta arm should be untouched (nothing here writes it):

  ls -l checkpoints/${TASK}_${BETA_TAG}_Agent*_${DEMOS}/100.ckpt

Evaluate on the same 100 unseen seeds every other study used:

  bash policy/Diffusion-Policy/eval_cls_sweep.sh \\
      ${TASK} configs/table/lift_barrier.yaml ${DEMOS} 100 1000 1099 10 ${MAX_STEPS} ${STAGE2_TAG}

max_steps=${MAX_STEPS} holds the env-step budget equal to Study B's 250 cycles x 6 executed
steps; at h25 the executed slice is 23. 30 Euler steps is baked into the checkpoint config,
so no trailing step argument is needed.

Because ${BETA_TAG} already exists and differs in exactly one training-time knob, running
both on seeds 1000-1099 gives the one clean sigma_dist ablation this project has — paired,
same seeds, same prior, same everything else. Read it with McNemar, not an independent
two-proportion test.

Reference numbers, all on those same 100 seeds:
  Study B     (stochastic, monolithic, DDPM h8)    61%
  Study B-H25 (stochastic, monolithic, DDPM h25)   66%
  Study FM    (stochastic, monolithic, flow h8)    67%  at 30 steps, uniform sigma
  Study FM-v2 (same, Beta sigma)                   46%  at 30 steps
  Study DET   (deterministic, monolithic, DDPM h8) 49%
  Study FG    (deterministic, factorized, DDPM h8) 55%

Two things worth doing before spending another 14 GPU-hours, both free:

  1. Sweep shift on the existing Beta checkpoint. shift is applied in sigma_schedule as well
     as sample_sigma, and training ran at 1.0, so it is a pure inference knob. shift<1 pulls
     the grid toward sigma=0, where Beta training left the field weakest; shift=0.3 cut the
     FM-v2 checkpoint's excess chunk jerk from 1.88x the demonstrations to 1.62x, against
     uniform's 1.55x. If that recovers the success rate, this retrain is unnecessary.

  2. Sweep solver=heun. Never tested, second-order, also free at eval.

Note that open-loop action MSE will not adjudicate any of this — it moves 3% across step
counts that swing success by 47 points. Chunk jerk tracks it; MSE does not.
EOF
