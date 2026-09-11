#!/usr/bin/env bash
# Study B-FG: Study B recipe + factorized latent (stochastic prior), either Stage 2 head.
#
# Clean answer to "does factorization help?" — same Adam, 14x14 SigLIP, 150 LiftBarrier
# demos, and stochastic CVAE as Study B; only the latent is split into z_self / z_team.
#
# This is NOT Study FG (`train_study_fg.sh`), which stacks the split on Study DET's
# deterministic latent. DET alone cost 12 points, which is what made Study FG's 55%
# uninterpretable. Existing *_ctxfg_* / *_clsdpfg_* checkpoints are left alone.
#
#   Stage 1  cls_stage1_bfg.yaml                    -> *_ctxbfg_*     (shared by both heads)
#   Stage 2  cls_dp_bfg.yaml + SAMPLER              -> *_clsdpbfg_*   (SAMPLER=ddpm, default)
#                                                   -> *_clsdpbfgfm_* (SAMPLER=flow)
#
# HORIZON=h25 moves both stages to 25-step chunks, giving the full stack of everything that
# has looked good so far — Study B's stochastic CVAE + factorized latent + flow head +
# 25-step chunks:
#   Stage 1  cls_stage1_bfg.yaml horizon=h25        -> *_ctxbfgh25_*
#   Stage 2  cls_dp_bfg.yaml sampler=flow horizon=h25 -> *_clsdpbfgfmh25_*
#
#   bash policy/Diffusion-Policy/train_study_bfg.sh                        # bfg, ddpm, h8
#   SAMPLER=flow bash policy/Diffusion-Policy/train_study_bfg.sh           # bfg, flow, h8
#   SAMPLER=flow HORIZON=h25 bash policy/Diffusion-Policy/train_study_bfg.sh
#
# Stage 1 depends on the horizon (it reconstructs n_future_states) but not on the sampler,
# so priors are shared across SAMPLER and separate across HORIZON. Every stage skips agents
# whose final already exists, so re-invoking with a different SAMPLER trains only the new
# Stage 2 head.
#
# On attribution: the h25 stack moves three things at once against Study B and is a
# "does the combination win" run, not an ablation. Study FG's 55% showed factorization
# helping even on DET's weaker base, and Study FM already beats Study B on its own, so the
# individual factorization ablation (SAMPLER=ddpm, h8) is optional rather than a
# prerequisite — but it is the only run that isolates the split, and it is cheap once
# *_ctxbfg_* exists.
#
# Reuses the Study B SigLIP cache. Does not start until you run this script.
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

# ddpm = factorization ablation against Study B (default).
# flow = same prior, flow-matching Stage 2 head only.
SAMPLER=${SAMPLER:-ddpm}
# h8 = Study B's chunk length (default). h25 = 25-step chunks, own Stage 1.
HORIZON=${HORIZON:-h8}

case "${SAMPLER}" in
    ddpm) HEAD_TAG="" ;;
    flow) HEAD_TAG=fm ;;
    *)
        echo "SAMPLER must be ddpm or flow, got: ${SAMPLER}"
        exit 1
        ;;
esac

# Mirrors checkpoint_name = clsdp{latent}{head}{space}{horizon}{run} in cls_dp.yaml.
# MAX_STEPS keeps the env-step budget equal to Study B's 250 cycles x 6 executed steps;
# at h25 the executed slice is 23, so 1500/23 ~ 65 cycles buys the same wall-clock episode.
case "${HORIZON}" in
    h8)
        HORIZON_ARGS=()
        CTX_TAG_BASE=ctxbfg
        HORIZON_TAG=""
        MAX_STEPS=250
        ;;
    h25)
        HORIZON_ARGS=(horizon=h25)
        CTX_TAG_BASE=ctxbfgh25
        HORIZON_TAG=h25
        MAX_STEPS=65
        ;;
    *)
        echo "HORIZON must be h8 or h25, got: ${HORIZON}"
        exit 1
        ;;
esac

CTX_TAG_NAME=${CTX_TAG_BASE}
STAGE2_TAG=clsdpbfg${HEAD_TAG}${HORIZON_TAG}

ZARR="data/zarr_data/${TASK}_multi_${DEMOS}.zarr"
LOG_DIR="data/outputs/study_bfg${HORIZON_TAG:+_${HORIZON_TAG}}"
mkdir -p "${LOG_DIR}"

if [ ! -d "${ZARR}" ]; then
    echo "missing ${ZARR}"
    exit 1
fi
python - <<PY
import zarr, sys
root = zarr.open("${ZARR}", mode="r")
n = int(root.attrs.get("siglip_n_image_tokens", -1))
print(f"SigLIP cache: {n} image tokens")
if n != 197:
    print("expected 197 (1 + 14x14); re-run precompute_siglip_features.py --pool_grid 14")
    sys.exit(1)
PY

echo "=== Study B-FG: SAMPLER=${SAMPLER} HORIZON=${HORIZON} -> *_${STAGE2_TAG}_* (stochastic base + z_self/z_team split) ==="

# Stage 1 is head-agnostic, so a second SAMPLER invocation must not retrain it — both arms
# have to provably share one prior, and train_cls_stage1.sh would refuse the overwrite and
# abort anyway. It IS horizon-dependent (n_future_states), hence the separate ctx tag.
export CONFIG_NAME=cls_stage1_bfg
for agent in $(seq 0 $((AGENTS - 1))); do
    ckpt="checkpoints/${TASK}_${CTX_TAG_NAME}_Agent${agent}_${DEMOS}/100.ckpt"
    if [ -f "${ckpt}" ]; then
        echo "=== skip Study B-FG Stage 1 agent ${agent} (${ckpt} exists) ==="
        continue
    fi
    echo "=== Study B-FG: Stage 1 agent ${agent} (${HORIZON}) ==="
    bash policy/Diffusion-Policy/train_cls_stage1.sh \
        "${TASK}" "${DEMOS}" "${agent}" "${AGENTS}" "${SEED}" "${GPU}" \
        "${HORIZON_ARGS[@]}" \
        | tee "${LOG_DIR}/stage1_agent${agent}.log"
done

for agent in $(seq 0 $((AGENTS - 1))); do
    ckpt="checkpoints/${TASK}_${CTX_TAG_NAME}_Agent${agent}_${DEMOS}/100.ckpt"
    if [ ! -f "${ckpt}" ]; then
        echo "MISSING ${ckpt} — Stage 1 agent ${agent} did not finish"
        exit 1
    fi
done

echo
echo "=== Study B-FG: Stage 1 gates ==="
echo "Three lines per agent. Read them in order:"
echo "  teammate reconstruction  -- the usual gate, measured on prior + residual"
echo "  prior-only               -- what Stage 2 and deployment actually get"
echo "  leak probe               -- z_self should NOT predict teammates"
grep -h "Stage 1 gate" "${LOG_DIR}"/stage1_agent*.log || echo "(no gate lines found)"
echo

# One Stage 2 yaml; SAMPLER flips DDPM <-> flow via the Hydra sampler group.
# The flow head inherits its current defaults: uniform sigma, no clamp, 30 steps. The
# existing *_clsdpbfgfmh25_* checkpoints predate that revert and trained under Beta(1.5,1);
# train_study_bfg_fm_h25_uniform.sh is the uniform arm, as a Stage-2-only re-run.
export CONFIG_NAME=cls_dp_bfg
export CTX_TAG=${CTX_TAG_NAME}
export SAMPLER
# shellcheck disable=SC2206
extra_overrides=( "${HORIZON_ARGS[@]}" ${EXTRA_OVERRIDES:-} )
for agent in $(seq 0 $((AGENTS - 1))); do
    ckpt="checkpoints/${TASK}_${STAGE2_TAG}_Agent${agent}_${DEMOS}/100.ckpt"
    if [ -f "${ckpt}" ]; then
        echo "=== skip Study B-FG Stage 2 agent ${agent} (${ckpt} exists) ==="
        continue
    fi
    echo "=== Study B-FG: Stage 2 agent ${agent} (cls_dp_bfg sampler=${SAMPLER} ${HORIZON}) ==="
    bash policy/Diffusion-Policy/train_cls_dp.sh \
        "${TASK}" "${DEMOS}" "${agent}" "${AGENTS}" "${SEED}" "${GPU}" 100 \
        "${extra_overrides[@]}" \
        | tee "${LOG_DIR}/stage2_${SAMPLER}_agent${agent}.log"
    if [ ! -f "${ckpt}" ]; then
        echo "MISSING ${ckpt} — Stage 2 agent ${agent} did not finish"
        exit 1
    fi
done

echo "=== Study B-FG training finished (SAMPLER=${SAMPLER} HORIZON=${HORIZON}) ==="
ls -l checkpoints/${TASK}_${CTX_TAG_NAME}_Agent*_${DEMOS}/100.ckpt \
      checkpoints/${TASK}_${STAGE2_TAG}_Agent*_${DEMOS}/100.ckpt

cat <<EOF

Next, evaluate on the same 100 unseen seeds Study B used:

  bash policy/Diffusion-Policy/eval_cls_sweep.sh \\
      ${TASK} configs/table/lift_barrier.yaml ${DEMOS} 100 1000 1099 10 ${MAX_STEPS} ${STAGE2_TAG}

max_steps=${MAX_STEPS} holds the env-step budget equal to Study B's 250 cycles x 6 executed
steps. The flow arm needs no trailing step argument: 30 is baked into the checkpoint config.
Pass one only to re-sweep a single checkpoint across solver budgets.

Reference numbers, all on those same 100 seeds:
  Study B    (stochastic, monolithic, DDPM h8)     61%
  Study FM   (stochastic, monolithic, flow h8)     67%  at 30 steps
  Study DET  (deterministic, monolithic, DDPM h8)  49%
  Study FG   (deterministic, factorized, DDPM h8)  55%  <-- +6 over DET on DET's weaker base

Two caveats worth writing down before reading this run:

  1. SAMPLER=flow HORIZON=h25 moves three things at once against Study B (factorized
     latent, flow head, 25-step chunks). It answers "is the combination the best policy we
     have", not "which part did the work". The h8 arms are what isolate the split, and
     SAMPLER=ddpm HORIZON=h8 is cheap once *_ctxbfg_* exists.
  2. Study FM's 67% and anything trained now share uniform sigma, so it is a clean control.
     The Beta experiment (train_study_fm_v2.sh) scored 46% and the default was reverted;
     *_clsdpbfgfmh25_* is the one checkpoint still carrying it.

If the leak probe still says NOT SEPARATED, treat success-rate movement cautiously — the
split may still be cosmetic.
EOF
