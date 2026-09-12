#!/usr/bin/env bash
# Study B-H25: Study B recipe + 25-step action / privileged-future windows.
#
# Clean answer to "does longer chunking help?" — same latent, Adam, 14x14 SigLIP,
# and 150 LiftBarrier demos as Study B; only horizon changes.
#
#   Stage 1  cls_stage1_h25.yaml           -> *_ctxh25_*
#   Stage 2  cls_dp_h25.yaml + SAMPLER     -> *_clsdph25_*   (SAMPLER=ddpm, default)
#                                           -> *_clsdpfmh25_* (SAMPLER=flow)
#
# Modular Stage 2 head flag (Stage 1 is identical either way):
#   bash policy/Diffusion-Policy/train_study_b_h25.sh
#   SAMPLER=flow bash policy/Diffusion-Policy/train_study_b_h25.sh
#
# SAMPLER=flow is now the deciding run for the combo regression. Study B-FG-FM-H25 stacks
# the factorized latent, the flow head and h25, and lost to all three of its parts
# (49% uniform / 38% Beta against B-H25's 66% and FM's 67%; McNemar z=2.6 and 2.5, so the
# deficit is real). Of those three components, two are measured and one never was:
#
#   flow head at h8   Study FM     67%   indistinguishable from B-H25 (z=0.00)
#   h25 at DDPM       Study B-H25  66%
#   factorized latent on a stochastic base           NEVER MEASURED
#   flow head at h25                                 NEVER MEASURED  <- this run
#
# *_ctxh25_* already exists, so this is Stage 2 only and splits the combo's 17-point
# deficit between the two untested components:
#
#   lands near 66%  ->  flow survives h25; factorization is what broke the combo
#   lands near 49%  ->  flow@h25 is what broke it; factorization is exonerated
#
# The factorization suspicion is not just process of elimination: the *_ctxbfgh25_* Stage 1
# leak probe reported NOT SEPARATED at ratio 1.21-1.26x, meaning z_self still predicts
# teammates about as well as the full prior. The split cost capacity and bought nothing.
#
# Receding-horizon ablation (predict 25, act 6):
#   EXTRA_OVERRIDES='n_exec_steps=6' bash policy/Diffusion-Policy/train_study_b_h25.sh
#
# Reuses the Study B SigLIP cache. Study A/B/DET/FG/FM checkpoints are left alone.
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

# ddpm = clean longer-chunk ablation vs Study B (default).
# flow = same *_ctxh25_* prior, flow-matching Stage 2 head only.
SAMPLER=${SAMPLER:-ddpm}

ZARR="data/zarr_data/${TASK}_multi_${DEMOS}.zarr"
LOG_DIR="data/outputs/study_b_h25"
mkdir -p "${LOG_DIR}"

case "${SAMPLER}" in
    ddpm) STAGE2_TAG=clsdph25 ;;
    flow) STAGE2_TAG=clsdpfmh25 ;;
    *)
        echo "SAMPLER must be ddpm or flow, got: ${SAMPLER}"
        exit 1
        ;;
esac

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

echo "=== Study B-H25: SAMPLER=${SAMPLER} -> *_${STAGE2_TAG}_* (Stage 2 config cls_dp_h25) ==="

# Stage 1 is shared across SAMPLER=ddpm and SAMPLER=flow. Skip any agent whose
# *_ctxh25_* final already exists so a second Stage 2 head does not retrain the prior.
export CONFIG_NAME=cls_stage1_h25
for agent in $(seq 0 $((AGENTS - 1))); do
    ckpt="checkpoints/${TASK}_ctxh25_Agent${agent}_${DEMOS}/100.ckpt"
    if [ -f "${ckpt}" ]; then
        echo "=== skip Study B-H25 Stage 1 agent ${agent} (${ckpt} exists) ==="
        continue
    fi
    echo "=== Study B-H25: Stage 1 agent ${agent} ==="
    bash policy/Diffusion-Policy/train_cls_stage1.sh \
        "${TASK}" "${DEMOS}" "${agent}" "${AGENTS}" "${SEED}" "${GPU}" \
        | tee "${LOG_DIR}/stage1_agent${agent}.log"
done

for agent in $(seq 0 $((AGENTS - 1))); do
    ckpt="checkpoints/${TASK}_ctxh25_Agent${agent}_${DEMOS}/100.ckpt"
    if [ ! -f "${ckpt}" ]; then
        echo "MISSING ${ckpt} — Stage 1 agent ${agent} did not finish"
        exit 1
    fi
done

echo
echo "=== Study B-H25: Stage 1 gate summary ==="
grep -h "Stage 1 gate" "${LOG_DIR}"/stage1_agent*.log || echo "(no gate lines found; Stage 1 was fully skipped)"
echo

# One Stage 2 yaml; SAMPLER flips DDPM <-> flow via the Hydra sampler group.
export CONFIG_NAME=cls_dp_h25
export CTX_TAG=ctxh25
export SAMPLER
# shellcheck disable=SC2206
extra_overrides=( ${EXTRA_OVERRIDES:-} )
for agent in $(seq 0 $((AGENTS - 1))); do
    ckpt="checkpoints/${TASK}_${STAGE2_TAG}_Agent${agent}_${DEMOS}/100.ckpt"
    if [ -f "${ckpt}" ]; then
        echo "=== skip Study B-H25 Stage 2 agent ${agent} (${ckpt} exists) ==="
        continue
    fi
    echo "=== Study B-H25: Stage 2 agent ${agent} (cls_dp_h25 sampler=${SAMPLER}) ==="
    bash policy/Diffusion-Policy/train_cls_dp.sh \
        "${TASK}" "${DEMOS}" "${agent}" "${AGENTS}" "${SEED}" "${GPU}" 100 \
        "${extra_overrides[@]}" \
        | tee "${LOG_DIR}/stage2_${SAMPLER}_agent${agent}.log"
    if [ ! -f "${ckpt}" ]; then
        echo "MISSING ${ckpt} — Stage 2 agent ${agent} did not finish"
        exit 1
    fi
done

echo "=== Study B-H25 training finished (SAMPLER=${SAMPLER}) ==="
ls -l checkpoints/${TASK}_ctxh25_Agent*_${DEMOS}/100.ckpt \
      checkpoints/${TASK}_${STAGE2_TAG}_Agent*_${DEMOS}/100.ckpt

cat <<EOF

Next, evaluate on the same 100 unseen seeds Study B used.
max_steps=65 matches Study B's env-step budget (250 policy cycles x 6 exec steps):

  bash policy/Diffusion-Policy/eval_cls_sweep.sh \\
      ${TASK} configs/table/lift_barrier.yaml ${DEMOS} 100 1000 1099 10 65 ${STAGE2_TAG}

Notes:
  - Default executed slice is 23 steps (25 - n_obs_steps + 1), not 25.
  - Receding horizon (predict 25, act 6): EXTRA_OVERRIDES='n_exec_steps=6' ...
  - Stage 1 at h25 is not interchangeable with Study B's *_ctx_* priors.
  - Re-running with SAMPLER=flow skips existing *_ctxh25_* Stage 1 and only trains the FM head.
  - Existing *_${STAGE2_TAG}_* finals are also skipped (safe resume).
  - Convenience alias (same composition): cls_dp_fm_h25.yaml
  - 30 Euler steps is baked into the flow checkpoint config; no trailing step argument needed.

max_steps is not a live variable on this task. Measured across every 100-seed run: successful
episodes finish in 4-5 policy cycles at h25 and 14-62 at h8 — about 100 environment steps
either way — and no success in any run has ever needed more than 65 cycles. Failures run to
whatever cap they are given. So 65 and 250 give identical success rates here, and the
historical B-H25 number (run at 250) is directly comparable to the combo numbers (run at 65).

Reference numbers, all seeds 1000-1099, paired:

  Study FM    h8,  flow @30      67%   \\ statistically identical to each other
  Study B-H25 h25, DDPM          66%   /  (McNemar z=0.00, 24 vs 23 disagreements)
  Study B     h8,  DDPM          61%
  Combo uniform (bfg+fm+h25)     49%
  Combo Beta   (bfg+fm+h25)      38%

Read SAMPLER=flow against B-H25's 66% with McNemar, not an independent two-proportion test —
every run shares the same seeds. Note that 100 seeds buys roughly +/-5pp of standard error on
this task, because success is decided by a single grasp in the first few cycles and there is
no recovery afterwards. Differences under ~10pp will not resolve at this sample size.
EOF
