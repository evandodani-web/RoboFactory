#!/usr/bin/env bash
# Study FG-FM-H25: factorized latent + flow-matching head + 25-step chunks.
#
# Three independent axes, one convenience composition:
#   Stage 1  cls_stage1_fg.yaml horizon=h25     -> *_ctxfgh25_*
#   Stage 2  cls_dp_fg.yaml sampler=flow horizon=h25  -> *_clsdpfgfmh25_*
#
# Pieces can be dropped on the CLI without a new file, e.g. keep DDPM with
# `CONFIG_NAME=cls_dp_fg` and omit sampler=flow, or receding-horizon execute-6 with
# `n_exec_steps=6`. Reuses the Study B SigLIP cache. Study A/B/DET/FG/FM checkpoints
# are left alone.
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

ZARR="data/zarr_data/${TASK}_multi_${DEMOS}.zarr"
LOG_DIR="data/outputs/study_fgfm_h25"
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

export CONFIG_NAME=cls_stage1_fg_h25
for agent in $(seq 0 $((AGENTS - 1))); do
    echo "=== Study FG-FM-H25: Stage 1 agent ${agent} ==="
    bash policy/Diffusion-Policy/train_cls_stage1_fg.sh \
        "${TASK}" "${DEMOS}" "${agent}" "${AGENTS}" "${SEED}" "${GPU}" \
        | tee "${LOG_DIR}/stage1_agent${agent}.log"
done

for agent in $(seq 0 $((AGENTS - 1))); do
    ckpt="checkpoints/${TASK}_ctxfgh25_Agent${agent}_${DEMOS}/100.ckpt"
    if [ ! -f "${ckpt}" ]; then
        echo "MISSING ${ckpt} — Stage 1 agent ${agent} did not finish"
        exit 1
    fi
done

echo
echo "=== Study FG-FM-H25: Stage 1 gates ==="
echo "Three lines per agent. Read them in order:"
echo "  teammate reconstruction  -- the usual gate, measured on prior + residual"
echo "  prior-only               -- what Stage 2 and deployment actually get"
echo "  leak probe               -- z_self should NOT predict teammates"
grep -h "Stage 1 gate" "${LOG_DIR}"/stage1_agent*.log || echo "(no gate lines found)"
echo

export CONFIG_NAME=cls_dp_fg_fm_h25
export CTX_TAG=ctxfgh25
for agent in $(seq 0 $((AGENTS - 1))); do
    echo "=== Study FG-FM-H25: Stage 2 agent ${agent} ==="
    bash policy/Diffusion-Policy/train_cls_dp_fg.sh \
        "${TASK}" "${DEMOS}" "${agent}" "${AGENTS}" "${SEED}" "${GPU}" \
        | tee "${LOG_DIR}/stage2_agent${agent}.log"
done

echo "=== Study FG-FM-H25 training finished ==="
ls -l checkpoints/${TASK}_ctxfgh25_Agent*_${DEMOS}/100.ckpt \
      checkpoints/${TASK}_clsdpfgfmh25_Agent*_${DEMOS}/100.ckpt

cat <<EOF

Next, evaluate on the same 100 unseen seeds Study B used:

  bash policy/Diffusion-Policy/eval_cls_sweep.sh \\
      ${TASK} configs/table/lift_barrier.yaml ${DEMOS} 100 1000 1099 10 65 clsdpfgfmh25

Notes:
  - Default executed slice is 23 steps (25 - n_obs_steps + 1), not 25.
  - max_steps=65 counts policy cycles and matches Study B's env-step budget (250 x 6).
  - Receding horizon (predict 25, act 6): train/eval with n_exec_steps=6.
  - FG Stage 1 at h25 is not interchangeable with Study B's *_ctx_* priors.

Drop any axis without a new file, e.g. DDPM instead of flow:
  CONFIG_NAME=cls_dp_fg CTX_TAG=ctxfgh25 bash policy/Diffusion-Policy/train_cls_dp_fg.sh \\
      ${TASK} ${DEMOS} 0 ${AGENTS} ${SEED} ${GPU} 100 horizon=h25
EOF
