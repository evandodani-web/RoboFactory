#!/usr/bin/env bash
# Study B-FG: Study B recipe + factorized latent (stochastic prior).
#
# Clean answer to "does factorization help?" — same Adam, 14x14 SigLIP, 150 LiftBarrier
# demos, and stochastic CVAE as Study B; only the latent is split into z_self / z_team.
#
# This is NOT Study FG (`train_study_fg.sh`), which stacks the split on Study DET's
# deterministic latent. Existing *_ctxfg_* / *_clsdpfg_* checkpoints are left alone.
#
#   Stage 1  cls_stage1_bfg.yaml  -> *_ctxbfg_*
#   Stage 2  cls_dp_bfg.yaml      -> *_clsdpbfg_*
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

ZARR="data/zarr_data/${TASK}_multi_${DEMOS}.zarr"
LOG_DIR="data/outputs/study_bfg"
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

echo "=== Study B-FG: stochastic Study B base + z_self/z_team split (*_ctxbfg_* / *_clsdpbfg_*) ==="

export CONFIG_NAME=cls_stage1_bfg
for agent in $(seq 0 $((AGENTS - 1))); do
    echo "=== Study B-FG: Stage 1 agent ${agent} ==="
    bash policy/Diffusion-Policy/train_cls_stage1.sh \
        "${TASK}" "${DEMOS}" "${agent}" "${AGENTS}" "${SEED}" "${GPU}" \
        | tee "${LOG_DIR}/stage1_agent${agent}.log"
done

for agent in $(seq 0 $((AGENTS - 1))); do
    ckpt="checkpoints/${TASK}_ctxbfg_Agent${agent}_${DEMOS}/100.ckpt"
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

export CONFIG_NAME=cls_dp_bfg
export CTX_TAG=ctxbfg
for agent in $(seq 0 $((AGENTS - 1))); do
    echo "=== Study B-FG: Stage 2 agent ${agent} ==="
    bash policy/Diffusion-Policy/train_cls_dp.sh \
        "${TASK}" "${DEMOS}" "${agent}" "${AGENTS}" "${SEED}" "${GPU}" \
        | tee "${LOG_DIR}/stage2_agent${agent}.log"
done

echo "=== Study B-FG training finished ==="
ls -l checkpoints/${TASK}_ctxbfg_Agent*_${DEMOS}/100.ckpt \
      checkpoints/${TASK}_clsdpbfg_Agent*_${DEMOS}/100.ckpt

cat <<EOF

Next, evaluate on the same 100 unseen seeds Study B used:

  bash policy/Diffusion-Policy/eval_cls_sweep.sh \\
      ${TASK} configs/table/lift_barrier.yaml ${DEMOS} 100 1000 1099 10 250 clsdpbfg

Baselines:
  Study B   (stochastic, monolithic)   61%
  Study DET (deterministic, monolithic) 49%
  Study FG  (deterministic, factorized) 55%   <-- confounded with DET
  Study B-FG (stochastic, factorized)   this run

If the leak probe still says NOT SEPARATED, treat success-rate movement cautiously —
the split may still be cosmetic.
EOF
