#!/usr/bin/env bash
# Study DET: the deterministic-latent variant of CLS-DP.
#
# Matches Study B exactly (LiftBarrier, 150 demos, Adam, full 14x14 SigLIP) except that
# the contextualizer emits z directly instead of a Gaussian, nothing is sampled, and the
# KL is replaced by its unit-variance limit 0.5*||z_E||^2. Study B is the run that
# reproduced the paper at 61.0%, so any delta here is attributable to the stochasticity.
#
# Reuses the SigLIP cache from Study B -- the encoders are untouched, so no recache.
# Study A/B checkpoints (*_ctx_*, *_clsdp_*) are left alone.
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
LOG_DIR="data/outputs/study_det"
mkdir -p "${LOG_DIR}"

if [ ! -d "${ZARR}" ]; then
    echo "missing ${ZARR}"
    exit 1
fi
# The deterministic variant changes nothing about the frozen encoders, so the Study B
# 14x14 cache is reused as-is. Fail loudly rather than silently training on a 4x4 cache.
python - <<PY
import zarr, sys
root = zarr.open("${ZARR}", mode="r")
n = int(root.attrs.get("siglip_n_image_tokens", -1))
print(f"SigLIP cache: {n} image tokens")
if n != 197:
    print(f"expected 197 (1 + 14x14); re-run precompute_siglip_features.py --pool_grid 14")
    sys.exit(1)
PY

for agent in $(seq 0 $((AGENTS - 1))); do
    echo "=== Study DET: Stage 1 agent ${agent} ==="
    bash policy/Diffusion-Policy/train_cls_stage1_det.sh \
        "${TASK}" "${DEMOS}" "${agent}" "${AGENTS}" "${SEED}" "${GPU}" \
        | tee "${LOG_DIR}/stage1_agent${agent}.log"
done

for agent in $(seq 0 $((AGENTS - 1))); do
    ckpt="checkpoints/${TASK}_ctxdet_Agent${agent}_${DEMOS}/100.ckpt"
    if [ ! -f "${ckpt}" ]; then
        echo "MISSING ${ckpt} — Stage 1 agent ${agent} did not finish"
        exit 1
    fi
done

echo "=== Study DET: Stage 1 gate summary ==="
grep -h "Stage 1 gate" "${LOG_DIR}"/stage1_agent*.log || echo "(no gate line found)"

for agent in $(seq 0 $((AGENTS - 1))); do
    echo "=== Study DET: Stage 2 agent ${agent} ==="
    bash policy/Diffusion-Policy/train_cls_dp_det.sh \
        "${TASK}" "${DEMOS}" "${agent}" "${AGENTS}" "${SEED}" "${GPU}" \
        | tee "${LOG_DIR}/stage2_agent${agent}.log"
done

echo "=== Study DET training finished ==="
ls -l checkpoints/${TASK}_ctxdet_Agent*_${DEMOS}/100.ckpt \
      checkpoints/${TASK}_clsdpdet_Agent*_${DEMOS}/100.ckpt

cat <<EOF

Next, evaluate on the same 100 unseen seeds Study B used:

  bash policy/Diffusion-Policy/eval_cls_sweep.sh \\
      ${TASK} configs/table/lift_barrier.yaml ${DEMOS} 100 1000 1099 10 250 clsdpdet

Baseline to beat (Study B, stochastic latent): 61.0%
EOF
