#!/usr/bin/env bash
# Study FG: the factorized-latent variant of CLS-DP.
#
# Matches Study DET exactly (which in turn matches Study B: LiftBarrier, 150 demos, Adam,
# full 14x14 SigLIP) except that the prior latent is split into a self half and a teammate
# half with separate decoders, and only the teammate half receives the privileged residual.
#
# Two measurement probes ride along, both stop-gradiented so they change nothing:
#   prior-only  -- is the prior sufficient on its own? Deployment has no residual.
#   leak        -- did the split hold, or did z_self absorb teammate information?
#
# Reuses the Study B SigLIP cache; the frozen encoders are untouched.
# Study A/B/DET checkpoints are left alone.
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
LOG_DIR="data/outputs/study_fg"
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

for agent in $(seq 0 $((AGENTS - 1))); do
    echo "=== Study FG: Stage 1 agent ${agent} ==="
    bash policy/Diffusion-Policy/train_cls_stage1_fg.sh \
        "${TASK}" "${DEMOS}" "${agent}" "${AGENTS}" "${SEED}" "${GPU}" \
        | tee "${LOG_DIR}/stage1_agent${agent}.log"
done

for agent in $(seq 0 $((AGENTS - 1))); do
    ckpt="checkpoints/${TASK}_ctxfg_Agent${agent}_${DEMOS}/100.ckpt"
    if [ ! -f "${ckpt}" ]; then
        echo "MISSING ${ckpt} — Stage 1 agent ${agent} did not finish"
        exit 1
    fi
done

echo
echo "=== Study FG: Stage 1 gates ==="
echo "Three lines per agent. Read them in order:"
echo "  teammate reconstruction  -- the usual gate, measured on prior + residual"
echo "  prior-only               -- what Stage 2 and deployment actually get"
echo "  leak probe               -- z_self should NOT predict teammates"
grep -h "Stage 1 gate" "${LOG_DIR}"/stage1_agent*.log || echo "(no gate lines found)"
echo

for agent in $(seq 0 $((AGENTS - 1))); do
    echo "=== Study FG: Stage 2 agent ${agent} ==="
    bash policy/Diffusion-Policy/train_cls_dp_fg.sh \
        "${TASK}" "${DEMOS}" "${agent}" "${AGENTS}" "${SEED}" "${GPU}" \
        | tee "${LOG_DIR}/stage2_agent${agent}.log"
done

echo "=== Study FG training finished ==="
ls -l checkpoints/${TASK}_ctxfg_Agent*_${DEMOS}/100.ckpt \
      checkpoints/${TASK}_clsdpfg_Agent*_${DEMOS}/100.ckpt

cat <<EOF

Next, evaluate on the same 100 unseen seeds Study B used:

  bash policy/Diffusion-Policy/eval_cls_sweep.sh \\
      ${TASK} configs/table/lift_barrier.yaml ${DEMOS} 100 1000 1099 10 250 clsdpfg

Baselines: Study B (stochastic, monolithic) 61.0%. Study DET is the direct parent.
EOF
