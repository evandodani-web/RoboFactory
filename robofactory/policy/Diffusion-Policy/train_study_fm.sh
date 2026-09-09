#!/usr/bin/env bash
# Study FM: flow-matching Stage 2 action expert (raw action space).
#
# Modular contract (do not break this):
#   Stage 1  — REUSE Study B's frozen contextualizers (*_ctx_*). Never retrain them.
#   Stage 2  — train only the flow-matching action expert (*_clsdpfm_*).
#
# Retraining Stage 1 into *_ctx_* would silently replace the priors that Study B's 61%
# result and every FM run pin to. train_cls_stage1.sh also refuses that overwrite unless
# FORCE_OVERWRITE_CTX=1; this script never calls it.
#
# The evaluation harness supports re-evaluating the same checkpoint at multiple
# sampler step counts via the optional 10th argument.
#
set -euo pipefail

REPO_ROOT=/workspace/RoboFactory
cd "${REPO_ROOT}/robofactory"
source "${REPO_ROOT}/.venv/bin/activate"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib_ckpt_guard.sh
source "${SCRIPT_DIR}/lib_ckpt_guard.sh"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

TASK=LiftBarrier-rf
DEMOS=150
AGENTS=2
SEED=42
GPU=0

ZARR="data/zarr_data/${TASK}_multi_${DEMOS}.zarr"
LOG_DIR="data/outputs/study_fm"

mkdir -p "${LOG_DIR}"

if [ ! -d "${ZARR}" ]; then
    echo "missing ${ZARR}"
    exit 1
fi

# SigLIP cache preflight: flow-matching swaps the action expert only; frozen encoders
# must match Study B's 14x14 grid (197 tokens including pooled embedding).
python - <<PY
import zarr, sys
root = zarr.open("${ZARR}", mode="r")
n = int(root.attrs.get("siglip_n_image_tokens", -1))
print(f"SigLIP cache: {n} image tokens")
if n != 197:
    print("expected 197 (1 + 14x14); re-run precompute_siglip_features.py --pool_grid 14")
    sys.exit(1)
PY

echo
echo "=== Study FM: reusing Study B Stage 1 (*_ctx_*) — will not retrain ==="
for agent in $(seq 0 $((AGENTS - 1))); do
    ckpt="checkpoints/${TASK}_ctx_Agent${agent}_${DEMOS}/100.ckpt"
    require_ckpt "${ckpt}" \
        "Train Study B Stage 1 first (train_study_b.sh / train_cls_stage1.sh). Study FM must reuse those priors, not rewrite them."
done

echo "Fingerprints of reused Study B contextualizers:"
for agent in $(seq 0 $((AGENTS - 1))); do
    print_ckpt_fingerprint "checkpoints/${TASK}_ctx_Agent${agent}_${DEMOS}/100.ckpt"
done | tee "${LOG_DIR}/reused_stage1_fingerprints.txt"

for agent in $(seq 0 $((AGENTS - 1))); do
    echo "=== Study FM: Stage 2 agent ${agent} ==="
    bash policy/Diffusion-Policy/train_cls_dp_fm.sh \
        "${TASK}" "${DEMOS}" "${agent}" "${AGENTS}" "${SEED}" "${GPU}" \
        | tee "${LOG_DIR}/stage2_agent${agent}.log"
done

echo "=== Study FM training finished ==="
ls -l checkpoints/${TASK}_clsdpfm_Agent*_${DEMOS}/100.ckpt

cat <<EOF

Next, evaluate on the same 100 unseen seeds:

  bash policy/Diffusion-Policy/eval_cls_sweep.sh \\
      ${TASK} configs/table/lift_barrier.yaml ${DEMOS} 100 1000 1099 10 250 clsdpfm

To re-evaluate a trained checkpoint at (example) 8 sampler steps:

  bash policy/Diffusion-Policy/eval_cls_sweep.sh \\
      ${TASK} configs/table/lift_barrier.yaml ${DEMOS} 100 1000 1099 10 250 clsdpfm 8

EOF
