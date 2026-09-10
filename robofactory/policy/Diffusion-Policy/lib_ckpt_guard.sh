#!/usr/bin/env bash
# Shared helpers for CLS-DP study scripts.
#
# Stage 1 checkpoint prefixes are study-owned:
#   Study B     *_ctx_*
#   Study B-H25 *_ctxh25_*
#   Study DET   *_ctxdet_*
#   Study FG    *_ctxfg_*  (DET + factorized; also *_ctxfgh25_* etc.)
#   Study B-FG  *_ctxbfg_* (Study B stochastic + factorized)
# with the horizon group appended when it is not h8, e.g. *_ctxbfgh25_*.
# Study FM does not own a Stage 1 prefix — it must reuse Study B's *_ctx_*.
#
# Stage 2 finals are guarded on the same terms. They all land in the shared repo-level
# checkpoints/${checkpoint_name}/ tree rather than the timestamped Hydra run dir, so two
# studies that compose to the same tag overwrite each other's results in place. (Resume is
# unaffected: it reads the Hydra run dir, so this never blocks resuming an interrupted run.)
#
# Intentional regeneration: FORCE_OVERWRITE_CKPT=1 bash .../train_*.sh ...

refuse_overwrite_ckpt() {
    local path=$1
    local owner=${2:-this study}

    # FORCE_OVERWRITE_CTX is the older Stage-1-only spelling, still honoured.
    if [ -f "${path}" ] \
        && [ "${FORCE_OVERWRITE_CKPT:-0}" != "1" ] \
        && [ "${FORCE_OVERWRITE_CTX:-0}" != "1" ]; then
        echo -e "\033[31mrefusing to overwrite existing ${path}\033[0m"
        echo "That checkpoint is owned by ${owner}. Retraining here would silently"
        echo "replace a result that other runs and the docs already cite."
        echo "Give this run its own tag (run_tag=... for Stage 2), or if you really mean"
        echo "to regenerate: FORCE_OVERWRITE_CKPT=1 <same command>"
        exit 1
    fi
}

# Ask Hydra what a config + overrides actually composes its checkpoint_name to.
#
# The alternative — a case statement mapping CONFIG_NAME to a tag — cannot see group
# overrides, so `horizon=h25` used to guard *_ctxbfg_* while writing *_ctxbfgh25_*: the
# wrong path protected, and a spurious refusal on the right one. Deriving the name from the
# same yaml the trainer reads makes that class of drift impossible.
resolve_checkpoint_name() {
    local config_dir=$1
    local config_name=$2
    shift 2

    python - "${config_dir}" "${config_name}" "$@" <<'PY'
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)
config_dir, config_name, *overrides = sys.argv[1:]
# Hydra rejects a trailing .yaml here; study scripts pass it either way.
config_name = config_name[:-5] if config_name.endswith(".yaml") else config_name
with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(config_name=config_name, overrides=list(overrides))
print(cfg.checkpoint_name)
PY
}

require_ckpt() {
    local path=$1
    local hint=${2:-}

    if [ ! -f "${path}" ]; then
        echo -e "\033[31mmissing required checkpoint ${path}\033[0m"
        if [ -n "${hint}" ]; then
            echo "${hint}"
        fi
        exit 1
    fi
}

print_ckpt_fingerprint() {
    local path=$1
    if [ ! -f "${path}" ]; then
        echo "  (missing) ${path}"
        return 0
    fi
    # sha256sum is enough to detect accidental swaps; cheap and always available.
    local digest
    digest=$(sha256sum "${path}" | awk '{print $1}')
    local bytes
    bytes=$(stat -c '%s' "${path}")
    echo "  ${digest}  ${bytes}B  ${path}"
}
