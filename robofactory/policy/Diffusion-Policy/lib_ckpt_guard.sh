#!/usr/bin/env bash
# Shared helpers for CLS-DP study scripts.
#
# Stage 1 checkpoint prefixes are study-owned:
#   Study B   *_ctx_*
#   Study DET *_ctxdet_*
#   Study FG  *_ctxfg_*  (and *_ctxfgh25_* etc.)
# Study FM does not own a Stage 1 prefix — it must reuse Study B's *_ctx_*.
#
# Refuse silent overwrites so a later study cannot clobber an earlier one's priors.
# Intentional regeneration: FORCE_OVERWRITE_CTX=1 bash .../train_cls_stage1.sh ...

refuse_overwrite_ckpt() {
    local path=$1
    local owner=${2:-this study}

    if [ -f "${path}" ] && [ "${FORCE_OVERWRITE_CTX:-0}" != "1" ]; then
        echo -e "\033[31mrefusing to overwrite existing ${path}\033[0m"
        echo "That checkpoint is owned by ${owner}. Retraining here would silently"
        echo "change every Stage 2 run that pins this path (including Study FM)."
        echo "To regenerate on purpose: FORCE_OVERWRITE_CTX=1 <same command>"
        exit 1
    fi
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
