#!/usr/bin/env bash
# Bootstrap uv (if needed) and sync the RoboFactory env from pyproject.toml
# and uv.lock. Idempotent unless --force is passed.
#
#   bash setup_uv.sh
#   bash setup_uv.sh --force    # delete .venv and resync
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

if [[ "$(uname -s)" == "Linux" && "$(uname -m)" != "x86_64" ]]; then
  echo "warning: sapien/mani_skill publish manylinux x86_64 wheels only" >&2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing to ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

cd "${ROOT}"

if [[ "${FORCE}" -eq 1 && -d .venv ]]; then
  echo "removing existing .venv (--force)"
  rm -rf .venv
fi

echo "syncing from ${ROOT}/pyproject.toml (python $(cat "${ROOT}/.python-version"))"
uv sync --python 3.9

echo
echo "done. Activate and check:"
echo "  source ${ROOT}/.venv/bin/activate"
echo "  cd ${ROOT}/robofactory"
echo "  python policy/Diffusion-Policy/verify_cls_dp.py"
echo "  python policy/Diffusion-Policy/verify_cls_pipeline.py"
