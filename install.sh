#!/usr/bin/env bash
set -euo pipefail

PLUGIN_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_PYTHON=/root/.venvs/vllm-hust-latest/bin/python
if [[ ! -x "$DEFAULT_PYTHON" ]]; then
  DEFAULT_PYTHON=$(command -v python3)
fi
PYTHON_BIN=${PYTHON_BIN:-$DEFAULT_PYTHON}
if [[ ${1:-} == --python ]]; then
  [[ $# -ge 2 ]] || { echo '--python requires an interpreter' >&2; exit 2; }
  PYTHON_BIN=$2
  shift 2
fi
[[ -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; exit 2; }
export PYTHONPATH="$PLUGIN_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m vllm_hust_vspec.runtime_installer --plugin-source "$PLUGIN_DIR" "$@"
