#!/usr/bin/env bash
set -euo pipefail

PLUGIN_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-}
if [[ ${1:-} == --python ]]; then
  [[ $# -ge 2 ]] || { echo '--python requires an interpreter' >&2; exit 2; }
  PYTHON_BIN=$2
  shift 2
fi
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN=$(command -v python3) || { echo 'python3 not found; pass --python' >&2; exit 2; }
fi
[[ -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; exit 2; }
export PYTHONPATH="$PLUGIN_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m vllm_hust_vspec.runtime_installer --plugin-source "$PLUGIN_DIR" "$@"
