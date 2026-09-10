#!/usr/bin/env bash
set -euo pipefail

PLUGIN_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_PYTHON=/root/.venvs/vllm-hust-latest/bin/python
if [[ ! -x "$DEFAULT_PYTHON" ]]; then
  DEFAULT_PYTHON=/root/miniconda3/envs/vllm-hust-dev/bin/python
fi
PYTHON_BIN=${PYTHON_BIN:-$DEFAULT_PYTHON}

PRESET=${1:-draft}
case "$PRESET" in
  draft)
    DEFAULT_CONFIG=$PLUGIN_DIR/configs/qwen25-14b-05b.toml
    shift
    ;;
  draft-adaptive)
    DEFAULT_CONFIG=$PLUGIN_DIR/configs/qwen25-14b-05b-adaptive-b128.toml
    shift
    ;;
  eagle)
    DEFAULT_CONFIG=$PLUGIN_DIR/configs/qwen25-14b-eagle.toml
    shift
    ;;
  eagle-relaxed)
    DEFAULT_CONFIG=$PLUGIN_DIR/configs/qwen25-14b-eagle-relaxed.toml
    shift
    ;;
  eagle3)
    DEFAULT_CONFIG=$PLUGIN_DIR/configs/qwen3-8b-eagle3.toml
    shift
    ;;
  *)
    DEFAULT_CONFIG=$PLUGIN_DIR/configs/qwen25-14b-05b.toml
    ;;
esac
CONFIG=${VSPEC_CONFIG:-${SPECSLO_CONFIG:-${SPECASCEND_CONFIG:-$DEFAULT_CONFIG}}}

exec "$PYTHON_BIN" -m vllm_hust_vspec --config "$CONFIG" "$@"
