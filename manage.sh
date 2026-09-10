#!/usr/bin/env bash
set -euo pipefail

PLUGIN_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXTENSION_ID=org.vllm-hust.vspec
PACKAGE_NAME=vllm-hust-vspec

DEFAULT_PYTHON=/root/.venvs/vllm-hust-latest/bin/python
if [[ ! -x "$DEFAULT_PYTHON" ]]; then
  DEFAULT_PYTHON=/root/miniconda3/envs/vllm-hust-dev/bin/python
fi
if [[ ! -x "$DEFAULT_PYTHON" ]]; then
  DEFAULT_PYTHON=$(command -v python3 || true)
fi
PYTHON_BIN=${PYTHON_BIN:-$DEFAULT_PYTHON}
DRY_RUN=${VSPEC_MANAGE_DRY_RUN:-0}

usage() {
  cat <<'EOF'
Usage:
  ./manage.sh [--python PATH] [--dry-run] install [--editable] [--wheel PATH] [--enable]
      [--model-dir PATH] [--model-registry PATH] [--no-model-download|--skip-model-setup]
  ./manage.sh [--python PATH] [--dry-run] upgrade --version VERSION [--enable]
  ./manage.sh [--python PATH] [--dry-run] upgrade --wheel PATH [--enable]
  ./manage.sh [--python PATH] [--dry-run] rollback --version VERSION [--enable]
  ./manage.sh [--python PATH] [--dry-run] rollback --wheel PATH [--enable]
  ./manage.sh [--python PATH] [--dry-run] uninstall
  ./manage.sh [--python PATH] [--dry-run] list [--json]
  ./manage.sh [--python PATH] [--dry-run] models [--model-dir PATH]
      [--model-registry PATH] [--no-download]
  ./manage.sh [--python PATH] [--dry-run] validate|inspect|check|status|plan|render
  ./manage.sh [--python PATH] [--dry-run] enable|disable|forget
  ./manage.sh [--python PATH] [--dry-run] run [--dry-run] -- COMMAND [ARG ...]

Commands:
  install     Install the current release wheel, or source with --editable.
              Detect and download the default Draft/EAGLE models unless skipped.
  upgrade     Upgrade to an exact package version or local wheel.
  rollback    Force-reinstall an exact package version or local wheel.
  uninstall   Disable and forget Manager intent, then uninstall only vSpec.
  list        List installed Extension Bundles.
  models      Detect/download default models and write the runtime registry.
  validate    Validate the static vSpec manifest.
  check       Check host and protocol compatibility.
  plan        Preview the lifecycle and launch actions.
  render      Render provider-owned launch artifacts.
  run         Launch a command through Manager; inner --dry-run only renders.
  enable      Enable vSpec for future Manager-owned launches.
  disable     Disable vSpec without uninstalling it.
  forget      Remove saved Manager configuration and enablement intent.
  inspect     Show the discovered static extension manifest.
  status      Show installation, compatibility, and enablement state.

Environment overrides:
  PYTHON_BIN              Python interpreter whose environment is managed.
  VSPEC_MANAGER_BIN       vllm-hust-ext executable to use.
  VSPEC_MANAGE_DRY_RUN=1  Print commands without changing the environment.
  HUST_VSPEC_MODEL_DIR    Default directory for downloaded models.
  HUST_VSPEC_MODEL_REGISTRY  Override the model registry JSON path.
EOF
}

die() {
  printf 'vSpec manager: %s\n' "$*" >&2
  exit 2
}

print_command() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

run() {
  print_command "$@"
  if [[ "$DRY_RUN" != 1 ]]; then
    "$@"
  fi
}

run_best_effort() {
  print_command "$@"
  if [[ "$DRY_RUN" == 1 ]]; then
    return
  fi
  if ! "$@"; then
    printf 'vSpec manager: warning: command failed; continuing uninstall\n' >&2
  fi
}

package_is_visible() {
  "$PYTHON_BIN" - "$PACKAGE_NAME" <<'PY'
import importlib.metadata
import sys

try:
    importlib.metadata.version(sys.argv[1])
except importlib.metadata.PackageNotFoundError:
    raise SystemExit(1)
PY
}

plugin_version() {
  "$PYTHON_BIN" - "$PLUGIN_DIR/src/vllm_hust_vspec/_version.py" <<'PY'
import runpy
import sys

print(runpy.run_path(sys.argv[1])["__version__"])
PY
}

manager_path() {
  if [[ -n "${VSPEC_MANAGER_BIN:-}" ]]; then
    printf '%s\n' "$VSPEC_MANAGER_BIN"
    return
  fi
  local adjacent
  adjacent=$(cd "$(dirname "$PYTHON_BIN")" && pwd)/vllm-hust-ext
  if [[ -x "$adjacent" ]]; then
    printf '%s\n' "$adjacent"
    return
  fi
  command -v vllm-hust-ext || true
}

require_manager() {
  local manager
  manager=$(manager_path)
  [[ -n "$manager" && -x "$manager" ]] || die \
    "vllm-hust-ext was not found next to $PYTHON_BIN or on PATH"
  printf '%s\n' "$manager"
}

inspect_if_available() {
  local manager
  manager=$(manager_path)
  if [[ -n "$manager" && -x "$manager" ]]; then
    run "$manager" extension inspect "$EXTENSION_ID"
  else
    printf '%s\n' \
      'vSpec manager: warning: installed, but vllm-hust-ext is unavailable for inspection' >&2
  fi
}

check_with_manager() {
  local manager=$1
  run "$manager" extension validate "$EXTENSION_ID"
  run "$manager" extension check "$EXTENSION_ID"
}

setup_models() {
  local model_dir=$1
  local registry=$2
  local download=$3
  local -a arguments=()
  if [[ -n "$model_dir" ]]; then
    arguments+=(--model-dir "$model_dir")
  fi
  if [[ -n "$registry" ]]; then
    arguments+=(--registry "$registry")
  fi
  if [[ "$download" != 1 ]]; then
    arguments+=(--no-download)
  fi
  run "$PYTHON_BIN" -m vllm_hust_vspec.model_store "${arguments[@]}"
}

install_release() {
  local operation=$1
  local version=$2
  local wheel_path=$3
  local -a pip_arguments=(install --no-cache-dir)
  if [[ "$operation" == upgrade ]]; then
    pip_arguments+=(--upgrade)
  else
    pip_arguments+=(--force-reinstall)
  fi
  if [[ -n "$wheel_path" ]]; then
    if [[ "$wheel_path" != /* ]]; then
      wheel_path="$PWD/$wheel_path"
    fi
    [[ -f "$wheel_path" ]] || die "release wheel not found: $wheel_path"
    pip_arguments+=(--no-deps "$wheel_path")
  else
    pip_arguments+=("$PACKAGE_NAME==$version")
  fi
  run "$PYTHON_BIN" -m pip "${pip_arguments[@]}"
  inspect_if_available
  local manager
  manager=$(require_manager)
  check_with_manager "$manager"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --python)
      [[ $# -ge 2 ]] || die "--python requires a path"
      PYTHON_BIN=$2
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || die "Python is not executable: $PYTHON_BIN"

COMMAND=${1:-help}
if [[ $# -gt 0 ]]; then
  shift
fi

case "$COMMAND" in
  install)
    EDITABLE=0
    ENABLE=0
    WHEEL_PATH=
    MODEL_DIR=
    MODEL_REGISTRY=
    MODEL_DOWNLOAD=1
    MODEL_SETUP=1
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --editable)
          EDITABLE=1
          shift
          ;;
        --enable)
          ENABLE=1
          shift
          ;;
        --wheel)
          [[ $# -ge 2 ]] || die "--wheel requires a path"
          WHEEL_PATH=$2
          shift 2
          ;;
        --model-dir)
          [[ $# -ge 2 ]] || die "--model-dir requires a path"
          MODEL_DIR=$2
          shift 2
          ;;
        --model-registry)
          [[ $# -ge 2 ]] || die "--model-registry requires a path"
          MODEL_REGISTRY=$2
          shift 2
          ;;
        --no-model-download)
          MODEL_DOWNLOAD=0
          shift
          ;;
        --skip-model-setup)
          MODEL_SETUP=0
          shift
          ;;
        *)
          die "unknown install option: $1"
          ;;
      esac
    done
    if [[ "$EDITABLE" == 1 && -n "$WHEEL_PATH" ]]; then
      die "--editable and --wheel are mutually exclusive"
    fi
    if [[ "$EDITABLE" == 1 ]]; then
      run "$PYTHON_BIN" -m pip install --no-deps --editable "$PLUGIN_DIR"
    else
      if [[ -z "$WHEEL_PATH" ]]; then
        PLUGIN_VERSION=$(plugin_version)
        WHEEL_PATH="$PLUGIN_DIR/dist/vllm_hust_vspec-${PLUGIN_VERSION}-py3-none-any.whl"
      elif [[ "$WHEEL_PATH" != /* ]]; then
        WHEEL_PATH="$PWD/$WHEEL_PATH"
      fi
      [[ -f "$WHEEL_PATH" ]] || die \
        "release wheel not found: $WHEEL_PATH (build it or use --editable)"
      run "$PYTHON_BIN" -m pip install --force-reinstall --no-deps "$WHEEL_PATH"
    fi
    if [[ "$MODEL_SETUP" == 1 ]]; then
      setup_models "$MODEL_DIR" "$MODEL_REGISTRY" "$MODEL_DOWNLOAD"
    fi
    inspect_if_available
    if [[ "$ENABLE" == 1 ]]; then
      MANAGER_BIN=$(require_manager)
      check_with_manager "$MANAGER_BIN"
      run "$MANAGER_BIN" extension enable "$EXTENSION_ID"
    fi
    ;;
  upgrade|rollback)
    VERSION=
    WHEEL_PATH=
    ENABLE=0
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --version)
          [[ $# -ge 2 ]] || die "--version requires a value"
          VERSION=$2
          shift 2
          ;;
        --wheel)
          [[ $# -ge 2 ]] || die "--wheel requires a path"
          WHEEL_PATH=$2
          shift 2
          ;;
        --enable)
          ENABLE=1
          shift
          ;;
        *)
          die "unknown $COMMAND option: $1"
          ;;
      esac
    done
    if [[ -n "$VERSION" && -n "$WHEEL_PATH" ]]; then
      die "--version and --wheel are mutually exclusive"
    fi
    if [[ -z "$VERSION" && -z "$WHEEL_PATH" ]]; then
      die "$COMMAND requires --version VERSION or --wheel PATH"
    fi
    install_release "$COMMAND" "$VERSION" "$WHEEL_PATH"
    if [[ "$ENABLE" == 1 ]]; then
      MANAGER_BIN=$(require_manager)
      run "$MANAGER_BIN" extension enable "$EXTENSION_ID"
    fi
    printf '%s\n' \
      'vSpec manager: restart vLLM processes to load the installed plugin version'
    ;;
  uninstall)
    [[ $# -eq 0 ]] || die "uninstall does not accept additional arguments"
    MANAGER_BIN=$(manager_path)
    if [[ -n "$MANAGER_BIN" && -x "$MANAGER_BIN" ]]; then
      run_best_effort "$MANAGER_BIN" extension disable "$EXTENSION_ID"
      run_best_effort "$MANAGER_BIN" extension forget "$EXTENSION_ID"
    fi
    run "$PYTHON_BIN" -m pip uninstall -y "$PACKAGE_NAME"
    if [[ "$DRY_RUN" != 1 ]] && package_is_visible; then
      printf '%s\n' \
        'vSpec manager: vSpec is still inherited from another site-packages directory.' \
        'Use a clean virtual environment or uninstall it from the parent environment explicitly.' >&2
      exit 1
    fi
    printf '%s\n' \
      'vSpec manager: existing vLLM processes are unchanged; restart them to unload the plugin'
    ;;
  list)
    if [[ $# -gt 1 || ($# -eq 1 && "$1" != --json) ]]; then
      die "list only accepts --json"
    fi
    MANAGER_BIN=$(require_manager)
    run "$MANAGER_BIN" extension list "$@"
    ;;
  models)
    MODEL_DIR=
    MODEL_REGISTRY=
    MODEL_DOWNLOAD=1
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --model-dir)
          [[ $# -ge 2 ]] || die "--model-dir requires a path"
          MODEL_DIR=$2
          shift 2
          ;;
        --model-registry)
          [[ $# -ge 2 ]] || die "--model-registry requires a path"
          MODEL_REGISTRY=$2
          shift 2
          ;;
        --no-download)
          MODEL_DOWNLOAD=0
          shift
          ;;
        *)
          die "unknown models option: $1"
          ;;
      esac
    done
    setup_models "$MODEL_DIR" "$MODEL_REGISTRY" "$MODEL_DOWNLOAD"
    ;;
  validate|inspect|check|status|plan|render|enable|disable|forget)
    [[ $# -eq 0 ]] || die "$COMMAND does not accept additional arguments"
    MANAGER_BIN=$(require_manager)
    run "$MANAGER_BIN" extension "$COMMAND" "$EXTENSION_ID"
    ;;
  run|launch)
    MANAGER_DRY_RUN=0
    if [[ ${1:-} == --dry-run ]]; then
      MANAGER_DRY_RUN=1
      shift
    fi
    [[ ${1:-} == -- ]] || die "$COMMAND requires -- before the command"
    shift
    [[ $# -gt 0 ]] || die "$COMMAND requires a command to launch"
    MANAGER_BIN=$(require_manager)
    if [[ "$MANAGER_DRY_RUN" == 1 ]]; then
      run "$MANAGER_BIN" run --dry-run -- "$@"
    else
      run "$MANAGER_BIN" run -- "$@"
    fi
    ;;
  help)
    usage
    ;;
  *)
    die "unknown command: $COMMAND"
    ;;
esac
