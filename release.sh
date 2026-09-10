#!/usr/bin/env bash
set -euo pipefail

PLUGIN_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_PYTHON=/root/.venvs/vllm-hust-latest/bin/python
if [[ ! -x "$DEFAULT_PYTHON" ]]; then
  DEFAULT_PYTHON=$(command -v python3 || true)
fi
PYTHON_BIN=${PYTHON_BIN:-$DEFAULT_PYTHON}
DIST_DIR=$PLUGIN_DIR/dist

die() {
  printf 'vSpec release: %s\n' "$*" >&2
  exit 2
}

version() {
  "$PYTHON_BIN" - "$PLUGIN_DIR/src/vllm_hust_vspec/_version.py" <<'PY'
import runpy
import sys

print(runpy.run_path(sys.argv[1])["__version__"])
PY
}

verify() {
  "$PYTHON_BIN" "$PLUGIN_DIR/scripts/verify_release.py" "$@"
}

build() {
  verify --source-only
  mkdir -p "$DIST_DIR"
  find "$DIST_DIR" -mindepth 1 -maxdepth 1 \
    \( -name '*.whl' -o -name '*.tar.gz' \) -type f -delete
  if command -v uv >/dev/null 2>&1; then
    uv build --no-sources --out-dir "$DIST_DIR" "$PLUGIN_DIR"
  else
    printf '%s\n' 'vSpec release: uv unavailable; falling back to python -m build' >&2
    "$PYTHON_BIN" -m build --outdir "$DIST_DIR" "$PLUGIN_DIR"
  fi
  verify
}

publish() {
  local release_version
  release_version=$(version)
  [[ -n "${UV_PUBLISH_TOKEN:-}" ]] || die "UV_PUBLISH_TOKEN is required"
  [[ "${VSPEC_RELEASE_CONFIRM:-}" == "$release_version" ]] || die \
    "set VSPEC_RELEASE_CONFIRM=$release_version to confirm this irreversible upload"
  command -v uv >/dev/null 2>&1 || die "uv is required for publishing"
  verify
  if git -C "$PLUGIN_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    [[ -z "$(git -C "$PLUGIN_DIR" status --short)" ]] || die \
      "refusing to publish from a dirty worktree"
    git -C "$PLUGIN_DIR" rev-parse HEAD
  else
    die "publishing requires a Git checkout so the release commit can be recorded"
  fi
  uv publish \
    --check-url https://pypi.org/simple \
    "$DIST_DIR/vllm_hust_vspec-$release_version-py3-none-any.whl" \
    "$DIST_DIR/vllm_hust_vspec-$release_version.tar.gz"
}

case "${1:-check}" in
  check)
    [[ $# -eq 0 || $# -eq 1 ]] || die "check does not accept arguments"
    verify
    ;;
  source-check)
    [[ $# -eq 1 ]] || die "source-check does not accept arguments"
    verify --source-only
    ;;
  build)
    [[ $# -eq 1 ]] || die "build does not accept arguments"
    build
    ;;
  publish)
    [[ $# -eq 1 ]] || die "publish does not accept arguments"
    publish
    ;;
  *)
    die "usage: ./release.sh [source-check|build|check|publish]"
    ;;
esac
