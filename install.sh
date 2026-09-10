#!/usr/bin/env bash
set -euo pipefail

PLUGIN_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$PLUGIN_DIR/manage.sh" install --editable "$@"
