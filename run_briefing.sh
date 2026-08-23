#!/bin/bash
#
# Atlas Morning Briefing runner wrapper.
#
# Portable: resolves its own location instead of a hardcoded path, so it works
# from any checkout (KiroCrew skill dir, ~/projects, CI, etc.).
#
# Environment overrides (all optional):
#   ATLAS_HOME      Project root (default: this script's directory)
#   ATLAS_ENV_FILE  Path to a dotenv file with API keys (default: $ATLAS_HOME/.env)
#   ATLAS_VENV      Path to a virtualenv to activate (default: $ATLAS_HOME/venv, if present)
#   ATLAS_CONFIG    Config YAML path (default: $ATLAS_HOME/config.yaml)
#
set -euo pipefail

# Resolve the directory this script lives in (portable, no hardcoded path).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATLAS_HOME="${ATLAS_HOME:-$SCRIPT_DIR}"
cd "$ATLAS_HOME"

# Activate a virtualenv if one is configured or present.
ATLAS_VENV="${ATLAS_VENV:-$ATLAS_HOME/venv}"
if [ -f "$ATLAS_VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$ATLAS_VENV/bin/activate"
fi

# Load API keys from a dotenv file if it exists (never fails if absent).
ATLAS_ENV_FILE="${ATLAS_ENV_FILE:-$ATLAS_HOME/.env}"
if [ -f "$ATLAS_ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ATLAS_ENV_FILE"
    set +a
fi

ATLAS_CONFIG="${ATLAS_CONFIG:-$ATLAS_HOME/config.yaml}"

# Clean previous run artifacts.
rm -f Atlas-Briefing-*.md Atlas-Briefing-*.pdf status.json

python3 scripts/briefing_runner.py --config "$ATLAS_CONFIG" "$@"
