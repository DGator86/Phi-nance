#!/bin/bash
set -euo pipefail

# Only run in remote (Claude Code on the web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

echo "Installing Phi-nance dependencies..."

# Ensure a modern pip is available in user space (avoids system pip limitations)
pip3 install --user --upgrade pip setuptools wheel --quiet --root-user-action=ignore

# Install project requirements using the upgraded pip
"${HOME}/.local/bin/pip" install \
  -r "${CLAUDE_PROJECT_DIR}/requirements.txt" \
  --ignore-installed \
  --quiet \
  --root-user-action=ignore

echo "Dependencies installed successfully."
