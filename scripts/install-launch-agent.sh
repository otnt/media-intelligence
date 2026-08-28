#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_SRC="$ROOT/scripts/com.media-pipeline.server.plist.template"
DEST="$HOME/Library/LaunchAgents/com.media-pipeline.server.plist"
PYTHON="$ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Run scripts/install.sh first so .venv exists." >&2
  exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents" "$HOME/AIContent/logs"
sed \
  -e "s|__VENV_PYTHON__|$PYTHON|g" \
  -e "s|__REPO_ROOT__|$ROOT|g" \
  -e "s|__HOME__|$HOME|g" \
  "$PLIST_SRC" > "$DEST"

launchctl unload "$DEST" 2>/dev/null || true
launchctl load "$DEST"
echo "Loaded $DEST"
echo "Dashboard: http://127.0.0.1:8875/ (LAN; Tailscale: ./scripts/install-tailscale-serve.sh)"
