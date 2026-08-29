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

# Prefer bootout/bootstrap on modern macOS; fall back to unload/load.
uid="$(id -u)"
launchctl bootout "gui/$uid" "$DEST" 2>/dev/null || launchctl unload "$DEST" 2>/dev/null || true
launchctl bootstrap "gui/$uid" "$DEST" 2>/dev/null || launchctl load "$DEST"
launchctl enable "gui/$uid/com.media-pipeline.server" 2>/dev/null || true
# kickstart is async-friendly; do not wait forever if the job is already restarting
launchctl kickstart -k "gui/$uid/com.media-pipeline.server" 2>/dev/null || true

echo "Loaded $DEST"
echo "Dashboard: http://127.0.0.1:8875/ (localhost; remote via Tailscale: ./scripts/tailscale/install.sh)"
echo "KeepAlive + ThrottleInterval=30; binds 127.0.0.1 so it does not fight Tailscale on :8875."
