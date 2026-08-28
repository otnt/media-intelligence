#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_SRC="$ROOT/scripts/com.media-pipeline.tailscale-serve.plist.template"
DEST="$HOME/Library/LaunchAgents/com.media-pipeline.tailscale-serve.plist"
PYTHON="$ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Run scripts/install.sh first so .venv exists." >&2
  exit 1
fi

if ! command -v tailscale >/dev/null 2>&1 && \
   [[ ! -x /Applications/Tailscale.app/Contents/MacOS/Tailscale ]]; then
  echo "Tailscale is not installed. Install: brew install --cask tailscale-app" >&2
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
echo "Serve config: $ROOT/scripts/tailscale-serve.json"
echo "Apply now:    $ROOT/scripts/apply-tailscale-serve.sh"
echo ""
echo "Tailscale app must stay signed in (login item). This LaunchAgent re-applies Serve on login."
echo "Example URL:  http://<your-machine>.ts.net:8875/"
