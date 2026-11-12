#!/usr/bin/env bash
# Run a bulk download job in the background: ./scripts/install-bulk-agent.sh aie
# Remove it again with:                       ./scripts/install-bulk-agent.sh aie --remove
set -euo pipefail

JOB="${1:?usage: $0 JOB_NAME [--remove]}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_SRC="$ROOT/scripts/com.media-pipeline.bulk.plist.template"
LABEL="com.media-pipeline.bulk.$JOB"
DEST="$HOME/Library/LaunchAgents/$LABEL.plist"
PYTHON="$ROOT/.venv/bin/python"
uid="$(id -u)"

if [[ ! "$JOB" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Job names are letters, digits, - and _ only." >&2
  exit 1
fi

if [[ "${2:-}" == "--remove" ]]; then
  launchctl bootout "gui/$uid/$LABEL" 2>/dev/null || launchctl unload "$DEST" 2>/dev/null || true
  [[ -f "$DEST" ]] && mv "$DEST" "$HOME/.Trash/$LABEL.plist.$(date +%s)"
  echo "Removed $LABEL"
  exit 0
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Run scripts/install.sh first so .venv exists." >&2
  exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents" "$HOME/AIContent/logs"
sed \
  -e "s|__VENV_PYTHON__|$PYTHON|g" \
  -e "s|__REPO_ROOT__|$ROOT|g" \
  -e "s|__HOME__|$HOME|g" \
  -e "s|__JOB__|$JOB|g" \
  "$PLIST_SRC" > "$DEST"

launchctl bootout "gui/$uid/$LABEL" 2>/dev/null || launchctl unload "$DEST" 2>/dev/null || true
launchctl bootstrap "gui/$uid" "$DEST" 2>/dev/null || launchctl load "$DEST"
launchctl enable "gui/$uid/$LABEL" 2>/dev/null || true

echo "Loaded $LABEL ($DEST)"
echo "Progress: http://127.0.0.1:8875/bulk  ·  log: ~/AIContent/logs/bulk-$JOB.log"
