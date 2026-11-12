#!/usr/bin/env bash
# Self-check every 5 minutes (API, bulk jobs, disk): ./scripts/install-watchdog.sh  (--remove to stop)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.media-pipeline.watchdog"
DEST="$HOME/Library/LaunchAgents/$LABEL.plist"
PYTHON="$ROOT/.venv/bin/python"
uid="$(id -u)"

if [[ "${1:-}" == "--remove" ]]; then
  launchctl bootout "gui/$uid/$LABEL" 2>/dev/null || true
  [[ -f "$DEST" ]] && mv "$DEST" "$HOME/.Trash/$LABEL.plist.$(date +%s)"
  echo "Removed $LABEL"
  exit 0
fi

mkdir -p "$HOME/Library/LaunchAgents" "$HOME/AIContent/logs"
sed \
  -e "s|__VENV_PYTHON__|$PYTHON|g" \
  -e "s|__REPO_ROOT__|$ROOT|g" \
  -e "s|__HOME__|$HOME|g" \
  "$ROOT/scripts/com.media-pipeline.watchdog.plist.template" > "$DEST"
launchctl bootout "gui/$uid/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$uid" "$DEST"
echo "Loaded $LABEL: every 5 minutes, log ~/AIContent/logs/watchdog.log, result ~/AIContent/logs/watchdog.json"
