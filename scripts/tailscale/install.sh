#!/usr/bin/env bash
# Install machine-wide Tailscale Serve config under ~/.config/tailscale/.
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/tailscale"
DEST_PLIST="$HOME/Library/LaunchAgents/com.tailscale.serve.plist"
LEGACY_PLIST="$HOME/Library/LaunchAgents/com.media-pipeline.tailscale-serve.plist"

if ! command -v tailscale >/dev/null 2>&1 && \
   [[ ! -x /Applications/Tailscale.app/Contents/MacOS/Tailscale ]]; then
  echo "Tailscale is not installed. Install: brew install --cask tailscale-app" >&2
  exit 1
fi

mkdir -p "$CONFIG_DIR" "$HOME/Library/LaunchAgents" "$HOME/AIContent/logs"
install -m 755 "$SRC/apply-serve.py" "$CONFIG_DIR/apply-serve.py"

if [[ ! -f "$CONFIG_DIR/serve.json" ]]; then
  install -m 644 "$SRC/serve.example.json" "$CONFIG_DIR/serve.json"
  echo "Created $CONFIG_DIR/serve.json from example."
else
  echo "Keeping existing $CONFIG_DIR/serve.json"
fi

sed \
  -e "s|__CONFIG_DIR__|$CONFIG_DIR|g" \
  -e "s|__HOME__|$HOME|g" \
  "$SRC/com.tailscale.serve.plist.template" > "$DEST_PLIST"

uid="$(id -u)"
launchctl bootout "gui/$uid" "$LEGACY_PLIST" 2>/dev/null || launchctl unload "$LEGACY_PLIST" 2>/dev/null || true
rm -f "$LEGACY_PLIST"
launchctl bootout "gui/$uid" "$DEST_PLIST" 2>/dev/null || launchctl unload "$DEST_PLIST" 2>/dev/null || true
launchctl bootstrap "gui/$uid" "$DEST_PLIST" 2>/dev/null || launchctl load "$DEST_PLIST"
launchctl enable "gui/$uid/com.tailscale.serve" 2>/dev/null || true

# Apply once now (short waits). Periodic StartInterval will retry later.
BACKEND_WAIT_ATTEMPTS="${BACKEND_WAIT_ATTEMPTS:-15}" \
TAILSCALE_WAIT_ATTEMPTS="${TAILSCALE_WAIT_ATTEMPTS:-30}" \
  /usr/bin/python3 "$CONFIG_DIR/apply-serve.py" || true

echo "Installed Tailscale Serve tooling:"
echo "  Config:      $CONFIG_DIR/serve.json"
echo "  Apply:       $CONFIG_DIR/apply-serve.py"
echo "  LaunchAgent: $DEST_PLIST (RunAtLoad + every 5 minutes)"
echo ""
echo "Edit serve.json for all local services, then run:"
echo "  python3 $CONFIG_DIR/apply-serve.py"
echo ""
echo "Tailscale app must stay signed in (login item)."
echo "media-pipeline must bind 127.0.0.1:8875 (not 0.0.0.0) while Serve uses :8875."
