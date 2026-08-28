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

launchctl unload "$LEGACY_PLIST" 2>/dev/null || true
rm -f "$LEGACY_PLIST"
launchctl unload "$DEST_PLIST" 2>/dev/null || true
launchctl load "$DEST_PLIST"

echo "Installed Tailscale Serve tooling:"
echo "  Config:      $CONFIG_DIR/serve.json"
echo "  Apply:       $CONFIG_DIR/apply-serve.py"
echo "  LaunchAgent: $DEST_PLIST"
echo ""
echo "Edit serve.json for all local services, then run:"
echo "  python3 $CONFIG_DIR/apply-serve.py"
echo ""
echo "Tailscale app must stay signed in (login item)."
