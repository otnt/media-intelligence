#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required. Install it with: brew install ffmpeg" >&2
  exit 1
fi

uv python pin 3.13
uv venv --python 3.13
uv pip install -e ".[whisper,qwen,dev]"
uv run python scripts/generate_icons.py
uv run media-pipeline init

mkdir -p "$HOME/AIContent/videos" "$HOME/AIContent/audio" "$HOME/AIContent/artifacts" "$HOME/AIContent/logs"

echo
echo "Installed the local pipeline."
echo
echo "1. Start the background service:"
echo "   uv run media-pipeline serve"
echo
echo "2. Optional speaker diarization:"
echo "   uv pip install -e '.[diarize]'"
echo "   For speaker diarization, set diarization.hf_token in ~/.config/media-pipeline/config.yaml"
echo
echo "3. Chrome → chrome://extensions → Developer mode → Load unpacked"
echo "   Select: $ROOT/extension"
echo
echo "4. Open a Bilibili, YouTube, or Xiaohongshu post and click ✨ Extract"
echo "5. Inspect tasks and keyframes at http://127.0.0.1:8875/"
