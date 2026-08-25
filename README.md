# Local Video → Obsidian Transcription Pipeline

One-click capture from Bilibili and YouTube.
The local service downloads the video, extracts speech audio, transcribes it, labels speakers, and writes an Obsidian note.

## What V1 does

Open a Bilibili or YouTube video page.
Click **✨ Save & Transcribe**.
Choose an ASR model, then click **Start**.
The browser can be closed.
Later, the Obsidian vault contains a note with metadata and a timestamped, speaker-aware transcript.

Supported ASR models:

- MLX Whisper — Whisper large-v3
- MLX Whisper — Whisper large-v3-turbo
- Qwen3-ASR-1.7B

ASR and speaker diarization are separate stages.
Whisper and Qwen3-ASR answer *what was said*.
A dedicated diarization stage answers *who spoke when*.
Names are assigned only when the transcript or metadata contains evidence.

## Requirements

- Apple Silicon Mac
- Python 3.11–3.13 (`uv` pins 3.13)
- [ffmpeg](https://ffmpeg.org/)
- Google Chrome
- An Obsidian vault

The installer looks up the open vault from Obsidian's local config.
Override `paths.vault` in `~/.config/media-pipeline/config.yaml` if needed.

## Install

```bash
./scripts/install.sh
```

That creates a virtualenv, installs MLX Whisper, writes a default config, and prepares `~/AIContent/`.

Optional extras:

```bash
uv pip install -e '.[qwen,diarize]'
```

Qwen3-ASR needs the `qwen` extra.
Speaker diarization needs `pyannote.audio` plus a Hugging Face token in `diarization.hf_token`.
If diarization is unavailable, the transcript still completes and speakers stay as `Speaker 1`.

Keep the local service running:

```bash
uv run media-pipeline serve
```

To start it at login:

```bash
./scripts/install-launch-agent.sh
```

Check the machine:

```bash
uv run media-pipeline doctor
```

## Load the Chrome extension

1. Open `chrome://extensions`.
2. Enable **Developer mode**.
3. Click **Load unpacked**.
4. Select the `extension` folder in this repository.

The button appears on `bilibili.com/video/...` and `youtube.com/watch?...`.
The last ASR model you used is remembered.

## Output

Videos land in `~/AIContent/videos/`.
Audio lands in `~/AIContent/audio/`.
Intermediate JSON artifacts land in `~/AIContent/artifacts/{video_id}/`.
Obsidian notes land in `{vault}/Transcripts/{Video Title}.md`.

Retrying a failed task reuses downloaded video and extracted audio.
Switching ASR models on the same video reuses the same audio file:

```bash
uv run media-pipeline retry TASK_ID --asr-model whisper-large-v3
```

## Local API

The extension talks only to `127.0.0.1`.

```text
POST /v1/tasks
{
  "url": "https://www.bilibili.com/video/BVxxxx",
  "asr_model": "whisper-large-v3-turbo"
}
```

Task states: `queued` → `fetching_metadata` → `downloading` → `extracting_audio` → `transcribing` → `diarizing` → `aligning` → `completed`.
Any stage may transition to `failed`.

## Out of scope for V1

Summarization, scoring, keyframes, OCR, cloud inference, and other sites are intentionally omitted.
ASR model choice and a future analysis-model choice are separate settings.
