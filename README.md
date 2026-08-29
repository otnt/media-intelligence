# Local Video → Obsidian Extraction Pipeline

One-click capture from Bilibili, YouTube, Xiaohongshu, and RedNote.
The local service downloads the media, then for video posts extracts a timestamped speaker-aware transcript and writes an Obsidian note.
Keyframe extraction is off by default because it is slow.
Check **Extract keyframes** in the extension, pass `--keyframes` on the CLI, or click **Extract keyframes** on the dashboard to add stills and `multimodal.json`.
Image/text Xiaohongshu posts are downloaded into the vault without ASR or visual extraction.

## What V1 does

Open a Bilibili, YouTube, Xiaohongshu, or RedNote page.
Click **✨ Extract**.
Choose an ASR model.
Leave **Extract keyframes** unchecked unless you want stills in the note.
Then submit.
The browser can be closed.
Inspect progress at `http://127.0.0.1:8875/`.
Later, the Obsidian vault contains metadata and a transcript for videos, or the post text and images for Xiaohongshu image posts.
If you opted into keyframes, the note also includes a visual timeline.

This stage is high-recall extraction, not summarization.

Supported ASR models:

- MLX Whisper — Whisper large-v3
- MLX Whisper — Whisper large-v3-turbo
- MLX Qwen3-ASR-1.7B (multilingual, including mixed Chinese/English; ~3.4 GB on Apple Silicon)

ASR and speaker diarization are separate stages.
Whisper and Qwen3-ASR answer *what was said*.
A dedicated diarization stage answers *who spoke when*.
Names are assigned only when the transcript or metadata contains evidence.

Language handling:

- Default is `language: auto`.
- Whisper auto-detects **one** language for the whole file, then transcribes in that language. Mixed Chinese/English speech often gets mangled.
- Qwen3-ASR-1.7B auto-detects per utterance and can keep mixed-language / code-switched speech. Forcing `Chinese` or `English` turns that off.
- Qwen runs on MLX, the same Apple Silicon stack as Whisper.
- Qwen timestamps are coarse segment start/end times from VAD speech windows, not word-level forced alignment.
- Install the Qwen extra before choosing that model: `uv pip install -e '.[qwen]'`.

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

The dashboard is `http://127.0.0.1:8875/`.

To start it at login:

```bash
./scripts/install-launch-agent.sh
```

The LaunchAgent runs `media-pipeline serve --host 127.0.0.1 --port 8875`.
It uses `KeepAlive` and a 30s `ThrottleInterval` so the API comes back after crashes or login.
Bind is **localhost only**: Tailscale Serve already owns the Tailscale IP on port 8875; binding `0.0.0.0` fights that listener and launchd will thrash then stop.

### LAN and Tailscale

Use Tailscale for phone and off-LAN access (including while you are on home Wi‑Fi):

```text
http://<your-machine>.ts.net:8875/
```

Do not run LaunchAgent with `--lan` while Serve publishes the same port.
For local debugging, use `http://127.0.0.1:8875/`.

Tailscale Serve is **not** owned by this repo.
Install it once from here:

```bash
./scripts/tailscale/install.sh
```

That deploys to `~/.config/tailscale/`:

| Path | Role |
|------|------|
| `~/.config/tailscale/serve.json` | Routes for all local services on this Mac |
| `~/.config/tailscale/apply-serve.py` | Applies `serve.json` via the Tailscale CLI |
| `~/Library/LaunchAgents/com.tailscale.serve.plist` | Re-applies Serve after login and every 5 minutes |

The example in `scripts/tailscale/serve.example.json` publishes media-pipeline at `http://<your-machine>.ts.net:8875/` → `http://127.0.0.1:8875`.
Edit `~/.config/tailscale/serve.json` to add other backends on other ports, then run:

```bash
python3 ~/.config/tailscale/apply-serve.py
```

Host keys like `:8875` are expanded with your MagicDNS name automatically.
Tailscale Serve is tailnet-only; do not enable Funnel.
Set `server.ingest_token` in `~/.config/media-pipeline/config.yaml` before exposing the API off localhost.

Check the machine:

```bash
uv run media-pipeline doctor
```

## Command line

Transcribe one Bilibili, YouTube, or Xiaohongshu URL without the browser:

```bash
uv run media-pipeline transcribe 'https://www.bilibili.com/video/BVxxxx'
uv run media-pipeline transcribe 'http://xhslink.com/o/xxxxxxxx'
uv run media-pipeline transcribe --keyframes 'https://www.bilibili.com/video/BVxxxx'
```

Xiaohongshu video posts follow the same download → audio → ASR path as Bilibili and YouTube.
Pass `--keyframes` to also run scene detection, stills, and the visual timeline.
Image/text posts are saved to `~/AIContent/videos/{note_id}/` and copied into the vault as wikilinked attachments.
Xiaohongshu uses Chrome cookies (`download.cookies_from_browser`) and `curl-cffi` to fetch note pages.

The command runs in the foreground and writes the Obsidian note as soon as metadata is available, then fills in the transcript.
Visual stills are included only when `--keyframes` is set.
The default ASR model is `qwen3-asr-1.7b` with `language: auto`, so mixed Chinese/English is kept when the model can hear it.
Use `--asr-model whisper-large-v3-turbo` or `whisper-large-v3` to pick Whisper instead.

## Load the Chrome extension

1. Open `chrome://extensions`.
2. Enable **Developer mode**.
3. Click **Load unpacked**.
4. Select the `extension` folder in this repository.

The button appears on Bilibili, YouTube, Xiaohongshu, and RedNote (`rednote.com`) post pages, including `xhslink.com` short links.
On the Explore feed, search, and profile grids, each post card also gets an **✨ Extract** button.
One click queues that note; ASR model and keyframe settings come from the floating panel or the extension popup.
The last ASR model you used is remembered.
**Extract keyframes** stays off unless you check it; that choice is remembered too.

## Output

Videos land in `~/AIContent/videos/`.
Xiaohongshu image posts land in `~/AIContent/videos/{note_id}/`.
Audio lands in `~/AIContent/audio/`.
Intermediate JSON artifacts land in `~/AIContent/artifacts/{video_id}/`.
Obsidian notes land in `{vault}/Transcripts/{Video Title}.md`.

Retrying a failed task reuses downloaded video and extracted audio.
Switching ASR models on the same video reuses the same audio file:

```bash
uv run media-pipeline retry TASK_ID --asr-model whisper-large-v3
uv run media-pipeline retry TASK_ID --stage detecting_scenes
```

Visual extraction reuses the downloaded video.
Changing the sampling interval does not download or transcribe again.

## Local API

The extension talks only to `127.0.0.1`.

```text
POST /v1/tasks
{
  "url": "https://www.bilibili.com/video/BVxxxx",
  "asr_model": "qwen3-asr-1.7b",
  "language": "auto",
  "extract_keyframes": false
}
```

`extract_keyframes` defaults to `false`.
Set it to `true` to run scene detection and stills after ASR.

Task states for videos: `queued` → `fetching_metadata` → `downloading` → `extracting_audio` → `transcribing` → `diarizing` → `aligning` → `completed`.
When keyframes are enabled, scene detection, frame sampling, and multimodal alignment run after the transcript is aligned.
Xiaohongshu image posts skip audio and ASR and go from `downloading` to `writing_outputs` → `completed`.
Any stage may transition to `failed`.

## Out of scope for V1

Summarization, scoring, cloud inference, and other sites are intentionally omitted.
ASR model choice and a future analysis-model choice are separate settings.
