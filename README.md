# Local Video → Obsidian Extraction Pipeline

One-click capture from Bilibili, YouTube, Xiaohongshu, RedNote, Threads, LinkedIn, X (Twitter) and Xiaoyuzhou (小宇宙) podcasts.
The local service downloads the media, then for video posts extracts a timestamped speaker-aware transcript and writes an Obsidian note.
Keyframe extraction is off by default because it is slow.
Check **Extract keyframes** in the extension, pass `--keyframes` on the CLI, or click **Extract keyframes** on the dashboard to add stills and `multimodal.json`.
Image/text posts (Xiaohongshu, Threads, LinkedIn) are downloaded into the vault without ASR or visual extraction.
LinkedIn posts (`linkedin.com/posts/...`, `/feed/update/...`, and `lnkd.in/p/...` share links) are read from the public
post page: the text, the post it reshares, images or the video, with `lnkd.in` links in the text resolved to their
destination. Posts that need a login are reported as such.
Xiaoyuzhou episodes (`xiaoyuzhoufm.com/episode/...`): the public audio is transcribed like a video, and the show notes
(timeline, guests, links) become the description. Paid episodes are reported as such.
X posts (`x.com/<user>/status/<id>`, also twitter.com): read through FxTwitter's public API, so the text, the quoted
post, photos and video (transcribed) come through without a login, and X Articles arrive in full (title, headings,
lists). If FxTwitter is down, X's embed endpoint is used, which gives an article's title and preview only.

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

## Gemini via agy (no API key)

If the [Antigravity CLI](https://antigravity.google) `agy` is installed and signed in, pick **Gemini via agy** (`agy-gemini`) as the ASR model.
It transcribes and tags speakers in one pass with your own Google login, and pyannote is skipped.

How it runs:

- Audio is cut into 3-minute windows that overlap by 45 s; up to 3 run at once. Longer windows let Gemini's timestamps drift by 20 s or more.
- Each window is copied alone into a temp folder and `agy -p` runs there with `--sandbox`, `--mode accept-edits` and `allowNonWorkspaceAccess: false`. Any tool call other than reading that one file aborts the window.
- Gemini numbers speakers per window; the overlap is transcribed twice and votes on which label is which speaker.
- Gemini timestamps drift by a few seconds, so segment edges are snapped to local VAD speech boundaries.
- Window 1 runs first and returns a speaker roster ("male host…", "female guest Kay…"); later windows run in parallel with that roster and reuse its labels.
- Gemini's clock runs ~10% fast inside a window, so timestamps that overrun the window are rescaled before snapping.
- Default model is `gemini-4-argon-medium`. On a 24-minute two-person interview it returned exactly 2 speakers (host asked 29/29 questions) in ~3 minutes; Gemini 3.8 Flash split the same audio into 4–6 speakers.

Settings live under `agy:` in the config (`asr_model`, `window_sec`, `overlap_sec`, `concurrency`, `timeout_sec`).

## Alibaba Cloud (DashScope) mode

Everything heavy can run on Alibaba Cloud Model Studio (百炼) instead of this Mac:

| Stage | Cloud model | Setting |
|-------|-------------|---------|
| ASR + speaker tags | `fun-asr` (one call, `diarization_enabled`) | `asr.default: fun-asr` or pick **Fun-ASR** in the extension |
| Keyframe judging | `qwen3.8-flash` | `analysis.provider: dashscope` |
| Summary | `qwen3.8-max` | `summary.providers: [qwen3.8-max]` |

```yaml
dashscope:
  api_key: ""        # or export DASHSCOPE_API_KEY
  base_url: ""       # empty = Beijing; https://dashscope-intl.aliyuncs.com for a Singapore key
  speaker_count: 0   # 0 = auto
asr:
  default: fun-asr
analysis:
  provider: dashscope
summary:
  providers: [qwen3.8-max]
```

How Fun-ASR is called: the 16 kHz WAV is re-encoded to ~48 kbps AAC, uploaded to DashScope temporary storage (`oss://`, deleted after 48 h), transcribed asynchronously, then polled.
Each sentence comes back with a `speaker_id`, so the pyannote stage is skipped and no Hugging Face token is needed.
Cloud stages do not take the local `worker.model_jobs` slot, and keyframes are judged `dashscope.vision_concurrency` at a time.
Diarization is recommended for audio under 2 hours.

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

The dashboard is `http://127.0.0.1:8875/`: a reader built for phones first (library → note with summary, speaker-coloured transcript with timestamps that open the video at that moment, and frames).
On iPhone, open the Tailscale URL in Safari and use Share → Add to Home Screen for a full-screen app.
Settings, keyframe decisions and prompt editing are in the previous dashboard at `/classic`.
While a job is running, the detail page polls about every 4 seconds.
After a task is `completed` (and summary is not regenerating), it stops rewriting the extracted note and summary so text selection is not cleared.

To start it at login:

```bash
./scripts/install-launch-agent.sh
```

The LaunchAgent runs `media-pipeline serve --host 127.0.0.1 --port 8875`.
It uses `KeepAlive` and a 30s `ThrottleInterval` so the API comes back after crashes or login.
Bind is **localhost only**: Tailscale Serve already owns the Tailscale IP on port 8875; binding `0.0.0.0` fights that listener and launchd will thrash then stop.

### iPhone share sheet (Media Extract shortcut)

`scripts/shortcuts/build_media_extract.py` builds the "Media Extract" shortcut: Share → Media Extract sends the shared link or share text to `/v1/inbox` over Tailscale and shows the server's reply as a notification (`✅ 已加入队列 · YouTube`, or `❌ …` with the reason).
Run with no share input, it uses the clipboard.

```bash
python3 scripts/shortcuts/build_media_extract.py /tmp/me.unsigned.shortcut
shortcuts sign --mode anyone --input /tmp/me.unsigned.shortcut --output "Media Extract.shortcut"
```

Open the signed file on the iPhone (e.g. from iCloud Drive) to install it.
Edit the Tailscale host at the top of the script for another machine.
Text parameters must be serialized as `WFTextTokenString`; a `WFTextTokenAttachment` there silently resolves to an empty string (the v1 shortcut sent `url=` for every share).

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

## Disk usage

Media does not pile up on the Mac, and originals are not thrown away.
After a video is processed:

- the 16 kHz WAV (derived, ~115 MB/hour) is replaced by a mono 48 kbps copy (~22 MB/hour) used to re-transcribe; it goes after `storage.keep_audio_days` (30);
- the original download follows `storage.mode`. **archive** (the default with iCloud Drive) moves it to `iCloud Drive/Media Intelligence/videos/`; once iCloud has it, it is evicted from this Mac (`brctl evict`, which iCloud refuses until the upload is done) and keeps only its name locally. **keep** leaves it in `~/AIContent/videos`, **delete** removes it.

The pipeline only touches media when a stage needs it: re-sharing a finished video reuses its transcript, a different ASR model rebuilds the WAV from the audio copy, and extracting keyframes later fetches the archived original back from iCloud (`brctl download`; services may not be allowed to fault in cloud-only files on their own).
The worker sweeps at startup and every 30 minutes in archive mode (6 hours otherwise).

```bash
uv run media-pipeline cleanup --dry-run   # what would be compacted, archived or evicted
uv run media-pipeline cleanup
```

## Bulk downloads (a whole channel)

`media-pipeline bulk` downloads a channel or playlist in the background, optionally transcribes each video as it
arrives, and keeps a per-video table of every stage (download, transcription, and pipeline processing if any).
Transcripts are saved where the pipeline keeps them, so summarizing a video later reuses its file and transcript
instead of downloading or paying for ASR again.

```bash
uv run media-pipeline bulk add aie --source 'https://www.youtube.com/@aiDotEngineer/videos' --title 'AI Engineer'
uv run media-pipeline bulk asr aie                   # also transcribe (asr.default, e.g. fun-asr; --off to stop)
uv run media-pipeline bulk summary aie               # and summarize at half price (deepseek-v4.1-flash, 22:00-08:00 Beijing)
uv run media-pipeline bulk import-aie aie            # transcripts, chapters, slides from ai.engineer/data (see below)
uv run media-pipeline bulk archive-aie aie           # every published file of the library, to <dest>/official
uv run media-pipeline bulk youtube aie off           # stop downloading from YouTube; ASR, summaries, NAS moves go on
uv run media-pipeline bulk pace aie 45-90            # seconds between downloads, applied while it runs
uv run media-pipeline bulk run aie --max 3          # a trial in the foreground
./scripts/install-bulk-agent.sh aie                  # the whole job as a LaunchAgent (resumes after reboot)
uv run media-pipeline bulk status aie
uv run media-pipeline bulk dest aie '/Volumes/<share>/Media Intelligence/AI Engineer'   # NAS, once mounted
```

- **Progress table**: `http://<mac>.ts.net:8875/bulk` (phone-friendly, filters per stage, search, CSV export, pause/resume, each transcript one tap away).
- **AI Engineer's official talk library** (`bulk import-aie aie`, from [ai.engineer/data](https://ai.engineer/data)): the organizers publish transcripts, chapters and deduplicated 1080p slides (with titles, alt text and on-slide text) for ~1,460 talks as public JSON. The import matches them to the job's YouTube ids and saves the transcript (a free `ai-engineer` transcript), chapters, all slide text and 8 slide images per talk; summaries then see the slides (Flash ≈ 970 tokens per image) and need no download. Requests are paced under the site's 60/min limit. The separate Hugging Face research dataset is restricted and not used.
- **YouTube's own captions and chapters** come with each download (`youtube_extras`, on for YouTube jobs): metadata and chapters are saved with the video, and captions become a free `youtube-captions` transcript (no speaker turns). YouTube rate-limits its captions endpoint on its own (HTTP 429 after a few dozen requests), so captions are spaced out (≥ 90 s, older downloads ≥ 3 min) and a 429 only cools captions down while videos keep downloading.
- **Transcription** (`bulk asr`, `--policy captions-first` by default) runs alongside the downloads, 3 at a time: the ASR model only gets videos without captions, or still without them 6 hours after the download. Failures are retried twice, 5 minutes apart; five in a row pause transcription for 10 minutes. The page shows what transcripts cost.
- **Summaries** (`bulk summary aie`, DeepSeek V4.1 Flash by default) turn each transcribed video into a normal note + summary, reusing its file, metadata and transcript, only during Alibaba Cloud's half-price hours (22:00-08:00 Beijing time), 4 at a time. The chapters go into the prompt as the speaker's outline. Notes land in `<notes>/<collection>/` (e.g. `Transcripts/AI Engineer/`); in the reader the collection has its own 📚 chip, so the library itself stays yours.
- **Polite by design**: one video at a time, 8–20 s apart, guest session (no browser cookies, so the Google account is never at risk), 720p H.264 + the original audio track (about 150 MB per hour of talk).
- **YouTube limits are detected and handled**: bot checks, HTTP 429 or "rate-limited / try again later" pause the job for 15 minutes (doubling up to 4 hours), and one probe request must succeed before it resumes. Four failures in a row, or the last five downloads running below 150 KB/s (throttling), pause it too. Private, removed, members-only and age-restricted videos are marked *skipped*. Every pause and resume is listed under 事件 on the page.
- **Storage**: files download into `~/AIContent/bulk/<job>/media/` and are copied (size-checked) to `dest` once its share is mounted; downloading pauses while the Mac has less than 40 GB free (`--min-free-gb`).
- **Self-check**: `./scripts/install-watchdog.sh` runs `media-pipeline watchdog` every 5 minutes. It checks the API, each bulk job's download and transcription heartbeats, and disk space; a problem seen twice in a row restarts the stuck LaunchAgent (so recovery takes at most ~10 minutes). YouTube is not contacted, since the job itself watches every download. The result is shown on `/bulk`.
- yt-dlp gets a JavaScript runtime (deno, node or bun, whichever is installed). Without one YouTube only offers its HLS formats, which are about 6× bigger.

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

After pulling API/port changes, click **Reload** on the unpacked extension card.
If the popup still says the local service is not running while `curl http://127.0.0.1:8875/v1/health` works, open the extension’s service worker console and clear a stale `apiBase` with:

```js
chrome.storage.local.remove("apiBase")
```

Then reload the extension again.
The default base URL is `http://127.0.0.1:8875`.

The button appears on Bilibili, YouTube, Xiaohongshu, RedNote (`rednote.com`), Threads and LinkedIn post pages, including `xhslink.com` and `lnkd.in` short links.
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
