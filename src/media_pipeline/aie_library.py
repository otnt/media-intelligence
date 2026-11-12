"""AI Engineer's official public talk library (https://ai.engineer/data, documented in /data.md).

The organizers publish, for their YouTube talks, JSON with no key: talk records (speakers, topics, event,
a summary, slide text), chapters, per-talk transcripts (timestamped paragraphs, automated, no speaker
turns) and per-talk slide manifests (deduplicated 1080p slide images with titles, alt text and OCR text).

``media-pipeline bulk import-aie <job>`` matches the library to a bulk job's videos by YouTube id and saves,
per talk, what the pipeline reads:

* the transcript as transcript model ``ai-engineer`` (``artifacts/<id>/asr/ai-engineer.json``, one speaker),
  which completes the video's transcription stage for free;
* ``aie.json``: the talk record, its chapters and all slide titles/text;
* up to ``slides_per_talk`` slide images in ``artifacts/<id>/slides/`` (+ ``slides.json``), attached to the
  summary for models that see images;
* ``metadata.json`` when the video has none yet, so the pipeline never asks YouTube for it.

Requests are paced below the site's documented limit (60 per minute per IP) and 429/529/503 are retried
after ``Retry-After``. The separate Hugging Face research dataset is restricted and is not used.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from media_pipeline.config import AppConfig
from media_pipeline.models import Transcript, TranscriptSegment, VideoMetadata

logger = logging.getLogger(__name__)

BASE = "https://ai.engineer/api/data"
USER_AGENT = "media-intelligence/0.1 (personal research; paced import of the public talk library)"
TRANSCRIPT_MODEL = "ai-engineer"
SPEAKER = "SPEAKER_00"
SLIDES_DIR = "slides"


class Library:
    """Paced, polite GETs: API calls ``api_interval`` apart, slide images ``asset_interval`` apart."""

    def __init__(
        self,
        fetch: Callable[[str], bytes] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        api_interval: float = 1.5,
        asset_interval: float = 0.3,
    ) -> None:
        self._fetch = fetch or _http_get
        self._sleep = sleep
        self._intervals = {"api": api_interval, "asset": asset_interval}
        self._last = {"api": 0.0, "asset": 0.0}

    def get(self, url: str, kind: str = "api") -> bytes:
        wait = self._last[kind] + self._intervals[kind] - time.monotonic()
        if wait > 0:
            self._sleep(wait)
        for attempt in range(5):
            self._last[kind] = time.monotonic()
            try:
                return self._fetch(url)
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 503, 529) or attempt == 4:
                    raise
                retry_after = _retry_after(exc.headers.get("Retry-After") if exc.headers else None)
                self._sleep(retry_after or 30.0 * (attempt + 1))
        raise RuntimeError(f"giving up on {url}")

    def json(self, url: str, kind: str = "api") -> Any:
        return json.loads(self.get(url, kind).decode("utf-8"))


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json, */*"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _retry_after(value: str | None) -> float | None:
    try:
        return max(1.0, float(value)) if value else None
    except ValueError:
        return None


def _rows(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                return value
    return []


def to_transcript(segments: list[dict[str, Any]], duration_ms: float | None) -> Transcript:
    """Timestamped paragraphs → segments ending where the next one starts."""
    items = sorted((s for s in segments if str(s.get("text") or "").strip()), key=lambda s: float(s.get("startMs") or 0))
    out: list[TranscriptSegment] = []
    for index, item in enumerate(items):
        start = float(item.get("startMs") or 0) / 1000.0
        if item.get("endMs") is not None:
            end = float(item["endMs"]) / 1000.0
        elif index + 1 < len(items):
            end = float(items[index + 1].get("startMs") or 0) / 1000.0
        else:
            end = float(duration_ms or 0) / 1000.0 or start
        out.append(TranscriptSegment(start, max(end, start), " ".join(str(item["text"]).split()), speaker=SPEAKER))
    return Transcript(language="en", segments=out, provider="ai.engineer", model=TRANSCRIPT_MODEL)


def slide_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """One entry per deduplicated slide: its first frame, title, alt text and OCR text, in time order."""
    index = manifest.get("index") if isinstance(manifest.get("index"), dict) else manifest
    entries = []
    for group in index.get("groups") or []:
        if str(group.get("kind") or "slide") != "slide":
            continue
        frames = [f for f in group.get("frames") or [] if f.get("src")]
        if not frames:
            continue
        frame = frames[0]
        entries.append({
            "title": str(group.get("title") or "").strip(),
            "startMs": float(frame.get("captureMs") or frame.get("startMs") or 0),
            "src": str(frame["src"]),
            "alt": str(frame.get("alt") or "").strip(),
            "text": " ".join(str(frame.get("text") or "").split()),
        })
    return sorted(entries, key=lambda e: e["startMs"])


def pick_spread(entries: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or not entries:
        return []
    if len(entries) <= limit:
        return list(entries)
    step = (len(entries) - 1) / (limit - 1) if limit > 1 else 0
    return [entries[round(i * step)] for i in range(limit)]


def import_job(
    config: AppConfig,
    store: Any,
    *,
    library: Library | None = None,
    slides_per_talk: int = 8,
    refresh: bool = False,
    limit: int | None = None,
    log: Callable[[str], None] = logger.info,
) -> dict[str, int]:
    from media_pipeline.artifacts import ArtifactStore

    library = library or Library()
    talks = {t["videoId"]: t for t in _rows(library.json(f"{BASE}/talks?format=json")) if t.get("videoId")}
    chapters: dict[str, list[dict[str, Any]]] = {}
    for row in _rows(library.json(f"{BASE}/chapters?format=json")):
        chapters.setdefault(str(row.get("talkSlug")), []).append(row)
    transcripts = {str(r.get("talkSlug")): r.get("jsonUrl") for r in _rows(library.json(f"{BASE}/transcripts?format=json"))}
    videos = {row["id"]: row for row in store.rows()}
    matched = [vid for vid in videos if vid in talks]
    stats = {"talks": len(talks), "matched": len(matched), "transcripts": 0, "slides": 0, "images": 0, "failed": 0}
    log(f"library: {len(talks)} talks, {len(transcripts)} transcripts; {len(matched)} match the job's {len(videos)} videos")
    for count, vid in enumerate(matched[:limit] if limit else matched, start=1):
        talk = talks[vid]
        slug = str(talk.get("slug") or "")
        artifacts = ArtifactStore(config.paths.artifacts, vid)
        try:
            talk_chapters = [
                {"start": float(c.get("startMs") or 0) / 1000.0, "end": float(c.get("endMs") or 0) / 1000.0,
                 "title": str(c.get("title") or "").strip()}
                for c in sorted(chapters.get(slug, []), key=lambda c: float(c.get("startMs") or 0))
                if str(c.get("title") or "").strip()
            ]
            record_path = artifacts.root / "aie.json"
            record = artifacts.read_json(record_path) if record_path.is_file() and not refresh else {}
            slides = record.get("slides") or []
            if not slides and isinstance(talk.get("slides"), dict) and talk["slides"].get("url"):
                slides = slide_entries(library.json(talk["slides"]["url"], kind="api"))
            artifacts.write_json(record_path, {
                "slug": slug, "url": talk.get("url"), "title": talk.get("title"), "event": talk.get("event"),
                "speakers": [s.get("name") for s in talk.get("speakers") or [] if s.get("name")],
                "topics": [t.get("name") for t in talk.get("topics") or [] if t.get("name")],
                "summary": talk.get("summary") or "", "durationMs": talk.get("durationMs"),
                "chapters": talk_chapters, "slides": slides,
            })
            if not artifacts.metadata_path.exists():
                speakers = ", ".join(s.get("name") for s in talk.get("speakers") or [] if s.get("name"))
                artifacts.save_metadata(VideoMetadata(
                    url=f"https://www.youtube.com/watch?v={vid}", title=str(talk.get("title") or vid), platform="YouTube",
                    author=speakers or "AI Engineer", video_id=vid,
                    duration=float(talk["durationMs"]) / 1000.0 if talk.get("durationMs") else None,
                    published="", description=str(talk.get("summary") or ""), thumbnail_url="",
                    asr_model=TRANSCRIPT_MODEL,
                ))
            transcript = artifacts.load_transcript(TRANSCRIPT_MODEL) if not refresh else None
            if (transcript is None or not transcript.segments) and transcripts.get(slug):
                transcript = to_transcript(_rows(library.json(transcripts[slug])), talk.get("durationMs"))
                if transcript.segments:
                    artifacts.save_transcript(TRANSCRIPT_MODEL, transcript)
            stats["images"] += _save_slide_images(library, artifacts, pick_spread(slides, slides_per_talk))
            if slides:
                stats["slides"] += 1
            if transcript is not None and transcript.segments:
                stats["transcripts"] += 1
                store.execute(
                    "UPDATE videos SET asr_status='done', transcript_source='ai-engineer', segments=?, speakers=1, "
                    "language='en', asr_error='', asr_finished_at=? WHERE id=? AND (asr_status!='done' OR "
                    "transcript_source IN ('', 'youtube'))",
                    (len(transcript.segments), time.time(), vid))
                # YouTube's captions are not needed any more: one request less per video.
                store.execute("UPDATE videos SET captions_status='skipped' WHERE id=? AND captions_status IN "
                              "('pending', 'failed')", (vid,))
            store.execute("UPDATE videos SET chapters=? WHERE id=? AND COALESCE(chapters, 0) < ?",
                          (len(talk_chapters), vid, len(talk_chapters)))
        except Exception as exc:  # noqa: BLE001 - one talk must not stop the import
            stats["failed"] += 1
            log(f"failed {vid} ({slug}): {exc}")
        if count % 50 == 0:
            log(f"{count}/{len(matched)} talks · {stats['transcripts']} transcripts · {stats['images']} slide images")
    log(f"done: {stats}")
    return stats


def _save_slide_images(library: Library, artifacts: Any, chosen: list[dict[str, Any]]) -> int:
    if not chosen:
        return 0
    folder = artifacts.root / SLIDES_DIR
    folder.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []
    saved = 0
    for entry in chosen:
        # Named after the image itself (its content hash), so a file always holds the same slide.
        name = Path(entry["src"].split("?")[0]).name or "slide.webp"
        target = folder / "".join(ch for ch in name if ch.isalnum() or ch in "._-")
        if not (target.is_file() and target.stat().st_size > 0):
            target.write_bytes(library.get(entry["src"], kind="asset"))
            saved += 1
        index.append({"file": target.name, "startMs": entry["startMs"], "title": entry["title"], "alt": entry["alt"]})
    artifacts.write_json(folder / "slides.json", index)
    return saved


ARCHIVE_COLLECTIONS = ("talks", "speakers", "topics", "organizations", "chapters", "transcripts")


def archive_library(
    dest: Path,
    *,
    library: Library | None = None,
    artifacts_root: Path | None = None,
    log: Callable[[str], None] = logger.info,
) -> dict[str, int]:
    """Everything the organizers publish, saved as files under ``dest`` (resumable; existing files are kept):

    index/  the six collections as JSON and CSV, the data status and the guide (data.md)
    talks/<videoId>-<slug>/  talk.json (record), talk.md (write-up, slides and timestamped transcript),
                             transcript.json, slides.json (manifest) and slides/<sha256>.webp (every slide)
    Slide images already fetched for summaries (``artifacts_root``) are copied instead of downloaded.
    """
    library = library or Library()
    index_dir = dest / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    for name in ARCHIVE_COLLECTIONS:
        for fmt in ("json", "csv"):
            _write_new(index_dir / f"{name}.{fmt}", lambda name=name, fmt=fmt: library.get(f"{BASE}/{name}?format={fmt}"))
    _write_new(index_dir / "status.json", lambda: library.get(f"{BASE}/status"), refresh=True)
    _write_new(index_dir / "data.md", lambda: library.get("https://ai.engineer/data.md"))
    talks = _rows(json.loads((index_dir / "talks.json").read_text(encoding="utf-8")))
    transcripts = {str(r.get("talkSlug")): r.get("jsonUrl")
                   for r in _rows(json.loads((index_dir / "transcripts.json").read_text(encoding="utf-8")))}
    stats = {"talks": len(talks), "markdown": 0, "transcripts": 0, "manifests": 0, "slides": 0,
             "slides_copied": 0, "failed": 0}
    for count, talk in enumerate(talks, start=1):
        vid, slug = str(talk.get("videoId") or ""), str(talk.get("slug") or "")
        folder = dest / "talks" / _safe(f"{vid}-{slug}" if vid else slug)
        try:
            folder.mkdir(parents=True, exist_ok=True)
            _write_new(folder / "talk.json", lambda: json.dumps(talk, ensure_ascii=False, indent=1).encode())
            if talk.get("url") and _write_new(folder / "talk.md", lambda: library.get(str(talk["url"]) + ".md")):
                stats["markdown"] += 1
            if transcripts.get(slug) and _write_new(folder / "transcript.json", lambda: library.get(transcripts[slug])):
                stats["transcripts"] += 1
            manifest_url = (talk.get("slides") or {}).get("url") if isinstance(talk.get("slides"), dict) else None
            if manifest_url:
                if _write_new(folder / "slides.json", lambda: library.get(manifest_url, kind="asset")):
                    stats["manifests"] += 1
                manifest = json.loads((folder / "slides.json").read_text(encoding="utf-8"))
                for entry in slide_entries(manifest):
                    name = "".join(ch for ch in Path(entry["src"].split("?")[0]).name if ch.isalnum() or ch in "._-")
                    target = folder / "slides" / name
                    if target.is_file() and target.stat().st_size > 0:
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    local = artifacts_root / vid / SLIDES_DIR / name if artifacts_root and vid else None
                    if local is not None and local.is_file() and local.stat().st_size > 0:
                        _write_new(target, local.read_bytes)
                        stats["slides_copied"] += 1
                    else:
                        _write_new(target, lambda src=entry["src"]: library.get(src, kind="asset"))
                        stats["slides"] += 1
        except Exception as exc:  # noqa: BLE001 - one talk must not stop the archive
            stats["failed"] += 1
            log(f"archive failed for {vid} {slug}: {exc}")
        if count % 50 == 0:
            log(f"archive {count}/{len(talks)} talks · {stats}")
    (dest / "README.md").write_text(
        "# AI Engineer official talk library (archive)\n\n"
        "Downloaded from https://ai.engineer/data (public JSON published by the AI Engineer organizers; the guide\n"
        "is in index/data.md). Transcripts are automated and may contain errors.\n\n"
        "- index/: talks, speakers, topics, organizations, chapters and transcripts as JSON and CSV; status.json\n"
        "- talks/<videoId>-<slug>/: talk.json, talk.md (write-up + slides + timestamped transcript),\n"
        "  transcript.json, slides.json and slides/*.webp (every deduplicated slide)\n\n"
        f"Last run: {time.strftime('%Y-%m-%d %H:%M')} · {json.dumps(stats)}\n",
        encoding="utf-8",
    )
    log(f"archive done: {stats}")
    return stats


def _write_new(path: Path, produce: Callable[[], bytes], refresh: bool = False) -> bool:
    """Write ``produce()`` to ``path`` atomically unless it is already there. True if written."""
    if not refresh and path.is_file() and path.stat().st_size > 0:
        return False
    data = produce()
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(data)
    partial.replace(path)
    return True


def _safe(name: str) -> str:
    return "".join(ch for ch in name if ch.isalnum() or ch in "._-")[:160] or "talk"
