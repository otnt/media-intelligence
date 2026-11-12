"""Bulk jobs (a whole YouTube channel): download, then transcribe, with a per-video, per-stage table.

``media-pipeline bulk add NAME --source URL|FILE`` lists the videos once; ``bulk run NAME`` (normally a
LaunchAgent, see ``scripts/install-bulk-agent.sh``) works through them and records every video's state
for each stage in ``<paths.bulk>/NAME/progress.sqlite3``. The web UI shows the table at ``/bulk``.

**Download stage.** Polite on purpose: one at a time, a random pause between videos, and no browser
cookies (a guest session cannot get the owner's Google account flagged). YouTube limits are watched all
the time:

* bot checks, HTTP 429 and "rate-limited / try again later" pause downloading (15 min, doubling up to
  4 h); one probe request must succeed before it resumes;
* several failures in a row, or a sustained drop in download speed (throttling), also pause it;
* private, removed, members-only or age-restricted videos are marked ``skipped`` (not a block).

Videos download into a local staging folder and move to ``dest`` (e.g. a NAS share) once it is mounted,
so a half-written file never sits on the NAS. Downloading pauses while the Mac has less than
``min_free_gb`` free.

**YouTube extras** (``youtube_extras``, on for YouTube jobs). The extraction that downloads a video also
yields its metadata and chapters (saved as ``metadata.json`` and ``youtube.json``); captions need one more
request and become a ``youtube-captions`` transcript. YouTube rate-limits the captions endpoint on its own
(HTTP 429 after a few dozen requests), so captions are fetched at most once every 45 s by their own worker,
and a 429 only cools captions down (30 min, doubling up to 4 h) while videos keep downloading. Videos
downloaded before this existed are backfilled; ones that already have a transcript only get their
metadata and chapters.

**Transcription stage** (when the job has an ``asr_model``, e.g. ``fun-asr``). With the default
``transcript_policy`` ``captions-first``, a video with captions is already transcribed (free, one speaker);
the ASR model only runs on videos without captions, or whose captions are still missing 6 hours after the
download (rate limits). It runs alongside the downloads, a few at a time.

**Summary stage** (when the job has a ``summary_model``). Each transcribed video becomes a normal task
through the local server (``/v1/bulk/<job>/videos/<id>/process``): a note in ``<notes>/<collection>/``
and a summary, reusing the file, metadata and transcript. Tasks are only submitted inside
``summary_window`` (default 22:00-08:00 UTC+8, Alibaba Cloud's half-price hours), a few at a time. Transcripts are saved exactly where the pipeline keeps
them (``artifacts/<video_id>/asr/<model>.json``), so processing a video later (summary, note) reuses
the transcript instead of paying for ASR again; the pipeline also finds the downloaded file through
:func:`find_bulk_media`. A file moves to the NAS after it has been transcribed (sooner if the disk is
low).
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import random
import shutil
import sqlite3
import statistics
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from media_pipeline.config import AppConfig

logger = logging.getLogger(__name__)

DB_NAME = "progress.sqlite3"
# 720p is where slide text stays readable for the keyframe/vision steps; H.264 + AAC merge into mp4
# without re-encoding. With a JS runtime YouTube's DASH files are small (~150 MB per hour of talk).
DEFAULT_FORMAT = "bv*[height<=720][vcodec^=avc1]+ba[ext=m4a]/bv*[height<=720]+ba/b[height<=720]/b"
# List prices (Alibaba Cloud international), only for the running cost estimate on the page.
ASR_USD_PER_SEC = {"fun-asr": 0.000035}
BLOCK_MARKERS = (
    "not a bot",
    "http error 429",
    "too many requests",
    "rate-limited",
    "rate limited",
    "try again later",
    "content isn't available",
    "content isn’t available",
)
SKIP_MARKERS = (
    "private video",
    "members-only",
    "join this channel",
    "has been removed",
    "removed for violating",
    "account associated with this video has been terminated",
    "not made this video available in your country",
    "confirm your age",
    "age-restricted",
    "inappropriate for some users",
    "premieres in",
    "live event will begin",
)
ACTIVE_STATES = ("running", "paused", "limited")
# Alibaba Cloud bills DeepSeek models at half price from 22:00 to 08:00 Beijing time.
DEFAULT_SUMMARY_WINDOW = "22:00-08:00@+08:00"
# Measured: a 42-minute talk with DeepSeek V4.1 Flash, 16k tokens in, 4.4k out (thinking included).
SUMMARY_USD_PER_VIDEO = {"deepseek-v4.1-flash": 0.005}
# Columns added after the first jobs were created; added in place when an older table is opened.
LATER_COLUMNS = (
    ("asr_status", "TEXT DEFAULT 'pending'"),
    ("asr_attempts", "INTEGER DEFAULT 0"),
    ("asr_error", "TEXT DEFAULT ''"),
    ("asr_started_at", "REAL"),
    ("asr_finished_at", "REAL"),
    ("asr_seconds", "REAL"),
    ("segments", "INTEGER"),
    ("speakers", "INTEGER"),
    ("language", "TEXT DEFAULT ''"),
    ("transcript_source", "TEXT DEFAULT ''"),
    ("captions_status", "TEXT DEFAULT 'pending'"),
    ("captions_attempts", "INTEGER DEFAULT 0"),
    ("captions_error", "TEXT DEFAULT ''"),
    ("captions_kind", "TEXT DEFAULT ''"),
    ("captions_at", "REAL"),
    ("chapters", "INTEGER"),
    ("task_id", "TEXT DEFAULT ''"),
    ("summary_status", "TEXT DEFAULT ''"),
    ("summary_error", "TEXT DEFAULT ''"),
    ("summary_at", "REAL"),
    ("summary_attempts", "INTEGER DEFAULT 0"),
)


@dataclass
class Limits:
    workers: int = 1
    # ~70-90 videos an hour. 8-20 s pauses plus captions ran into YouTube's bot check after ~85 videos.
    pause_between: tuple[float, float] = (20.0, 40.0)
    min_free_gb: float = 40.0
    slow_bytes_per_sec: float = 150_000.0
    slow_window: int = 5
    slow_min_bytes: int = 5_000_000
    fail_streak: int = 4
    max_attempts: int = 3
    max_blocks_per_video: int = 3
    first_backoff: float = 15 * 60
    max_backoff: float = 4 * 3600
    asr_workers: int = 3
    asr_max_attempts: int = 3
    asr_retry_after: float = 5 * 60
    asr_fail_streak: int = 5
    asr_backoff: float = 10 * 60
    captions_min_interval: float = 90.0  # captions right after a download (one extra request)
    backfill_min_interval: float = 180.0  # a separate metadata extraction for an older download
    captions_wait: float = 6 * 3600
    captions_first_backoff: float = 30 * 60
    captions_max_backoff: float = 4 * 3600
    summary_inflight: int = 4
    summary_max_attempts: int = 2
    summary_retry_after: float = 10 * 60


def classify(error: str) -> str:
    """blocked (YouTube is limiting us), skipped (this video can never be downloaded) or failed."""
    text = (error or "").lower()
    if any(marker in text for marker in BLOCK_MARKERS):
        return "blocked"
    if any(marker in text for marker in SKIP_MARKERS):
        return "skipped"
    return "failed"


def _window(window: str) -> tuple[int, int, timezone] | None:
    """"22:00-08:00@+08:00" -> (start minute, end minute, tz); "" means any time."""
    if not window:
        return None
    spec, _, zone = window.partition("@")
    start, _, end = spec.partition("-")
    sign = -1 if zone.startswith("-") else 1
    hours, _, minutes = zone.lstrip("+-").partition(":")
    tz = timezone(sign * timedelta(hours=int(hours or 0), minutes=int(minutes or 0)))

    def minute(text: str) -> int:
        h, _, m = text.strip().partition(":")
        return int(h) * 60 + int(m or 0)

    return minute(start), minute(end), tz


def in_window(window: str, now: float) -> bool:
    parsed = _window(window)
    if parsed is None:
        return True
    start, end, tz = parsed
    moment = datetime.fromtimestamp(now, tz)
    minute = moment.hour * 60 + moment.minute
    return start <= minute < end if start < end else (minute >= start or minute < end)


def next_window_start(window: str, now: float) -> float:
    """When the window opens next (now if it is open)."""
    parsed = _window(window)
    if parsed is None or in_window(window, now):
        return now
    start, _end, tz = parsed
    moment = datetime.fromtimestamp(now, tz)
    opening = moment.replace(hour=start // 60, minute=start % 60, second=0, microsecond=0)
    if opening <= moment:
        opening += timedelta(days=1)
    return opening.timestamp()


# ---------------------------------------------------------------------------------------------- store
class BulkStore:
    def __init__(self, db: Path) -> None:
        db.parent.mkdir(parents=True, exist_ok=True)
        self.db = db
        self._lock = threading.Lock()
        with self._lock, self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS videos (
                  id TEXT PRIMARY KEY, position INTEGER, title TEXT, duration REAL, url TEXT,
                  status TEXT DEFAULT 'pending', bytes INTEGER DEFAULT 0, speed REAL, path TEXT DEFAULT '',
                  location TEXT DEFAULT '', attempts INTEGER DEFAULT 0, blocks INTEGER DEFAULT 0,
                  error TEXT DEFAULT '', started_at REAL, finished_at REAL);
                CREATE INDEX IF NOT EXISTS videos_status ON videos(status, position);
                CREATE TABLE IF NOT EXISTS events (at REAL, kind TEXT, detail TEXT);
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
                """
            )
            have = {row["name"] for row in conn.execute("PRAGMA table_info(videos)")}
            for name, spec in LATER_COLUMNS:
                if name not in have:
                    conn.execute(f"ALTER TABLE videos ADD COLUMN {name} {spec}")
            # Transcripts made before sources were recorded came from the job's ASR model.
            conn.execute("UPDATE videos SET transcript_source='fun-asr' WHERE asr_status='done' AND transcript_source=''")

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def execute(self, sql: str, args: tuple = ()) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(sql, args)

    def query(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._conn() as conn:
            return list(conn.execute(sql, args))

    def set_meta(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value, ensure_ascii=False)),
        )

    def meta(self, key: str, default: Any = None) -> Any:
        rows = self.query("SELECT value FROM meta WHERE key=?", (key,))
        return json.loads(rows[0]["value"]) if rows else default

    def all_meta(self) -> dict[str, Any]:
        return {row["key"]: json.loads(row["value"]) for row in self.query("SELECT key, value FROM meta")}

    def event(self, kind: str, detail: str = "", at: float | None = None) -> None:
        logger.info("bulk %s: %s", kind, detail)
        self.execute("INSERT INTO events(at, kind, detail) VALUES(?, ?, ?)", (at or time.time(), kind, detail[:500]))

    def add_videos(self, entries: list[dict[str, Any]]) -> int:
        """Insert videos not seen before; known ones keep their state."""
        added = 0
        with self._lock, self._conn() as conn:
            for position, entry in enumerate(entries):
                vid = str(entry.get("id") or "").strip()
                if not vid:
                    continue
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO videos(id, position, title, duration, url) VALUES(?, ?, ?, ?, ?)",
                    (vid, position, str(entry.get("title") or ""), _float(entry.get("duration")),
                     entry.get("url") or f"https://www.youtube.com/watch?v={vid}"),
                )
                added += cursor.rowcount
        return added

    def work_left(self, limits: "Limits | None" = None) -> tuple[int, int, int]:
        """(downloads, transcriptions, summaries) a runner would still do; used-up retries do not count."""
        limits = limits or Limits()
        downloads = self.query(
            "SELECT COUNT(*) n FROM videos WHERE status IN ('pending','downloading') OR "
            "(status='failed' AND attempts < ?)", (limits.max_attempts,))[0]["n"]
        if self.meta("youtube_extras", False):
            downloads += self.query(
                "SELECT COUNT(*) n FROM videos WHERE status='done' AND (captions_status IN ('pending','fetching') OR "
                "(captions_status='failed' AND captions_attempts < ?))", (limits.max_attempts,))[0]["n"]
        asr = 0
        if self.meta("asr_model", ""):
            asr = self.query(
                "SELECT COUNT(*) n FROM videos WHERE status='done' AND (asr_status IN ('pending','running') OR "
                "(asr_status='failed' AND asr_attempts < ?))", (limits.asr_max_attempts,))[0]["n"]
        summaries = 0
        if self.meta("summary_model", ""):
            summaries = self.query(
                "SELECT COUNT(*) n FROM videos WHERE status IN ('pending','downloading','done') AND "
                "(summary_status IN ('', 'queued', 'running') OR (summary_status='failed' AND summary_attempts < ?)) "
                "AND NOT (status='done' AND asr_status='failed' AND asr_attempts >= ?)",
                (limits.summary_max_attempts, limits.asr_max_attempts))[0]["n"]
        return downloads, asr, summaries

    def video(self, video_id: str) -> dict[str, Any] | None:
        rows = self.query("SELECT * FROM videos WHERE id=?", (video_id,))
        return dict(rows[0]) if rows else None

    def rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.query(
            "SELECT id, position, title, duration, url, status, bytes, speed, location, attempts, error, "
            "started_at, finished_at, asr_status, asr_attempts, asr_error, asr_started_at, asr_finished_at, "
            "segments, speakers, language, transcript_source, captions_status, captions_kind, captions_error, "
            "chapters, task_id, summary_status, summary_error FROM videos ORDER BY position")]

    def events(self, limit: int = 40) -> list[dict[str, Any]]:
        return [dict(row) for row in self.query(
            "SELECT at, kind, detail FROM events ORDER BY at DESC, rowid DESC LIMIT ?", (limit,))]

    def summary(self, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        meta = self.all_meta()
        counts = {row["status"]: row["n"] for row in self.query("SELECT status, COUNT(*) n FROM videos GROUP BY status")}
        asr_counts = {row["asr_status"]: row["n"] for row in self.query(
            "SELECT asr_status, COUNT(*) n FROM videos WHERE status='done' GROUP BY asr_status")}
        sources = {row["transcript_source"]: row["n"] for row in self.query(
            "SELECT transcript_source, COUNT(*) n FROM videos WHERE asr_status='done' GROUP BY transcript_source")}
        captions = {row["captions_status"]: row["n"] for row in self.query(
            "SELECT captions_status, COUNT(*) n FROM videos WHERE status='done' GROUP BY captions_status")}
        with_chapters = self.query("SELECT COUNT(*) n FROM videos WHERE chapters > 0")[0]["n"]
        summary_counts = {row["summary_status"] or "pending": row["n"] for row in self.query(
            "SELECT summary_status, COUNT(*) n FROM videos WHERE asr_status='done' GROUP BY summary_status")}
        totals = self.query(
            "SELECT COUNT(*) n, COALESCE(SUM(duration),0) dur, COALESCE(SUM(CASE WHEN status='done' THEN bytes END),0) "
            "bytes, COALESCE(SUM(CASE WHEN status='done' THEN duration END),0) done_dur, "
            "COALESCE(SUM(CASE WHEN status='done' AND location='nas' THEN 1 END),0) on_nas, "
            "COALESCE(SUM(CASE WHEN asr_status='done' AND transcript_source!='youtube' "
            "THEN COALESCE(asr_seconds, duration) END),0) asr_secs "
            "FROM videos"
        )[0]
        speeds = [row["speed"] for row in self.query(
            "SELECT speed FROM videos WHERE status='done' AND speed IS NOT NULL AND bytes >= 5000000 "
            "ORDER BY finished_at DESC LIMIT 10")]
        heartbeat = float(meta.get("heartbeat") or 0)
        alive = now - heartbeat < 180
        state = meta.get("state", "idle")
        if not alive and state in ACTIVE_STATES:
            state = "stopped"
        done_bytes = int(totals["bytes"])
        done_dur = float(totals["done_dur"])
        staging = meta.get("staging", "")
        asr_model = meta.get("asr_model", "")
        price = ASR_USD_PER_SEC.get(asr_model, 0.0)
        asr_pause_until = float(meta.get("asr_pause_until") or 0)
        return {
            "name": meta.get("name", self.db.parent.name),
            "title": meta.get("title", "") or meta.get("name", self.db.parent.name),
            "source": meta.get("source", ""),
            "state": state,
            "state_detail": meta.get("state_detail", ""),
            "alive": alive,
            "control": meta.get("control", ""),
            "pause_until": float(meta.get("pause_until") or 0),
            "pause_reason": meta.get("pause_reason", ""),
            "last_ok": float(meta.get("last_ok") or 0),
            "last_probe": meta.get("last_probe") or {},
            "heartbeat": heartbeat,
            "counts": counts,
            "total": int(totals["n"]),
            "hours": round(float(totals["dur"]) / 3600, 1),
            "done_hours": round(done_dur / 3600, 1),
            "gigabytes": round(done_bytes / 1e9, 2),
            "on_nas": int(totals["on_nas"]),
            "estimated_total_gb": round(done_bytes / done_dur * float(totals["dur"]) / 1e9, 0) if done_dur > 600 else None,
            "recent_speed_mb_s": round(statistics.median(speeds) / 1e6, 2) if speeds else None,
            "dest": meta.get("dest", ""),
            "dest_ready": dest_ready(meta.get("dest", "")) is not None,
            "staging": staging,
            "free_gb": round(_free_gb(Path(staging)), 1) if staging else None,
            "min_free_gb": meta.get("min_free_gb"),
            "asr_model": asr_model,
            "asr_counts": asr_counts,
            "asr_hours": round(float(totals["asr_secs"]) / 3600, 1),
            "asr_cost_usd": round(float(totals["asr_secs"]) * price, 2),
            "asr_estimated_total_usd": None if meta.get("youtube_extras") else (
                round(float(totals["dur"]) * price, 0) if price else None),
            "transcript_sources": sources,
            "captions_counts": captions if meta.get("youtube_extras") else {},
            "with_chapters": with_chapters,
            "youtube_extras": bool(meta.get("youtube_extras")),
            "captions_pause_until": float(meta.get("captions_pause_until") or 0)
            if float(meta.get("captions_pause_until") or 0) > now else 0.0,
            "captions_pause_reason": meta.get("captions_pause_reason", "")
            if float(meta.get("captions_pause_until") or 0) > now else "",
            "transcript_policy": meta.get("transcript_policy", "captions-first"),
            "summary_model": meta.get("summary_model", ""),
            "summary_counts": summary_counts if meta.get("summary_model") else {},
            "summary_window": meta.get("summary_window", DEFAULT_SUMMARY_WINDOW),
            "summary_in_window": in_window(meta.get("summary_window", DEFAULT_SUMMARY_WINDOW), now),
            "summary_next_window": next_window_start(meta.get("summary_window", DEFAULT_SUMMARY_WINDOW), now),
            "summary_cost_usd": round(summary_counts.get("done", 0)
                                      * SUMMARY_USD_PER_VIDEO.get(meta.get("summary_model", ""), 0.0), 2),
            "asr_pause_until": asr_pause_until if asr_pause_until > now else 0.0,
            "asr_pause_reason": meta.get("asr_pause_reason", "") if asr_pause_until > now else "",
        }

    def to_csv(self) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["#", "id", "title", "minutes", "download", "MB", "location", "attempts", "download error",
                         "transcript", "source", "segments", "speakers", "language", "asr attempts", "asr error",
                         "captions", "chapters", "url"])
        for row in self.rows():
            writer.writerow([
                row["position"] + 1, row["id"], row["title"], round((row["duration"] or 0) / 60, 1), row["status"],
                round((row["bytes"] or 0) / 1e6, 1), row["location"], row["attempts"], row["error"],
                row["asr_status"] if row["status"] == "done" else "", row["transcript_source"] or "",
                row["segments"] or "", row["speakers"] or "", row["language"] or "", row["asr_attempts"] or 0,
                row["asr_error"], row["captions_status"] if row["status"] == "done" else "", row["chapters"] or "",
                row["url"],
            ])
        return buffer.getvalue()


def bulk_root(config: AppConfig) -> Path:
    return config.paths.bulk


def job_dir(config: AppConfig, name: str) -> Path:
    if not name or Path(name).name != name or name.startswith("."):
        raise ValueError(f"Invalid bulk job name: {name!r}")
    return bulk_root(config) / name


def open_job(config: AppConfig, name: str, create: bool = False) -> BulkStore:
    db = job_dir(config, name) / DB_NAME
    if not create and not db.exists():
        raise FileNotFoundError(name)
    return BulkStore(db)


def list_jobs(config: AppConfig) -> list[str]:
    root = bulk_root(config)
    if not root.exists():
        return []
    return sorted(path.parent.name for path in root.glob(f"*/{DB_NAME}"))


def find_bulk_media(config: AppConfig, video_id: str) -> Path | None:
    """A finished bulk download of this video (local staging or the NAS), if one is reachable now."""
    if not video_id:
        return None
    for name in list_jobs(config):
        try:
            with sqlite3.connect(job_dir(config, name) / DB_NAME, timeout=5) as conn:
                row = conn.execute("SELECT path FROM videos WHERE id=? AND status='done'", (video_id,)).fetchone()
        except sqlite3.Error:
            continue
        if row and row[0]:
            path = Path(row[0])
            try:
                if path.is_file() and path.stat().st_size > 0:
                    return path
            except OSError:
                continue
    return None


def transcript_text(config: AppConfig, store: BulkStore, video_id: str) -> str | None:
    """A bulk video's transcript as plain text (speaker turns with timestamps), or None."""
    from media_pipeline.artifacts import ArtifactStore
    from media_pipeline.models import format_timestamp

    video = store.video(video_id)
    model = store.meta("asr_model", "")
    if video is None or not model:
        return None
    if not (config.paths.artifacts / video_id / "asr" / f"{model}.json").is_file():
        return None
    transcript = ArtifactStore(config.paths.artifacts, video_id).load_transcript(model)
    if transcript is None:
        return None
    lines = [video["title"] or video_id, video["url"], ""]
    speaker, block, start = None, [], 0.0
    for segment in transcript.segments:
        text = segment.text.strip()
        if not text:
            continue
        if segment.speaker != speaker and block:
            lines += [f"[{format_timestamp(start)}] {speaker or ''}".rstrip(), " ".join(block), ""]
            block = []
        if not block:
            speaker, start = segment.speaker, segment.start
        block.append(text)
    if block:
        lines += [f"[{format_timestamp(start)}] {speaker or ''}".rstrip(), " ".join(block), ""]
    return "\n".join(lines)


def read_list(source: str) -> list[dict[str, Any]]:
    """Videos from a file (``id|seconds|title`` lines or one URL/id per line) or a channel/playlist URL."""
    path = Path(source).expanduser()
    if path.is_file():
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|", 2)
            vid = parts[0].strip()
            if "youtube.com/" in vid or "youtu.be/" in vid:
                from media_pipeline.media import parse_video_ref

                vid = parse_video_ref(vid)[1]
            entries.append({
                "id": vid,
                "duration": _float(parts[1]) if len(parts) > 1 else None,
                "title": parts[2].strip() if len(parts) > 2 else "",
            })
        return entries
    return list_channel(source)


def list_channel(url: str) -> list[dict[str, Any]]:
    from media_pipeline.media import _ydl

    with _ydl({"quiet": True, "no_warnings": True, "ignoreconfig": True, "extract_flat": "in_playlist",
               "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False) or {}
    entries = []
    for entry in info.get("entries") or []:
        if entry and entry.get("entries"):  # a channel's tabs (Videos, Shorts, ...)
            entries.extend(item for item in entry["entries"] if item)
        elif entry:
            entries.append(entry)
    return [{"id": e.get("id"), "title": e.get("title") or "", "duration": e.get("duration")}
            for e in entries if e.get("id")]


def dest_ready(raw: str) -> Path | None:
    """The destination folder if it can be written now; a /Volumes path needs its share mounted."""
    if not raw:
        return None
    dest = Path(raw).expanduser()
    parts = dest.parts
    if len(parts) >= 3 and parts[0] == "/" and parts[1] == "Volumes":
        if not os.path.ismount(os.path.join("/", parts[1], parts[2])):
            return None
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return dest if os.access(dest, os.W_OK) else None


# --------------------------------------------------------------------------------------------- runner
Downloader = Callable[[str, str, Path, Callable[[int], None]], tuple[Path, int, float]]
# (media file, video row, ASR model id) -> {"segments", "speakers", "language", "seconds"}
Transcriber = Callable[[Path, dict[str, Any], str], dict[str, Any]]


class BulkRunner:
    def __init__(
        self,
        config: AppConfig,
        store: BulkStore,
        *,
        limits: Limits | None = None,
        fmt: str = DEFAULT_FORMAT,
        downloader: Downloader | None = None,
        prober: Callable[[], None] | None = None,
        transcriber: Transcriber | None = None,
        extras: Callable[[str, str, bool], None] | None = None,
        submit_summary: Callable[[str, str], str] | None = None,
        task_status: Callable[[str], tuple[str, str]] | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] | None = None,
        free_gb: Callable[[Path], float] | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.limits = limits or Limits()
        self.fmt = fmt
        self.staging = store.db.parent / "media"
        self.download = downloader or self._yt_dlp_download
        self.probe = prober or self._yt_dlp_probe
        self.transcribe = transcriber or self._asr_transcribe
        self.fetch_extras = extras or self._yt_dlp_extras
        self.submit_summary = submit_summary or self._submit_to_server
        self.task_status = task_status or self._task_status_from_store
        self._asr_done = threading.Event()
        self._captions_done = threading.Event()
        self._caption_lock = threading.Lock()
        self._last_caption_at = 0.0
        self._captions_backoff = self.limits.captions_first_backoff
        self.clock = clock
        self._stop = threading.Event()
        self.sleep = sleep or (lambda seconds: self._stop.wait(max(0.0, seconds)))
        self.free_gb = free_gb or _free_gb
        self._backoff = self.limits.first_backoff
        self._fail_streak = 0
        self._asr_streak = 0
        self._claim_lock = threading.Lock()
        self._probe_lock = threading.Lock()
        self._move_lock = threading.Lock()
        self._asr_lock = threading.Lock()
        self._downloads_done = threading.Event()
        self._last_beat: dict[str, float] = {}
        self._last_idle_move = 0.0
        self._done = 0

    # ------------------------------------------------------------------ main loop
    def run(self, max_videos: int | None = None) -> str:
        self.staging.mkdir(parents=True, exist_ok=True)
        self.store.set_meta("staging", str(self.staging))
        self.store.set_meta("min_free_gb", self.limits.min_free_gb)
        self.store.set_meta("pid", os.getpid())
        self.store.execute("UPDATE videos SET status='pending' WHERE status='downloading'")
        self.store.execute("UPDATE videos SET asr_status='pending' WHERE asr_status='running'")
        self.store.execute("UPDATE videos SET captions_status='pending' WHERE captions_status='fetching'")
        self._beat("download", force=True)
        self._beat("asr", force=True)
        asr_model = self.store.meta("asr_model", "")
        self.store.event(
            "start",
            f"downloads={self.limits.workers} asr={asr_model or 'off'}x{self.limits.asr_workers if asr_model else 0} "
            f"dest={self.store.meta('dest', '') or '(Mac)'}",
            at=self.clock(),
        )
        downloads = [threading.Thread(target=self._worker, args=(max_videos,), daemon=True)
                     for _ in range(max(1, self.limits.workers))]
        transcribers = [threading.Thread(target=self._asr_worker, args=(asr_model,), daemon=True)
                        for _ in range(self.limits.asr_workers if asr_model else 0)]
        summarizers = [threading.Thread(target=self._summary_worker, daemon=True)] \
            if self.store.meta("summary_model", "") else []
        captioners = [threading.Thread(target=self._captions_worker, daemon=True)] if self._extras_on() else []
        for thread in downloads + captioners + transcribers + summarizers:
            thread.start()
        for thread in downloads:
            thread.join()
        self._downloads_done.set()
        for thread in captioners:
            thread.join()
        self._captions_done.set()
        for thread in transcribers:
            thread.join()
        self._asr_done.set()
        for thread in summarizers:
            thread.join()
        self._move_staged(force=True)
        left = self.store.query(
            "SELECT COUNT(*) n FROM videos WHERE status='pending' OR (status='failed' AND attempts < ?)",
            (self.limits.max_attempts,))[0]["n"]
        asr_left = self._asr_left() if asr_model else 0
        summary_left = self.store.work_left(self.limits)[2]
        state = "finished" if not left and not asr_left and not summary_left else "stopped"
        detail = "" if state == "finished" else f"还剩 {left} 个待下载、{asr_left} 个待转写、{summary_left} 个待总结"
        self._set_state(state, detail)
        self.store.event("stop", f"left={left} asr_left={asr_left} summary_left={summary_left}", at=self.clock())
        return state

    def stop(self) -> None:
        self._stop.set()

    def _worker(self, max_videos: int | None) -> None:
        while not self._stop.is_set():
            if max_videos is not None and self._done >= max_videos:
                return
            if not self._wait_until_clear():
                return
            row = self._claim()
            if row is None:
                return
            self._one(row)
            self._done += 1
            self._move_staged()
            if self._stop.is_set() or (max_videos is not None and self._done >= max_videos):
                return
            self.sleep(random.uniform(*self._pause_range()))

    def _pause_range(self) -> tuple[float, float]:
        """Seconds between downloads: the job's ``pause_between`` (changeable while it runs) or the default."""
        value = self.store.meta("pause_between")
        if isinstance(value, list) and len(value) == 2:
            low, high = sorted(float(item) for item in value)
            return low, high
        return self.limits.pause_between

    def _claim(self) -> sqlite3.Row | None:
        with self._claim_lock:
            rows = self.store.query(
                "SELECT * FROM videos WHERE status='pending' OR (status='failed' AND attempts < ?) "
                "ORDER BY status='failed', position LIMIT 1",
                (self.limits.max_attempts,),
            )
            if not rows:
                return None
            self.store.execute(
                "UPDATE videos SET status='downloading', started_at=?, attempts=attempts+1, bytes=0 WHERE id=?",
                (self.clock(), rows[0]["id"]),
            )
            return rows[0]

    def _one(self, row: sqlite3.Row) -> None:
        vid = row["id"]

        def progress(downloaded: int) -> None:
            self.store.execute("UPDATE videos SET bytes=? WHERE id=?", (int(downloaded), vid))
            self._beat("download")

        try:
            path, size, seconds = self.download(row["url"], vid, self.staging, progress)
        except Exception as exc:  # noqa: BLE001 - every failure is classified and recorded
            self._failed(row, _short(exc))
            return
        speed = size / seconds if seconds > 0 else None
        self.store.execute(
            "UPDATE videos SET status='done', bytes=?, speed=?, path=?, location='mac', error='', finished_at=? "
            "WHERE id=?",
            (size, speed, str(path), self.clock(), vid),
        )
        self.store.set_meta("last_ok", self.clock())
        self._backoff = self.limits.first_backoff
        self._fail_streak = 0
        logger.info("done %s (#%s) %.0f MB at %.1f MB/s", vid, row["position"] + 1, size / 1e6, (speed or 0) / 1e6)
        self._check_speed()

    def _failed(self, row: sqlite3.Row, message: str) -> None:
        kind = classify(message)
        vid = row["id"]
        logger.warning("%s %s (#%s): %s", kind, vid, row["position"] + 1, message[:300])
        if kind == "skipped":
            self.store.execute("UPDATE videos SET status='skipped', error=? WHERE id=?", (message, vid))
            return
        if kind == "blocked":
            blocks = int(row["blocks"] or 0) + 1
            if blocks >= self.limits.max_blocks_per_video:
                # YouTube keeps answering "not available" for this one video while others download.
                self.store.execute("UPDATE videos SET status='failed', blocks=?, attempts=?, error=? WHERE id=?",
                                   (blocks, self.limits.max_attempts, message, vid))
            else:
                self.store.execute(
                    "UPDATE videos SET status='pending', blocks=?, attempts=attempts-1, error=? WHERE id=?",
                    (blocks, message, vid))
            self._limit("YouTube 限制：" + message[:160])
            return
        self.store.execute("UPDATE videos SET status='failed', error=? WHERE id=?", (message, vid))
        self._fail_streak += 1
        if self._fail_streak >= self.limits.fail_streak:
            self._fail_streak = 0
            self._limit(f"连续 {self.limits.fail_streak} 个视频下载失败（最近一个：{message[:120]}）")

    # ------------------------------------------------------------------ YouTube extras
    def _extras_on(self) -> bool:
        return bool(self.store.meta("youtube_extras", False))

    def _captions_worker(self) -> None:
        while not self._stop.is_set():
            self._beat("captions")
            if self.store.meta("control", "") == "pause":
                self.sleep(15)
                continue
            until = max(float(self.store.meta("captions_pause_until", 0) or 0),
                        float(self.store.meta("pause_until", 0) or 0))  # a YouTube limit stops captions too
            if until > self.clock():
                self.sleep(min(60.0, until - self.clock()))
                continue
            wait = self._caption_wait(self.limits.backfill_min_interval)
            if wait > 0:
                self.sleep(wait)
                continue
            row = self._claim_extras()
            if row is None:
                if self._downloads_done.is_set() and not self._extras_left():
                    return
                self.sleep(30)
                continue
            self._take_caption_slot(force=True)
            self._extras_one(row)

    def _caption_wait(self, interval: float | None = None) -> float:
        with self._caption_lock:
            gap = self.limits.captions_min_interval if interval is None else interval
            return max(0.0, self._last_caption_at + gap - self.clock())

    def _take_caption_slot(self, force: bool = False) -> bool:
        """May a captions request go out now? (spaced out, and never during a captions cooldown)"""
        with self._caption_lock:
            if max(float(self.store.meta("captions_pause_until", 0) or 0),
                   float(self.store.meta("pause_until", 0) or 0)) > self.clock():
                return False
            if not force and self.clock() - self._last_caption_at < self.limits.captions_min_interval:
                return False
            self._last_caption_at = self.clock()
            return True

    def _captions_cooldown(self, message: str) -> None:
        with self._probe_lock:
            if float(self.store.meta("captions_pause_until", 0) or 0) > self.clock():
                return
            wait = self._captions_backoff
            self._captions_backoff = min(self._captions_backoff * 2, self.limits.captions_max_backoff)
            self.store.set_meta("captions_pause_until", self.clock() + wait)
            reason = "YouTube 字幕接口限流：" + message[:120]
            self.store.set_meta("captions_pause_reason", reason)
            self.store.event("captions-pause", f"{reason}（字幕暂停 {round(wait / 60)} 分钟，视频照常下载）",
                             at=self.clock())

    def _claim_extras(self) -> sqlite3.Row | None:
        """A downloaded video whose captions/metadata were never fetched (backfill) or are due a retry.
        Videos still without a transcript come first: their captions save an ASR run."""
        if not self._extras_on():
            return None
        with self._claim_lock:
            rows = self.store.query(
                "SELECT * FROM videos WHERE status='done' AND (captions_status='pending' OR "
                "(captions_status='failed' AND captions_attempts < ? AND COALESCE(captions_at, 0) <= ?)) "
                "ORDER BY asr_status='done', captions_status='failed', position LIMIT 1",
                (self.limits.max_attempts, self.clock() - 600),
            )
            if not rows:
                return None
            self.store.execute("UPDATE videos SET captions_status='fetching' WHERE id=?", (rows[0]["id"],))
            return rows[0]

    def _extras_left(self) -> int:
        if not self._extras_on():
            return 0
        return self.store.query(
            "SELECT COUNT(*) n FROM videos WHERE status='done' AND (captions_status IN ('pending','fetching') OR "
            "(captions_status='failed' AND captions_attempts < ?))",
            (self.limits.max_attempts,))[0]["n"]

    def _extras_one(self, row: sqlite3.Row) -> None:
        # A video that already has a transcript only needs its metadata and chapters (no captions request).
        captions = row["asr_status"] != "done"
        try:
            self.fetch_extras(row["url"], row["id"], captions)
        except Exception as exc:  # noqa: BLE001 - recorded; captions are retried, then Fun-ASR takes over
            self._extras_failed(row["id"], _short(exc))

    def _extras_failed(self, video_id: str, message: str) -> None:
        kind = classify(message)
        logger.warning("captions %s %s: %s", kind, video_id, message[:300])
        if kind == "skipped":
            self.store.execute("UPDATE videos SET captions_status='none', captions_error=? WHERE id=?", (message, video_id))
            return
        if kind == "blocked":  # YouTube is limiting us, not this video: try it again later
            self.store.execute("UPDATE videos SET captions_status='pending', captions_error=? WHERE id=?",
                               (message, video_id))
            text = message.lower()
            if "not a bot" in text or "sign in to confirm" in text:
                self._limit("YouTube 限制（补抓元数据时）：" + message[:160])  # IP-wide: downloads stop too
            else:
                self._captions_cooldown(message)  # only the captions endpoint (HTTP 429)
            return
        self.store.execute(
            "UPDATE videos SET captions_status='failed', captions_attempts=captions_attempts+1, captions_error=?, "
            "captions_at=? WHERE id=?", (message, self.clock(), video_id))

    def _yt_dlp_extras(self, url: str, video_id: str, captions: bool = True) -> None:
        from media_pipeline.media import _ydl

        opts = self._guest_opts()
        opts["skip_download"] = True
        with _ydl(opts) as ydl:
            info = ydl.extract_info(url, download=False) or {}
            self._save_extras(ydl, info, video_id, captions=captions)

    def _save_extras(self, ydl: Any, info: dict[str, Any], video_id: str, captions: bool = True) -> None:
        """Metadata, chapters and (unless ``captions`` is False) captions from an extraction, saved where
        the pipeline reads them. Without captions the video stays 'pending' for the captions worker,
        or becomes 'skipped' if it already has a transcript."""
        from media_pipeline import youtube_captions as yc
        from media_pipeline.artifacts import ArtifactStore
        from media_pipeline.media import metadata_from_info

        artifacts = ArtifactStore(self.config.paths.artifacts, video_id)
        chapters = yc.chapters(info)
        transcript, kind, error = None, "", ""
        if captions:
            try:
                transcript, kind = yc.fetch_captions(ydl, info)
            except Exception as exc:  # noqa: BLE001
                error = _short(exc)
        if not artifacts.metadata_path.exists():
            model = yc.CAPTIONS_MODEL if transcript else (self.store.meta("asr_model", "") or self.config.asr.default)
            artifacts.save_metadata(metadata_from_info(
                info, f"https://www.youtube.com/watch?v={video_id}", model, fallback_id=video_id, platform="YouTube"))
        artifacts.write_json(artifacts.root / "youtube.json", {
            "chapters": chapters,
            "captions": {"kind": kind, "language": transcript.language if transcript else ""},
            "channel": info.get("channel") or info.get("uploader") or "",
            "upload_date": info.get("upload_date") or "",
            "view_count": info.get("view_count"),
            "like_count": info.get("like_count"),
            "tags": info.get("tags") or [],
        })
        self.store.execute("UPDATE videos SET chapters=? WHERE id=?", (len(chapters), video_id))
        if not captions:
            row = self.store.video(video_id) or {}
            if row.get("asr_status") == "done":
                self.store.execute("UPDATE videos SET captions_status='skipped', captions_at=? WHERE id=?",
                                   (self.clock(), video_id))
            elif row.get("captions_status") == "fetching":
                self.store.execute("UPDATE videos SET captions_status='pending' WHERE id=?", (video_id,))
            return
        if transcript is not None:
            self._captions_backoff = self.limits.captions_first_backoff
            artifacts.save_transcript(yc.CAPTIONS_MODEL, transcript)
            self.store.execute(
                "UPDATE videos SET captions_status='done', captions_kind=?, captions_error='', captions_at=? WHERE id=?",
                (kind, self.clock(), video_id))
            # A free transcript: this video's transcription stage is done (unless Fun-ASR already ran).
            self.store.execute(
                "UPDATE videos SET asr_status='done', transcript_source='youtube', segments=?, speakers=1, language=?, "
                "asr_error='', asr_finished_at=? WHERE id=? AND asr_status!='done'",
                (len(transcript.segments), transcript.language or "", self.clock(), video_id))
        elif error:
            self._extras_failed(video_id, error)
        else:
            self.store.execute("UPDATE videos SET captions_status='none', captions_at=? WHERE id=?",
                               (self.clock(), video_id))

    def _asr_gate(self) -> str:
        """With captions first, the ASR model only gets videos without usable captions, or whose captions are
        still missing ``captions_wait`` after the download (YouTube kept rate-limiting them)."""
        if self._extras_on() and self.store.meta("transcript_policy", "captions-first") == "captions-first":
            return (" AND (captions_status IN ('none','skipped') OR (captions_status='failed' AND captions_attempts >= %d)"
                    " OR (finished_at IS NOT NULL AND finished_at <= %f))" % (self.limits.max_attempts,
                                                              self.clock() - self.limits.captions_wait))
        return ""

    # ------------------------------------------------------------------ summaries
    def _summary_worker(self) -> None:
        while not self._stop.is_set():
            self._beat("summary")
            self._refresh_summaries()
            if self.store.meta("control", "") == "pause":
                self.sleep(15)
                continue
            inflight = self._summaries_inflight()
            window = self.store.meta("summary_window", DEFAULT_SUMMARY_WINDOW)
            if not in_window(window, self.clock()):
                if self._asr_done.is_set() and not inflight and not self.store.work_left(self.limits)[2]:
                    return
                self.sleep(60)
                continue
            if inflight >= self.limits.summary_inflight:
                self.sleep(15)
                continue
            row = self._claim_summary()
            if row is None:
                if self._asr_done.is_set() and not inflight and not self._summaries_waiting():
                    return
                self.sleep(30)
                continue
            try:
                task_id = self.submit_summary(self.store.db.parent.name, row["id"])
            except Exception as exc:  # noqa: BLE001 - e.g. the server is restarting; try again later
                self.store.execute(
                    "UPDATE videos SET summary_status='failed', summary_error=?, summary_at=? WHERE id=?",
                    (_short(exc), self.clock(), row["id"]))
                self.sleep(60)
                continue
            self.store.execute("UPDATE videos SET summary_status='queued', task_id=?, summary_error='' WHERE id=?",
                               (task_id, row["id"]))

    def _claim_summary(self) -> sqlite3.Row | None:
        with self._claim_lock:
            rows = self.store.query(
                "SELECT * FROM videos WHERE asr_status='done' AND (summary_status='' OR (summary_status='failed' "
                "AND summary_attempts < ? AND COALESCE(summary_at, 0) <= ?)) ORDER BY summary_status='failed', position "
                "LIMIT 1",
                (self.limits.summary_max_attempts, self.clock() - self.limits.summary_retry_after),
            )
            if not rows:
                return None
            self.store.execute(
                "UPDATE videos SET summary_status='queued', summary_attempts=summary_attempts+1, summary_at=? WHERE id=?",
                (self.clock(), rows[0]["id"]))
            return rows[0]

    def _summaries_inflight(self) -> int:
        return self.store.query(
            "SELECT COUNT(*) n FROM videos WHERE summary_status IN ('queued','running')")[0]["n"]

    def _summaries_waiting(self) -> int:
        """Transcribed videos not summarized yet, plus ones that may still be transcribed in this run."""
        ready = self.store.query(
            "SELECT COUNT(*) n FROM videos WHERE asr_status='done' AND (summary_status='' OR "
            "(summary_status='failed' AND summary_attempts < ?))", (self.limits.summary_max_attempts,))[0]["n"]
        return ready

    def _refresh_summaries(self) -> None:
        for row in self.store.query(
            "SELECT id, task_id FROM videos WHERE summary_status IN ('queued','running') AND task_id != ''"
        ):
            status, error = self.task_status(row["task_id"])
            if status in ("queued", "running"):
                if status == "running":
                    self.store.execute("UPDATE videos SET summary_status='running' WHERE id=? AND summary_status='queued'",
                                       (row["id"],))
                continue
            if status == "done":
                self.store.execute(
                    "UPDATE videos SET summary_status='done', summary_error='', summary_at=? WHERE id=?",
                    (self.clock(), row["id"]))
            else:
                self.store.execute(
                    "UPDATE videos SET summary_status='failed', summary_error=?, summary_at=? WHERE id=?",
                    (error or status, self.clock(), row["id"]))
                logger.warning("summary failed %s: %s", row["id"], (error or status)[:300])

    def _submit_to_server(self, job: str, video_id: str) -> str:
        import urllib.request

        url = f"http://127.0.0.1:{self.config.server.port}/v1/bulk/{job}/videos/{video_id}/process"
        request = urllib.request.Request(url, data=b"", method="POST")
        with urllib.request.urlopen(request, timeout=30) as response:
            return str(json.loads(response.read())["id"])

    def _task_status_from_store(self, task_id: str) -> tuple[str, str]:
        """("queued"|"running"|"done"|"failed", error) of a pipeline task, read from the task database."""
        from media_pipeline.artifacts import ArtifactStore
        from media_pipeline.store import TaskStore

        task = TaskStore(self.config.paths.db).get(task_id)
        if task is None:
            return "failed", "task disappeared"
        status = task.status.value
        if status == "queued":
            return "queued", ""
        if status == "failed":
            return "failed", f"{task.error_stage}: {task.error}"[:400]
        if status != "completed":
            return "running", ""
        summary = ArtifactStore(self.config.paths.artifacts, task.video_id).load_summary() or {}
        if summary.get("status") == "completed" and str(summary.get("markdown") or "").strip():
            return "done", ""
        return "failed", str(summary.get("error") or "no summary was written")[:400]

    # ------------------------------------------------------------------ transcription
    def _asr_worker(self, model: str) -> None:
        while not self._stop.is_set():
            self._beat("asr")
            if self.store.meta("control", "") == "pause":
                self.sleep(15)
                continue
            until = float(self.store.meta("asr_pause_until", 0) or 0)
            if until > self.clock():
                self.sleep(min(30.0, until - self.clock()))
                continue
            row = self._claim_asr()
            if row is None:
                if self._downloads_done.is_set() and not self._asr_left(waiting_only=True):
                    return
                self.sleep(15)
                continue
            self._asr_one(row, model)

    def _claim_asr(self) -> dict[str, Any] | None:
        with self._claim_lock:
            rows = self.store.query(
                "SELECT * FROM videos WHERE status='done' AND (asr_status='pending' OR "
                "(asr_status='failed' AND asr_attempts < ? AND COALESCE(asr_finished_at, 0) <= ?))"
                + self._asr_gate() + " ORDER BY asr_status='failed', position LIMIT 1",
                (self.limits.asr_max_attempts, self.clock() - self.limits.asr_retry_after),
            )
            if not rows:
                return None
            self.store.execute(
                "UPDATE videos SET asr_status='running', asr_started_at=?, asr_attempts=asr_attempts+1, asr_error='' "
                "WHERE id=?",
                (self.clock(), rows[0]["id"]),
            )
            return dict(rows[0])

    def _asr_left(self, waiting_only: bool = False) -> int:
        """Downloaded videos still to transcribe (``waiting_only``: not counting ones in progress)."""
        statuses = "('pending')" if waiting_only else "('pending','running')"
        # Once the captions worker is done, a video still waiting on captions will not get them in this run.
        gate = self._asr_gate() if self._captions_done.is_set() else ""
        return self.store.query(
            f"SELECT COUNT(*) n FROM videos WHERE status='done' AND (asr_status IN {statuses} OR "
            "(asr_status='failed' AND asr_attempts < ?))" + gate,
            (self.limits.asr_max_attempts,),
        )[0]["n"]

    def _asr_one(self, row: dict[str, Any], model: str) -> None:
        vid = row["id"]
        path = Path(row["path"] or "")
        if not path.is_file():  # e.g. moved to a NAS that is not mounted right now
            self.store.execute(
                "UPDATE videos SET asr_status='pending', asr_attempts=asr_attempts-1 WHERE id=?", (vid,))
            self.sleep(60)
            return
        try:
            stats = self.transcribe(path, row, model)
        except Exception as exc:  # noqa: BLE001 - recorded per video, retried later
            message = _short(exc)
            logger.warning("asr failed %s (#%s): %s", vid, row["position"] + 1, message[:300])
            self.store.execute("UPDATE videos SET asr_status='failed', asr_error=?, asr_finished_at=? WHERE id=?",
                               (message, self.clock(), vid))
            with self._asr_lock:
                self._asr_streak += 1
                if self._asr_streak >= self.limits.asr_fail_streak:
                    self._asr_streak = 0
                    resume = self.clock() + self.limits.asr_backoff
                    reason = f"连续 {self.limits.asr_fail_streak} 个视频转写失败（最近一个：{message[:120]}）"
                    self.store.set_meta("asr_pause_until", resume)
                    self.store.set_meta("asr_pause_reason", reason)
                    self.store.event("asr-pause", f"{reason}，{round(self.limits.asr_backoff / 60)} 分钟后继续",
                                     at=self.clock())
            return
        seconds = row.get("duration") or stats.get("seconds")
        self.store.execute(
            "UPDATE videos SET asr_status='done', asr_error='', asr_finished_at=?, asr_seconds=?, segments=?, "
            "speakers=?, language=?, transcript_source=? WHERE id=?",
            (self.clock(), seconds, stats.get("segments"), stats.get("speakers"), stats.get("language") or "",
             model, vid),
        )
        with self._asr_lock:
            self._asr_streak = 0
        self._beat("asr", force=True)
        logger.info("asr done %s (#%s) %s segments, %s speakers", vid, row["position"] + 1,
                    stats.get("segments"), stats.get("speakers"))

    def _asr_transcribe(self, path: Path, row: dict[str, Any], model: str) -> dict[str, Any]:
        """The pipeline's own ASR step, saved where the pipeline looks for it later."""
        from media_pipeline.artifacts import ArtifactStore
        from media_pipeline.asr.language import resolve_provider_language
        from media_pipeline.asr.registry import require_provider
        from media_pipeline.models import ASR_MODELS, ASROptions
        from media_pipeline.transcript import clean_transcript

        artifacts = ArtifactStore(self.config.paths.artifacts, row["id"])
        transcript = artifacts.load_transcript(model)
        if not (transcript and transcript.segments):
            provider = require_provider(model)
            bind = getattr(provider, "bind_config", None)
            if callable(bind):
                bind(self.config)
            language = resolve_provider_language(ASR_MODELS[model]["provider"], self.config.asr.language or "auto")
            context = "\n".join(part for part in (row.get("title"), self.store.meta("title", "")) if part)
            transcript = clean_transcript(provider.transcribe(path, ASROptions(context=context, language=language)))
            if not transcript.segments:
                raise RuntimeError("ASR returned no speech")
            artifacts.save_transcript(model, transcript)
        return {
            "segments": len(transcript.segments),
            "speakers": len({segment.speaker for segment in transcript.segments if segment.speaker}),
            "language": transcript.language or "",
            "seconds": max((segment.end for segment in transcript.segments), default=0.0),
        }

    # ------------------------------------------------------------ limits & pauses
    def _wait_until_clear(self) -> bool:
        while not self._stop.is_set():
            self._beat("download")
            if self.store.meta("control", "") == "pause":
                self._set_state("paused", "已手动暂停")
                self._idle_move()
                self.sleep(15)
                continue
            until = float(self.store.meta("pause_until", 0) or 0)
            if until > self.clock():
                self._set_state("limited", self.store.meta("pause_reason", ""))
                self._idle_move()  # a YouTube pause is a good time to copy finished files to the NAS
                self.sleep(min(30.0, until - self.clock()))
                continue
            if until:
                self._probe_after_pause()
                continue
            free = self.free_gb(self.staging)
            if free < self.limits.min_free_gb:
                self._move_staged(force=True)
                free = self.free_gb(self.staging)
            if free < self.limits.min_free_gb:
                self._set_state("paused", f"Mac 只剩 {free:.0f} GB；NAS 挂上后已下载的会自动移过去，然后继续")
                self.sleep(60)
                continue
            self._set_state("running", "")
            return True
        return False

    def _probe_after_pause(self) -> None:
        with self._probe_lock:
            until = float(self.store.meta("pause_until", 0) or 0)
            if not until or until > self.clock():
                return  # another worker already probed, or the pause was extended
            try:
                self.probe()
            except Exception as exc:  # noqa: BLE001
                self.store.set_meta("last_probe", {"at": self.clock(), "ok": False, "detail": _short(exc)[:200]})
                self.store.set_meta("pause_until", 0)
                self._limit_locked("恢复前的探测仍失败：" + _short(exc)[:160])
                return
            self.store.set_meta("last_probe", {"at": self.clock(), "ok": True, "detail": ""})
            self.store.set_meta("pause_until", 0)
            self.store.set_meta("pause_reason", "")
            self.store.set_meta("window_start", self.clock())
            self.store.event("resume", "探测成功，继续下载", at=self.clock())

    def _limit(self, reason: str) -> None:
        with self._probe_lock:
            self._limit_locked(reason)

    def _limit_locked(self, reason: str) -> None:
        if float(self.store.meta("pause_until", 0) or 0) > self.clock():
            return  # already paused (two workers hit the same limit)
        wait = self._backoff
        self._backoff = min(self._backoff * 2, self.limits.max_backoff)
        resume = self.clock() + wait
        self.store.set_meta("pause_until", resume)
        self.store.set_meta("pause_reason", reason)
        self._set_state("limited", reason)
        self.store.event("pause", f"{reason}（{round(wait / 60)} 分钟后探测）", at=self.clock())

    def _check_speed(self) -> None:
        since = float(self.store.meta("window_start", 0) or 0)
        speeds = [row["speed"] for row in self.store.query(
            "SELECT speed FROM videos WHERE status='done' AND speed IS NOT NULL AND bytes >= ? AND finished_at >= ? "
            "ORDER BY finished_at DESC LIMIT ?",
            (self.limits.slow_min_bytes, since, self.limits.slow_window))]
        if len(speeds) == self.limits.slow_window and statistics.median(speeds) < self.limits.slow_bytes_per_sec:
            self.store.set_meta("window_start", self.clock() + 1e-6)
            self._limit(f"最近 {len(speeds)} 个视频的下载速度只有 ~{statistics.median(speeds) / 1e3:.0f} KB/s，疑似被限速")

    def _set_state(self, state: str, detail: str) -> None:
        if self.store.meta("state") != state or self.store.meta("state_detail") != detail:
            self.store.set_meta("state", state)
            self.store.set_meta("state_detail", detail)

    def _beat(self, kind: str = "", force: bool = False) -> None:
        """Liveness for the page and the watchdog; downloads and ASR beat separately, so one stuck
        stage is visible even while the other keeps the job looking alive."""
        now = self.clock()
        if force or now - self._last_beat.get(kind, 0.0) >= 10:
            self._last_beat[kind] = now
            self.store.set_meta("heartbeat", now)
            if kind:
                self.store.set_meta(f"heartbeat_{kind}", now)

    # ---------------------------------------------------------------- storage
    def _idle_move(self) -> None:
        if self.clock() - self._last_idle_move >= 60:
            self._last_idle_move = self.clock()
            self._move_staged()

    def _move_staged(self, force: bool = False) -> None:
        """Copy finished files to the NAS. With ASR on, a file waits until it is transcribed (read locally,
        faster) unless ``force`` (the disk is low, or the run is ending); never while it is being read."""
        dest = dest_ready(self.store.meta("dest", ""))
        if dest is None:
            return
        waiting = not force and bool(self.store.meta("asr_model", ""))
        with self._move_lock:
            for row in self.store.query(
                "SELECT id, path, asr_status, asr_attempts FROM videos WHERE status='done' AND location='mac'"
            ):
                if row["asr_status"] == "running":
                    continue
                if waiting and (row["asr_status"] == "pending" or (
                        row["asr_status"] == "failed" and row["asr_attempts"] < self.limits.asr_max_attempts)):
                    continue
                source = Path(row["path"])
                if not source.is_file():
                    continue
                target = dest / source.name
                try:
                    _copy_verified(source, target)
                except OSError as exc:
                    self.store.event("move-failed", f"{source.name}: {exc}", at=self.clock())
                    return
                with self._claim_lock:  # an ASR worker may have claimed it while it was copying
                    current = self.store.query("SELECT asr_status FROM videos WHERE id=?", (row["id"],))
                    if current and current[0]["asr_status"] == "running":
                        continue
                    source.unlink()
                    self.store.execute("UPDATE videos SET path=?, location='nas' WHERE id=?",
                                       (str(target), row["id"]))

    # ---------------------------------------------------------------- yt-dlp
    def _guest_opts(self) -> dict[str, Any]:
        from media_pipeline.media import _base_ydl_opts

        guest = replace(self.config, download=replace(self.config.download, cookies_from_browser=""))
        opts = _base_ydl_opts(guest)
        opts.update({"socket_timeout": 30, "sleep_interval_requests": 1})
        return opts

    def _yt_dlp_download(self, url: str, video_id: str, target_dir: Path,
                         progress: Callable[[int], None]) -> tuple[Path, int, float]:
        from media_pipeline.media import _ydl, find_media_file

        target_dir.mkdir(parents=True, exist_ok=True)
        done = {"bytes": 0, "seconds": 0.0, "reported": 0.0}

        def hook(status: dict[str, Any]) -> None:
            if status.get("status") == "finished":
                done["bytes"] += int(status.get("total_bytes") or status.get("downloaded_bytes") or 0)
                done["seconds"] += float(status.get("elapsed") or 0)
            elif status.get("status") == "downloading" and time.monotonic() - done["reported"] > 3:
                done["reported"] = time.monotonic()
                progress(done["bytes"] + int(status.get("downloaded_bytes") or 0))

        opts = self._guest_opts()
        opts.update({
            "format": self.fmt,
            "merge_output_format": "mp4",
            "outtmpl": str(target_dir / "%(id)s.%(ext)s"),
            "noprogress": True,
            "continuedl": True,
            "progress_hooks": [hook],
        })
        with _ydl(opts) as ydl:
            info = ydl.extract_info(url, download=True) or {}
            if self._extras_on():
                try:
                    self._save_extras(ydl, info, video_id, captions=self._take_caption_slot())
                except Exception as exc:  # noqa: BLE001 - never fail a finished download over extras
                    self._extras_failed(video_id, _short(exc))
        path = find_media_file(target_dir, video_id)
        if path is None or path.suffix in (".part", ".ytdl"):
            raise RuntimeError("yt-dlp finished but the video file was not found")
        return path, path.stat().st_size, done["seconds"]

    def _yt_dlp_probe(self) -> None:
        from media_pipeline.media import _ydl

        rows = self.store.query("SELECT url FROM videos WHERE status='done' ORDER BY finished_at DESC LIMIT 1") or \
            self.store.query("SELECT url FROM videos ORDER BY position LIMIT 1")
        opts = self._guest_opts()
        opts["skip_download"] = True
        with _ydl(opts) as ydl:
            ydl.extract_info(rows[0]["url"], download=False)


def _copy_verified(source: Path, target: Path) -> None:
    size = source.stat().st_size
    if target.is_file() and target.stat().st_size == size:
        return  # copied before a crash, the local file just was not removed yet
    partial = target.with_name(target.name + ".copying")
    shutil.copyfile(source, partial)
    if partial.stat().st_size != size:
        partial.unlink(missing_ok=True)
        raise OSError(f"copy of {source.name} is incomplete")
    os.replace(partial, target)


def _free_gb(path: Path) -> float:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1e9


def _float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _short(exc: BaseException | str) -> str:
    text = str(exc).strip()
    return text.removeprefix("ERROR: ").strip()[:400]
