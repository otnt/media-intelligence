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

**Transcription stage** (when the job has an ``asr_model``, e.g. ``fun-asr``). Runs alongside the
downloads on finished files, a few at a time. Transcripts are saved exactly where the pipeline keeps
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
)


@dataclass
class Limits:
    workers: int = 1
    pause_between: tuple[float, float] = (8.0, 20.0)
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


def classify(error: str) -> str:
    """blocked (YouTube is limiting us), skipped (this video can never be downloaded) or failed."""
    text = (error or "").lower()
    if any(marker in text for marker in BLOCK_MARKERS):
        return "blocked"
    if any(marker in text for marker in SKIP_MARKERS):
        return "skipped"
    return "failed"


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

    def work_left(self, limits: "Limits | None" = None) -> tuple[int, int]:
        """(downloads, transcriptions) that a runner would still do; retries that are used up do not count."""
        limits = limits or Limits()
        downloads = self.query(
            "SELECT COUNT(*) n FROM videos WHERE status IN ('pending','downloading') OR "
            "(status='failed' AND attempts < ?)", (limits.max_attempts,))[0]["n"]
        if not self.meta("asr_model", ""):
            return downloads, 0
        asr = self.query(
            "SELECT COUNT(*) n FROM videos WHERE status='done' AND (asr_status IN ('pending','running') OR "
            "(asr_status='failed' AND asr_attempts < ?))", (limits.asr_max_attempts,))[0]["n"]
        return downloads, asr

    def video(self, video_id: str) -> dict[str, Any] | None:
        rows = self.query("SELECT * FROM videos WHERE id=?", (video_id,))
        return dict(rows[0]) if rows else None

    def rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.query(
            "SELECT id, position, title, duration, url, status, bytes, speed, location, attempts, error, "
            "started_at, finished_at, asr_status, asr_attempts, asr_error, asr_started_at, asr_finished_at, "
            "segments, speakers, language FROM videos ORDER BY position")]

    def events(self, limit: int = 40) -> list[dict[str, Any]]:
        return [dict(row) for row in self.query(
            "SELECT at, kind, detail FROM events ORDER BY at DESC, rowid DESC LIMIT ?", (limit,))]

    def summary(self, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        meta = self.all_meta()
        counts = {row["status"]: row["n"] for row in self.query("SELECT status, COUNT(*) n FROM videos GROUP BY status")}
        asr_counts = {row["asr_status"]: row["n"] for row in self.query(
            "SELECT asr_status, COUNT(*) n FROM videos WHERE status='done' GROUP BY asr_status")}
        totals = self.query(
            "SELECT COUNT(*) n, COALESCE(SUM(duration),0) dur, COALESCE(SUM(CASE WHEN status='done' THEN bytes END),0) "
            "bytes, COALESCE(SUM(CASE WHEN status='done' THEN duration END),0) done_dur, "
            "COALESCE(SUM(CASE WHEN status='done' AND location='nas' THEN 1 END),0) on_nas, "
            "COALESCE(SUM(CASE WHEN asr_status='done' THEN COALESCE(asr_seconds, duration) END),0) asr_secs "
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
            "asr_estimated_total_usd": round(float(totals["dur"]) * price, 0) if price else None,
            "asr_pause_until": asr_pause_until if asr_pause_until > now else 0.0,
            "asr_pause_reason": meta.get("asr_pause_reason", "") if asr_pause_until > now else "",
        }

    def to_csv(self) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["#", "id", "title", "minutes", "download", "MB", "location", "attempts", "download error",
                         "asr", "segments", "speakers", "language", "asr attempts", "asr error", "url"])
        for row in self.rows():
            writer.writerow([
                row["position"] + 1, row["id"], row["title"], round((row["duration"] or 0) / 60, 1), row["status"],
                round((row["bytes"] or 0) / 1e6, 1), row["location"], row["attempts"], row["error"],
                row["asr_status"] if row["status"] == "done" else "", row["segments"] or "", row["speakers"] or "",
                row["language"] or "", row["asr_attempts"] or 0, row["asr_error"], row["url"],
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
        self._done = 0

    # ------------------------------------------------------------------ main loop
    def run(self, max_videos: int | None = None) -> str:
        self.staging.mkdir(parents=True, exist_ok=True)
        self.store.set_meta("staging", str(self.staging))
        self.store.set_meta("min_free_gb", self.limits.min_free_gb)
        self.store.set_meta("pid", os.getpid())
        self.store.execute("UPDATE videos SET status='pending' WHERE status='downloading'")
        self.store.execute("UPDATE videos SET asr_status='pending' WHERE asr_status='running'")
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
        for thread in downloads + transcribers:
            thread.start()
        for thread in downloads:
            thread.join()
        self._downloads_done.set()
        for thread in transcribers:
            thread.join()
        self._move_staged(force=True)
        left = self.store.query(
            "SELECT COUNT(*) n FROM videos WHERE status='pending' OR (status='failed' AND attempts < ?)",
            (self.limits.max_attempts,))[0]["n"]
        asr_left = self._asr_left() if asr_model else 0
        state = "finished" if not left and not asr_left else "stopped"
        detail = "" if state == "finished" else f"还剩 {left} 个待下载、{asr_left} 个待转写"
        self._set_state(state, detail)
        self.store.event("stop", f"left={left} asr_left={asr_left}", at=self.clock())
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
            self.sleep(random.uniform(*self.limits.pause_between))

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
                "(asr_status='failed' AND asr_attempts < ? AND COALESCE(asr_finished_at, 0) <= ?)) "
                "ORDER BY asr_status='failed', position LIMIT 1",
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
        return self.store.query(
            f"SELECT COUNT(*) n FROM videos WHERE status='done' AND (asr_status IN {statuses} OR "
            "(asr_status='failed' AND asr_attempts < ?))",
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
            "speakers=?, language=? WHERE id=?",
            (self.clock(), seconds, stats.get("segments"), stats.get("speakers"), stats.get("language") or "", vid),
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
                self.sleep(15)
                continue
            until = float(self.store.meta("pause_until", 0) or 0)
            if until > self.clock():
                self._set_state("limited", self.store.meta("pause_reason", ""))
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
            ydl.download([url])
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
