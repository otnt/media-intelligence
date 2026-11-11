"""Disk retention for downloaded media.

Once a video is processed, the note (transcript, speakers, summary) is the product. After a
successful run:

* the 16 kHz WAV (derived, ~115 MB/hour) is replaced by a mono 48 kbps AAC copy (~22 MB/hour) that
  rebuilds the WAV in about a second; that copy goes after ``storage.keep_audio_days``;
* the original download follows ``storage.mode``: ``archive`` moves it into iCloud Drive, where it is
  kept and, once uploaded, evicted from this Mac (``brctl evict``; eviction is refused until iCloud has
  the file, so nothing is lost); ``keep`` leaves it; ``delete`` removes it.

Media comes back on demand: the pipeline only touches media when a stage needs it, and fetches an
archived original back from iCloud (``brctl download``) before reading it, because a process may be
refused automatic download of a cloud-only file (EDEADLK).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from media_pipeline.config import AppConfig
from media_pipeline.models import Task, TaskStatus

logger = logging.getLogger(__name__)

TERMINAL = (TaskStatus.completed, TaskStatus.failed)
SF_DATALESS = 0x40000000
VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv", ".mov", ".flv", ".m4a", ".mp3", ".part", ".ytdl"}


@dataclass
class CompactResult:
    video_id: str
    removed: list[Path] = field(default_factory=list)
    archived: list[Path] = field(default_factory=list)
    freed: int = 0
    audio: Path | None = None

    def remove(self, path: Path, dry_run: bool) -> None:
        if not path.is_file():
            return
        self.freed += path.stat().st_size
        self.removed.append(path)
        if not dry_run:
            path.unlink(missing_ok=True)


@dataclass
class CleanupReport:
    compacted: list[CompactResult] = field(default_factory=list)
    skipped_active: list[str] = field(default_factory=list)
    orphans: list[Path] = field(default_factory=list)
    evicted: list[Path] = field(default_factory=list)
    evicted_bytes: int = 0
    awaiting_upload: list[Path] = field(default_factory=list)
    dry_run: bool = False

    @property
    def freed(self) -> int:
        return sum(item.freed for item in self.compacted)

    def summary(self) -> str:
        verb = "would free" if self.dry_run else "freed"
        lines = [f"{verb} {self.freed / 1e6:.0f} MB from {sum(1 for c in self.compacted if c.removed)} video(s)"]
        for item in self.compacted:
            if item.removed or item.archived:
                names = ", ".join(path.name for path in item.removed)
                moved = f"; moved to iCloud: {', '.join(path.name for path in item.archived)}" if item.archived else ""
                lines.append(f"  {item.video_id}: {item.freed / 1e6:.0f} MB deleted ({names or '-'}){moved}")
        if self.evicted:
            lines.append(f"  evicted from this Mac (kept in iCloud): {len(self.evicted)} file(s), {self.evicted_bytes / 1e6:.0f} MB")
        if self.awaiting_upload:
            lines.append(f"  waiting for iCloud upload before evicting: {len(self.awaiting_upload)} file(s)")
        if self.skipped_active:
            lines.append(f"  still processing, left alone: {', '.join(self.skipped_active)}")
        if self.orphans:
            size = sum(path.stat().st_size for path in self.orphans if path.exists())
            lines.append(f"  {len(self.orphans)} file(s) with no task ({size / 1e6:.0f} MB), not touched: "
                         + ", ".join(path.name for path in self.orphans[:6]))
        return "\n".join(lines)


def compact_audio_path(config: AppConfig, video_id: str) -> Path:
    return config.paths.audio / f"{video_id}.m4a"


def wav_path(config: AppConfig, video_id: str) -> Path:
    return config.paths.audio / f"{video_id}.wav"


def video_files(config: AppConfig, video_id: str) -> list[Path]:
    directory = config.paths.videos
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.glob(f"{video_id}.*")
        if path.is_file() and (path.suffix.lower() in VIDEO_SUFFIXES or path.name.endswith(".part"))
    )


def compact_video_media(config: AppConfig, video_id: str, *, dry_run: bool = False) -> CompactResult:
    """Leave only the compact audio copy for one processed video (or nothing if keep_audio_days == 0)."""
    result = CompactResult(video_id)
    wav = wav_path(config, video_id)
    m4a = compact_audio_path(config, video_id)
    videos = video_files(config, video_id)
    keep_audio = int(config.storage.keep_audio_days) != 0
    if keep_audio and not _nonempty(m4a):
        source = wav if _nonempty(wav) else (videos[0] if videos else None)
        if source is not None and not dry_run and not _encode_compact(source, m4a):
            # Could not make the small copy: keep whatever audio we have, but the video can still go
            # if a WAV is left to re-transcribe from.
            logger.warning("Keeping %s: compact audio encode failed", source.name)
            if source == wav:
                _dispose_originals(config, videos, result, dry_run)
                result.audio = wav
            return result
    _dispose_originals(config, videos, result, dry_run)
    result.remove(wav, dry_run)
    if keep_audio:
        result.audio = m4a if (_nonempty(m4a) or dry_run) else None
    else:
        result.remove(m4a, dry_run)
    if result.removed:
        logger.info("Compacted %s: %s MB freed", video_id, round(result.freed / 1e6))
    return result


def cleanup(config: AppConfig, store: Any, *, dry_run: bool = False, now: float | None = None) -> CleanupReport:
    """Compact every finished video and purge compact audio past its retention."""
    now = time.time() if now is None else now
    report = CleanupReport(dry_run=dry_run)
    tasks: list[Task] = list(store.list_all())
    by_video: dict[str, list[Task]] = {}
    for task in tasks:
        if task.video_id:
            by_video.setdefault(task.video_id, []).append(task)
    keep_days = int(config.storage.keep_audio_days)
    for video_id, items in sorted(by_video.items()):
        if any(_media_kind(config, task) == "image" for task in items):
            continue
        if any(task.status not in TERMINAL for task in items):
            report.skipped_active.append(video_id)
            continue
        done = any(task.status == TaskStatus.completed for task in items)
        newest = max(_updated(task) for task in items)
        expired = keep_days > 0 and now - newest > keep_days * 86400
        if done and config.storage.compact_after_processing:
            result = compact_video_media(config, video_id, dry_run=dry_run)
            m4a = compact_audio_path(config, video_id)
            if expired and m4a.exists():
                result.remove(m4a, dry_run)
                result.audio = None
            report.compacted.append(result)
        elif not done and expired:
            # Only failures, and nobody retried within the retention window: drop the derived audio,
            # and the original too unless it is meant to be kept.
            result = CompactResult(video_id)
            _dispose_originals(config, video_files(config, video_id), result, dry_run)
            for path in [wav_path(config, video_id), compact_audio_path(config, video_id)]:
                result.remove(path, dry_run)
            report.compacted.append(result)
        else:
            continue
        if not dry_run:
            _forget_paths(store, items, report.compacted[-1])
    if config.storage.mode == "archive" and config.storage.evict_archived:
        _evict_uploaded(config, report)
    known = set(by_video)
    for directory in (config.paths.videos, config.paths.audio):
        if directory.exists():
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.name.split(".")[0] not in known:
                    report.orphans.append(path)
    return report


def _dispose_originals(config: AppConfig, videos: list[Path], result: CompactResult, dry_run: bool) -> None:
    mode = config.storage.mode
    if mode == "keep":
        return
    if mode == "delete":
        for path in videos:
            result.remove(path, dry_run)
        return
    target_dir = config.storage.archive_dir
    for path in videos:
        if path.suffix.lower() in {".part", ".ytdl"} or path.name.endswith(".part"):
            result.remove(path, dry_run)  # unfinished download, nothing worth keeping
            continue
        target = target_dir / path.name
        result.archived.append(target)
        result.freed += path.stat().st_size
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))


def locate_archived(config: AppConfig, video_id: str) -> Path | None:
    directory = config.storage.archive_dir
    if not directory.is_dir():
        return None
    for path in sorted(directory.glob(f"{video_id}.*")):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES and path.stat().st_size > 0:
            return path
    return None


def is_dataless(path: Path) -> bool:
    """True when the file's content lives only in iCloud (SF_DATALESS)."""
    try:
        return bool(os.stat(path).st_flags & SF_DATALESS)
    except (OSError, AttributeError):
        return False


def materialize(path: Path, timeout: float | None = None) -> Path:
    """Make sure an iCloud file's content is on this Mac before anything reads it."""
    if not is_dataless(path):
        return path
    size_mb = max(1.0, path.stat().st_size / 1e6)
    deadline = time.monotonic() + (timeout if timeout is not None else min(1800.0, 120.0 + size_mb * 2.0))
    logger.info("Fetching %s (%.0f MB) back from iCloud", path.name, size_mb)
    subprocess.run(["brctl", "download", str(path)], check=False, capture_output=True, text=True)
    while is_dataless(path):
        if time.monotonic() > deadline:
            raise TimeoutError(f"iCloud did not deliver {path.name} in time; is this Mac online and signed in to iCloud?")
        time.sleep(1.0)
    return path


def _evict_uploaded(config: AppConfig, report: CleanupReport) -> None:
    directory = config.storage.archive_dir
    if not directory.is_dir() or shutil.which("brctl") is None:
        return
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name.startswith(".") or is_dataless(path):
            continue
        if report.dry_run:
            report.awaiting_upload.append(path)
            continue
        size = path.stat().st_size
        done = subprocess.run(["brctl", "evict", str(path)], check=False, capture_output=True, text=True)
        if done.returncode == 0 and is_dataless(path):
            report.evicted.append(path)
            report.evicted_bytes += size
        else:
            report.awaiting_upload.append(path)  # iCloud refuses until the upload has finished


def _forget_paths(store: Any, tasks: Iterable[Task], result: CompactResult) -> None:
    gone = {str(path) for path in result.removed}
    archived = str(result.archived[0]) if result.archived else ""
    for task in tasks:
        changed = False
        if task.video_path and (task.video_path in gone or not Path(task.video_path).exists()):
            task.video_path = archived
            changed = True
        if task.audio_path and task.audio_path in gone:
            task.audio_path = str(result.audio) if result.audio else ""
            changed = True
        if changed:
            store.update(task)


def _media_kind(config: AppConfig, task: Task) -> str:
    kind = str((task.extra or {}).get("media_kind") or "")
    if kind:
        return kind
    from media_pipeline.artifacts import ArtifactStore

    try:
        metadata = ArtifactStore(config.paths.artifacts, task.video_id).load_metadata()
    except Exception:
        metadata = None
    return metadata.media_kind if metadata else "video"


def _updated(task: Task) -> float:
    from datetime import datetime

    for value in (task.updated_at, task.created_at):
        try:
            return datetime.fromisoformat(str(value)).timestamp()
        except (TypeError, ValueError):
            continue
    return 0.0


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _encode_compact(source: Path, target: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial.m4a")
    command = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(source),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "48k", str(partial),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0 or not _nonempty(partial):
        partial.unlink(missing_ok=True)
        return False
    expected, actual = _duration(source), _duration(partial)
    if expected and (actual is None or abs(actual - expected) > max(2.0, expected * 0.01)):
        logger.warning("Compact audio for %s is %.0fs, expected %.0fs; keeping the original", source.name, actual or 0, expected)
        partial.unlink(missing_ok=True)
        return False
    partial.replace(target)
    return True


def _duration(path: Path) -> float | None:
    from media_pipeline.media import probe_duration

    try:
        return probe_duration(path)
    except Exception:
        return None
