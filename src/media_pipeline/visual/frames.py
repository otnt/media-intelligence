from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from media_pipeline.visual.timestamps import CHANGE_PAIR_GAP_SEC, pair_change_timestamps

logger = logging.getLogger(__name__)


def extract_frame(video_path: Path, timestamp: float, dest: Path, *, width: int = 960) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to extract frames")
    dest.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(0.0, timestamp):.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            f"scale={width}:-2",
            "-q:v",
            "3",
            str(dest),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
        detail = (completed.stderr or completed.stdout or "ffmpeg produced no frame").strip().splitlines()[-1:]
        raise RuntimeError(detail[-1] if detail else "ffmpeg frame extract failed")
    return dest


def visual_change_timestamps(
    video_path: Path,
    duration: float,
    *,
    threshold: float,
    fps: float = 1.0,
    width: int = 160,
    height: int = 90,
) -> list[float]:
    """Scan a low-resolution 1 fps stream and emit times where the picture changes.

    Each spike yields the gray frame before the change and the frame after it.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or duration <= 0:
        return []
    frame_bytes = width * height
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps},scale={width}:{height},format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    try:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        logger.warning("Could not start ffmpeg visual-change scan: %s", exc)
        return []
    assert proc.stdout is not None
    previous: list[int] | None = None
    after_stamps: list[float] = []
    index = 0
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break
            hist = _gray_hist(raw)
            if previous is not None:
                delta = _l1(hist, previous) / (2 * len(raw))
                if delta >= threshold:
                    stamp = round(index / fps, 3)
                    if stamp < duration and (not after_stamps or stamp - after_stamps[-1] >= 0.75):
                        after_stamps.append(stamp)
            previous = hist
            index += 1
    finally:
        proc.kill()
        proc.wait(timeout=5)
    gap = CHANGE_PAIR_GAP_SEC if fps <= 0 else 1.0 / fps
    return pair_change_timestamps(after_stamps, gap_sec=gap, duration=duration)


def _gray_hist(raw: bytes) -> list[int]:
    bins = [0] * 16
    for value in raw:
        bins[value >> 4] += 1
    return bins


def _l1(left: list[int], right: list[int]) -> int:
    return sum(abs(a - b) for a, b in zip(left, right, strict=False))
