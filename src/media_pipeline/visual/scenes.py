from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

from media_pipeline.visual.models import SceneSpan

logger = logging.getLogger(__name__)

_PTS = re.compile(r"pts_time:\s*([0-9.]+)")


class SceneDetector:
    name = "base"

    def detect(self, video_path: Path, *, threshold: float, min_scene_duration: float, duration: float) -> list[SceneSpan]:
        raise NotImplementedError


class FFmpegSceneDetector(SceneDetector):
    name = "ffmpeg"

    def detect(self, video_path: Path, *, threshold: float, min_scene_duration: float, duration: float) -> list[SceneSpan]:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required for scene detection")
        score = _ffmpeg_scene_score(threshold)
        completed = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-i",
                str(video_path),
                "-filter:v",
                f"select='gt(scene,{score:.4f})',showinfo",
                "-an",
                "-f",
                "null",
                "-",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        cuts = [0.0]
        for match in _PTS.finditer(completed.stderr or ""):
            stamp = float(match.group(1))
            if stamp - cuts[-1] >= max(0.12, min_scene_duration / 2):
                cuts.append(stamp)
        return _spans_from_cuts(cuts, duration, min_scene_duration)


class PySceneDetectDetector(SceneDetector):
    name = "pyscenedetect"

    def detect(self, video_path: Path, *, threshold: float, min_scene_duration: float, duration: float) -> list[SceneSpan]:
        from scenedetect import ContentDetector, SceneManager, open_video

        video = open_video(str(video_path))
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=_pyscene_threshold(threshold), min_scene_len=1))
        manager.detect_scenes(video, show_progress=False)
        raw = manager.get_scene_list()
        cuts = [0.0]
        for start, _end in raw:
            stamp = float(start.get_seconds())
            if stamp - cuts[-1] >= max(0.12, min_scene_duration / 2):
                cuts.append(stamp)
        return _spans_from_cuts(cuts, duration, min_scene_duration)


def build_scene_detector(preferred: str = "auto") -> SceneDetector:
    wanted = (preferred or "auto").strip().lower()
    if wanted in {"auto", "pyscenedetect", "scenedetect"}:
        try:
            import scenedetect  # noqa: F401

            return PySceneDetectDetector()
        except ImportError:
            if wanted != "auto":
                logger.warning("PySceneDetect is not installed; using ffmpeg scene detection")
    return FFmpegSceneDetector()


def _spans_from_cuts(cuts: list[float], duration: float, min_scene_duration: float) -> list[SceneSpan]:
    points = sorted({max(0.0, item) for item in cuts if item >= 0})
    if not points or points[0] > 0.01:
        points.insert(0, 0.0)
    end = max(duration, points[-1])
    if end - points[-1] > 0.05:
        points.append(end)
    elif points[-1] < end:
        points[-1] = end
    scenes: list[SceneSpan] = []
    for start, stop in zip(points, points[1:]):
        if stop - start < 0.04:
            continue
        if scenes and (stop - start) < min_scene_duration:
            scenes[-1] = SceneSpan(scenes[-1].start, stop)
            continue
        scenes.append(SceneSpan(start, stop))
    if not scenes:
        scenes.append(SceneSpan(0.0, max(0.0, duration)))
    return scenes


def _ffmpeg_scene_score(threshold: float) -> float:
    if threshold <= 1:
        return max(0.05, min(0.9, threshold))
    return max(0.05, min(0.9, threshold / 90.0))


def _pyscene_threshold(threshold: float) -> float:
    if threshold <= 1:
        return max(8.0, threshold * 90.0)
    return threshold
