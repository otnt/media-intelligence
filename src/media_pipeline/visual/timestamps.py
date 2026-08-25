from __future__ import annotations

from media_pipeline.models import frame_filename
from media_pipeline.visual.models import CandidateFrame, SceneSpan

SOURCE_SCENE = "scene_boundary"
SOURCE_PERIODIC = "periodic_sample"
SOURCE_CHANGE = "visual_change"
SOURCE_OCR = "ocr_change"
SOURCE_MANUAL = "manual"

NEAR_MERGE_SEC = 0.4


def periodic_timestamps(duration: float, interval_sec: float) -> list[float]:
    if duration <= 0 or interval_sec <= 0:
        return []
    stamps = [0.0]
    cursor = interval_sec
    while cursor < duration - 0.05:
        stamps.append(round(cursor, 3))
        cursor += interval_sec
    last = max(0.0, round(duration - 0.04, 3))
    if last - stamps[-1] >= min(1.0, interval_sec / 2):
        stamps.append(last)
    return stamps


def scene_boundary_timestamps(scenes: list[SceneSpan]) -> list[float]:
    stamps: list[float] = []
    for scene in scenes:
        start = max(0.0, round(scene.start, 3))
        if not stamps or start - stamps[-1] > 0.05:
            stamps.append(start)
    return stamps


def merge_candidate_times(
    groups: dict[str, list[float]],
    *,
    near_sec: float = NEAR_MERGE_SEC,
) -> list[tuple[float, list[str]]]:
    """Union timestamps and attach every source that landed near each keeper."""
    labeled: list[tuple[float, str]] = []
    for source, stamps in groups.items():
        for stamp in stamps:
            if stamp < 0:
                continue
            labeled.append((round(float(stamp), 3), source))
    labeled.sort()
    merged: list[tuple[float, list[str]]] = []
    for stamp, source in labeled:
        if merged and stamp - merged[-1][0] <= near_sec:
            if source not in merged[-1][1]:
                merged[-1][1].append(source)
            continue
        merged.append((stamp, [source]))
    return merged


def candidate_path_name(timestamp: float) -> str:
    return f"candidate_frames/{frame_filename(timestamp)}"


def keyframe_path_name(timestamp: float) -> str:
    return f"keyframes/{frame_filename(timestamp)}"


def candidates_from_groups(groups: dict[str, list[float]]) -> list[CandidateFrame]:
    return [
        CandidateFrame(timestamp=stamp, path=candidate_path_name(stamp), sources=sources)
        for stamp, sources in merge_candidate_times(groups)
    ]
