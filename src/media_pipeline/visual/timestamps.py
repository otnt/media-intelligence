from __future__ import annotations

from media_pipeline.models import frame_filename
from media_pipeline.visual.models import CandidateFrame, SceneSpan

SOURCE_SCENE = "scene_boundary"
SOURCE_PERIODIC = "periodic_sample"
SOURCE_CHANGE = "visual_change"
SOURCE_OCR = "ocr_change"
SOURCE_MANUAL = "manual"

NEAR_MERGE_SEC = 0.4
# Land inside the outgoing scene, not on the next GOP after a cut.
SCENE_PRE_CUT_SEC = 0.5
CHANGE_PAIR_GAP_SEC = 1.0


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


def scene_boundary_timestamps(
    scenes: list[SceneSpan],
    *,
    pre_cut_sec: float = SCENE_PRE_CUT_SEC,
) -> list[float]:
    """Start of each scene, plus a still just before the cut when the span is long enough.

    Raw ``scene.end`` is the next scene's first frame, so it is not sampled.
    """
    stamps: list[float] = []
    offset = max(0.0, float(pre_cut_sec))
    for scene in scenes:
        start = max(0.0, round(float(scene.start), 3))
        end = max(start, round(float(scene.end), 3))
        _append_distinct(stamps, start)
        if offset <= 0 or (end - start) <= offset + 0.05:
            continue
        pre_cut = round(max(start, end - offset), 3)
        if pre_cut - start > 0.05:
            _append_distinct(stamps, pre_cut)
    return stamps


def pair_change_timestamps(
    after_stamps: list[float],
    *,
    gap_sec: float = CHANGE_PAIR_GAP_SEC,
    duration: float,
) -> list[float]:
    """Keep the 1 fps gray frame before a visual-change spike and the frame after it."""
    gap = max(0.0, float(gap_sec))
    limit = max(0.0, float(duration))
    stamps: list[float] = []
    for after in after_stamps:
        after_stamp = round(float(after), 3)
        before_stamp = round(after_stamp - gap, 3)
        for stamp in (before_stamp, after_stamp):
            if stamp < 0 or stamp >= limit:
                continue
            stamps.append(stamp)
    stamps.sort()
    unique: list[float] = []
    for stamp in stamps:
        _append_distinct(unique, stamp)
    return unique


def _append_distinct(stamps: list[float], stamp: float, *, min_gap: float = 0.05) -> None:
    if not stamps or stamp - stamps[-1] > min_gap:
        stamps.append(stamp)


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
