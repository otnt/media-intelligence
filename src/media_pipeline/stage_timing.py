from __future__ import annotations

import time
from datetime import datetime
from typing import Any

STAGE_ORDER = (
    "fetching_metadata",
    "downloading",
    "extracting_audio",
    "transcribing",
    "diarizing",
    "aligning_transcript",
    "detecting_scenes",
    "sampling_frames",
    "deduplicating_frames",
    "aligning_multimodal",
    "writing_outputs",
)

_VISUAL_FROM_SCENES = frozenset({"detecting_scenes", "all"})
_VISUAL_FROM_SAMPLE = _VISUAL_FROM_SCENES | {"sampling_frames"}
_VISUAL_FROM_DEDUP = _VISUAL_FROM_SAMPLE | {"deduplicating_frames"}
_VISUAL_FROM_ALIGN = _VISUAL_FROM_DEDUP | {"aligning_multimodal", "writing_outputs"}


def now_iso(moment: datetime | None = None) -> str:
    stamp = moment or datetime.now().astimezone()
    if stamp.tzinfo is None:
        stamp = stamp.astimezone()
    return stamp.isoformat(timespec="seconds")


def public_entry(entry: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(entry, dict):
        return {}
    return {key: value for key, value in entry.items() if not str(key).startswith("_")}


def public_timings(raw: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, entry in (raw or {}).items():
        payload = public_entry(entry if isinstance(entry, dict) else None)
        if payload:
            out[str(key)] = payload
    return out


def timings_map(extra: dict[str, Any]) -> dict[str, Any]:
    current = extra.get("stage_timings")
    if not isinstance(current, dict):
        current = {}
        extra["stage_timings"] = current
    return current


def begin_stage(
    extra: dict[str, Any],
    key: str,
    *,
    now: datetime | None = None,
    monotonic: float | None = None,
) -> dict[str, Any]:
    entry = {
        "started_at": now_iso(now),
        "status": "running",
        "_mono": float(time.monotonic() if monotonic is None else monotonic),
    }
    timings_map(extra)[key] = entry
    return entry


def finish_stage(
    extra: dict[str, Any],
    key: str,
    *,
    succeeded: bool,
    now: datetime | None = None,
    monotonic: float | None = None,
) -> dict[str, Any]:
    timings = timings_map(extra)
    entry = dict(timings.get(key) or {})
    if entry.get("status") in {"succeeded", "failed"} and "_mono" not in entry:
        return public_entry(entry)
    started_mono = entry.pop("_mono", None)
    entry["ended_at"] = now_iso(now)
    if succeeded:
        entry["status"] = "succeeded"
        clock = time.monotonic() if monotonic is None else monotonic
        if started_mono is not None:
            entry["duration_sec"] = round(max(0.0, float(clock) - float(started_mono)), 3)
    else:
        entry["status"] = "failed"
        entry.pop("duration_sec", None)
    if "started_at" not in entry:
        entry["started_at"] = entry["ended_at"]
    timings[key] = entry
    return public_entry(entry)


def stage_keys_invalidated_by(stage: str) -> set[str]:
    """Stage timing keys that must be dropped when artifacts from `stage` are invalidated."""
    keys: set[str] = set()
    if stage in {"transcribing", "all"}:
        keys.add("transcribing")
    if stage in {"diarizing", "transcribing", "all"}:
        keys.add("diarizing")
    if stage in {"aligning_transcript", "diarizing", "transcribing", "all"}:
        keys.add("aligning_transcript")
    if stage in _VISUAL_FROM_SCENES:
        keys.add("detecting_scenes")
    if stage in _VISUAL_FROM_SAMPLE:
        keys.add("sampling_frames")
    if stage in _VISUAL_FROM_DEDUP:
        keys.add("deduplicating_frames")
    if stage in _VISUAL_FROM_ALIGN:
        keys.add("aligning_multimodal")
        keys.add("writing_outputs")
    return keys


def clear_invalidated_timings(extra: dict[str, Any], stage: str) -> dict[str, Any]:
    drop = stage_keys_invalidated_by(stage)
    timings = timings_map(extra)
    extra["stage_timings"] = {key: value for key, value in timings.items() if key not in drop}
    return extra["stage_timings"]


def merge_stage_timings(*sources: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for source in sources:
        merged.update(public_timings(source))
    return merged
