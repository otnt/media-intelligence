from __future__ import annotations

import json
import logging
import re
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

from media_pipeline.models import NamedSegment, VideoMetadata, format_timestamp, frame_filename
from media_pipeline.visual.models import FrameVerdict, Keyframe
from media_pipeline.visual.vlm import NullVisionProvider, VisionProvider

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v1"
CATEGORIES = (
    "slide",
    "diagram",
    "document",
    "demo",
    "product",
    "chart",
    "talking_head",
    "transition",
    "blur",
    "logo",
    "other",
)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def filter_keyframes(
    keyframes: list[Keyframe],
    artifacts,
    metadata: VideoMetadata,
    segments: list[NamedSegment],
    settings: dict,
    provider: VisionProvider | None = None,
    *,
    model_lock: AbstractContextManager[object] | None = None,
) -> tuple[list[FrameVerdict], list[Keyframe]]:
    """Score keyframes with a VLM. JPEGs stay on disk; only the selected list changes."""
    provider = provider or NullVisionProvider()
    threshold = float(settings.get("vlm_keep_threshold") or 0.45)
    before_sec = float(settings.get("context_before_sec") or 10)
    after_sec = float(settings.get("context_after_sec") or 20)
    overrides = artifacts.load_overrides()
    cached = {
        item.filename: item
        for item in (artifacts.load_frame_analysis() or [])
        if item.model == provider.model_id and item.prompt_version == PROMPT_VERSION
    }
    lock = model_lock if model_lock is not None else nullcontext()
    verdicts: list[FrameVerdict] = []
    selected: list[Keyframe] = []
    total = len(keyframes)
    inferred = 0
    for index, frame in enumerate(keyframes, start=1):
        name = Path(frame.image_path).name or frame_filename(frame.timestamp)
        previous = cached.get(name)
        if previous is not None:
            raw = previous
        elif provider.name == "none":
            raw = _unavailable_verdict(name, frame, provider)
        else:
            prompt = build_prompt(metadata, segments, frame.timestamp, before_sec, after_sec)
            image_path = artifacts.root / frame.image_path
            logger.info("Judging frame %s/%s %s", index, total, name)
            with lock:
                text = provider.judge(image_path, prompt)
            inferred += 1
            raw = parse_verdict(text, filename=name, timestamp=frame.timestamp, provider=provider)
        verdict = apply_threshold_and_overrides(raw, threshold, overrides)
        verdicts.append(verdict)
        if verdict.kept:
            selected.append(frame)
    artifacts.save_frame_analysis(verdicts)
    logger.info("Frame filter kept %s/%s keyframes (%s inferred)", len(selected), total, inferred)
    return verdicts, selected


def apply_threshold_and_overrides(
    verdict: FrameVerdict,
    threshold: float,
    overrides: dict[str, str],
) -> FrameVerdict:
    updated = FrameVerdict.from_dict(verdict.to_dict())
    override = (overrides.get(updated.filename) or "").strip().lower()
    if override == "keep":
        updated.kept = True
        updated.decision = "manual"
        return updated
    if override == "drop":
        updated.kept = False
        updated.decision = "manual"
        return updated
    if updated.decision in {"parse_error", "unavailable"}:
        updated.kept = True
        return updated
    updated.kept = bool(updated.informative) and updated.score >= threshold
    updated.decision = "auto"
    return updated


def selected_from_verdicts(keyframes: list[Keyframe], verdicts: list[FrameVerdict]) -> list[Keyframe]:
    kept = {item.filename for item in verdicts if item.kept}
    chosen: list[Keyframe] = []
    for frame in keyframes:
        name = Path(frame.image_path).name or frame_filename(frame.timestamp)
        if name in kept:
            chosen.append(frame)
    return chosen


def build_prompt(
    metadata: VideoMetadata,
    segments: list[NamedSegment],
    timestamp: float,
    before_sec: float,
    after_sec: float,
) -> str:
    start = max(0.0, timestamp - before_sec)
    end = timestamp + after_sec
    nearby = [item for item in segments if item.end >= start and item.start <= end]
    if nearby:
        transcript = "\n".join(
            f"[{format_timestamp(item.start)}-{format_timestamp(item.end)}] {item.speaker_label}: {item.text.strip()}"
            for item in nearby
            if item.text.strip()
        )
    else:
        transcript = "(no nearby speech)"
    categories = ", ".join(CATEGORIES)
    return (
        "You are filtering stills extracted from a video so a later reader only sees frames "
        "that add visual information.\n\n"
        f"Video title: {metadata.title or metadata.video_id}\n"
        f"Frame timestamp: {format_timestamp(timestamp)}\n\n"
        "Nearby transcript:\n"
        f"{transcript}\n\n"
        "Look at the image. Decide whether it helps someone understand the video's content "
        "beyond what the transcript already says.\n"
        "Keep (informative=true) if the frame shows slides, diagrams, documents, UI, products, "
        "charts, book covers, on-screen text, or a distinct visual demonstration.\n"
        "Drop (informative=false) if it is a talking-head, empty transition, blur, logo bumper, "
        "or a near-repeat of the same visual with no new information.\n\n"
        "Reply with JSON only:\n"
        '{"informative": true, "score": 0.0, "category": "slide", "reason": "...", "caption": "..."}\n'
        "score is 0-1 how much unique visual information this frame adds.\n"
        f"category is one of: {categories}\n"
        "caption is one sentence describing the visual content. Use an empty string if not informative."
    )


def parse_verdict(text: str, *, filename: str, timestamp: float, provider: VisionProvider) -> FrameVerdict:
    cleaned = _THINK_BLOCK.sub("", text or "").strip()
    match = _JSON_OBJECT.search(cleaned)
    if not match:
        return _parse_error(filename, timestamp, provider, "no_json")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return _parse_error(filename, timestamp, provider, "invalid_json")
    if not isinstance(data, dict):
        return _parse_error(filename, timestamp, provider, "invalid_json")
    category = str(data.get("category") or "other").strip().lower().replace(" ", "_")
    if category not in CATEGORIES:
        category = "other"
    try:
        score = float(data.get("score"))
    except (TypeError, ValueError):
        score = 1.0 if bool(data.get("informative", True)) else 0.0
    score = min(1.0, max(0.0, score))
    caption = " ".join(str(data.get("caption") or "").split())
    reason = " ".join(str(data.get("reason") or "").split())
    informative = bool(data.get("informative")) if "informative" in data else score >= 0.5
    return FrameVerdict(
        filename=filename,
        timestamp=timestamp,
        informative=informative,
        score=score,
        category=category,
        reason=reason,
        caption=caption,
        kept=True,
        decision="auto",
        model=provider.model_id,
        prompt_version=PROMPT_VERSION,
    )


def caption_for(frame: Keyframe, verdicts: list[FrameVerdict]) -> str:
    name = Path(frame.image_path).name or frame_filename(frame.timestamp)
    for item in verdicts:
        if item.filename == name:
            return item.caption
    return ""


def _unavailable_verdict(filename: str, frame: Keyframe, provider: VisionProvider) -> FrameVerdict:
    return FrameVerdict(
        filename=filename,
        timestamp=frame.timestamp,
        informative=True,
        score=1.0,
        category="other",
        reason="analysis_unavailable",
        caption="",
        kept=True,
        decision="unavailable",
        model=provider.model_id,
        prompt_version=PROMPT_VERSION,
    )


def _parse_error(filename: str, timestamp: float, provider: VisionProvider, reason: str) -> FrameVerdict:
    logger.warning("Could not parse VLM verdict for %s (%s)", filename, reason)
    return FrameVerdict(
        filename=filename,
        timestamp=timestamp,
        informative=True,
        score=1.0,
        category="other",
        reason=reason,
        caption="",
        kept=True,
        decision="parse_error",
        model=provider.model_id,
        prompt_version=PROMPT_VERSION,
    )
