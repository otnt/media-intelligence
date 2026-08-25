from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from media_pipeline.artifacts import ArtifactStore
from media_pipeline.models import NamedSegment, VideoMetadata, format_timestamp
from media_pipeline.visual.models import FrameVerdict, Keyframe
from media_pipeline.visual.vlm import VisionProvider

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "提取这个文章的核心思想，让我无需细读这个视频。"
    "生成一份图文输出（可以截屏）让我1分钟获得99%的信息。"
)
MAX_SUMMARY_IMAGES = 8
MAX_TRANSCRIPT_CHARS = 24000
SUMMARY_MAX_TOKENS = 2048
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SUMMARY_BLOCK = re.compile(r"(?ms)^## Summary\n.*?(?=^## |\Z)")


@dataclass
class SummaryStill:
    filename: str
    timestamp: float
    caption: str
    path: Path
    score: float = 0.0


def idle_summary(prompt: str = DEFAULT_PROMPT) -> dict:
    return {
        "status": "idle",
        "prompt": prompt or DEFAULT_PROMPT,
        "markdown": "",
        "model": "",
        "error": "",
        "image_count": 0,
        "updated_at": "",
    }


def extract_summary_section(body: str) -> str:
    match = _SUMMARY_BLOCK.search(body or "")
    if not match:
        return ""
    return re.sub(r"^## Summary\s*", "", match.group(0)).strip()


def strip_summary_section(body: str) -> str:
    return _SUMMARY_BLOCK.sub("", body or "").strip()


def upsert_summary_section(body: str, summary: str) -> str:
    block = "## Summary\n\n" + (summary or "").strip() + "\n"
    text = body or ""
    if _SUMMARY_BLOCK.search(text):
        return _SUMMARY_BLOCK.sub(block + "\n", text, count=1).strip() + "\n"
    rest = text.strip()
    if not rest:
        return block + "\n"
    return block + "\n" + rest + "\n"


def collect_stills(artifacts: ArtifactStore, extra_dir: Path | None = None) -> list[SummaryStill]:
    stills: list[SummaryStill] = []
    keyframes = artifacts.load_keyframes() or []
    analysis = artifacts.load_frame_analysis() or []
    by_name = {Path(frame.image_path).name: frame for frame in keyframes}
    if analysis:
        kept = [item for item in analysis if item.kept]
        source = kept or analysis
        for verdict in source:
            frame = by_name.get(verdict.filename)
            path = _still_path(artifacts, verdict, frame)
            if path is None:
                continue
            stills.append(
                SummaryStill(
                    filename=verdict.filename,
                    timestamp=verdict.timestamp,
                    caption=verdict.caption,
                    path=path,
                    score=float(verdict.score or 0.0),
                )
            )
    elif keyframes:
        for frame in keyframes:
            name = Path(frame.image_path).name
            path = artifacts.root / frame.image_path
            if not path.is_file():
                continue
            stills.append(SummaryStill(filename=name, timestamp=frame.timestamp, caption="", path=path, score=1.0))
    if stills:
        return stills
    if extra_dir is None or not extra_dir.is_dir():
        return []
    images = sorted(
        path
        for path in extra_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    return [
        SummaryStill(filename=path.name, timestamp=float(index), caption="", path=path, score=1.0)
        for index, path in enumerate(images, start=1)
        if path.stat().st_size > 0
    ]


def choose_images(stills: list[SummaryStill], limit: int = MAX_SUMMARY_IMAGES) -> list[SummaryStill]:
    if len(stills) <= limit:
        return sorted(stills, key=lambda item: (item.timestamp, item.filename))
    ranked = sorted(stills, key=lambda item: (-item.score, item.timestamp, item.filename))
    chosen = ranked[:limit]
    return sorted(chosen, key=lambda item: (item.timestamp, item.filename))


def build_model_prompt(
    user_prompt: str,
    metadata: VideoMetadata | None,
    transcript: str,
    stills: list[SummaryStill],
    attached: list[SummaryStill],
) -> str:
    title = (metadata.title if metadata else "") or (metadata.video_id if metadata else "")
    author = metadata.author if metadata else ""
    url = metadata.url if metadata else ""
    video_id = metadata.video_id if metadata else ""
    lines = [
        (user_prompt or DEFAULT_PROMPT).strip(),
        "",
        "Use the transcript and attached stills. Do not invent facts they do not support.",
        "Write a compact illustrated briefing. Prefer Chinese if the source is Chinese.",
        "When you show a still, emit an Obsidian wikilink on its own line:",
        f"![[attachments/{video_id or 'VIDEO_ID'}/FILENAME.jpg]]",
        "FILENAME must match one of the stills listed below.",
        "",
        f"Title: {title}",
        f"Author: {author}",
        f"URL: {url}",
        "",
        "Transcript:",
        transcript.strip() or "(no speech transcript)",
        "",
        "Stills in chronological order:",
    ]
    for index, still in enumerate(stills, start=1):
        stamp = format_timestamp(still.timestamp) if still.timestamp else str(index)
        caption = still.caption or ""
        attached_mark = " [attached image]" if still.filename in {item.filename for item in attached} else ""
        lines.append(f"[{index}] {stamp} {still.filename}{attached_mark} {caption}".rstrip())
    return "\n".join(lines).strip() + "\n"


def summarize_task(
    artifacts: ArtifactStore,
    provider: VisionProvider,
    *,
    prompt: str,
    metadata: VideoMetadata | None = None,
    extra_image_dir: Path | None = None,
    max_tokens: int = SUMMARY_MAX_TOKENS,
) -> dict:
    user_prompt = (prompt or "").strip() or DEFAULT_PROMPT
    stills = collect_stills(artifacts, extra_image_dir)
    named = artifacts.load_named() or []
    transcript = _plain_transcript(named)
    if metadata and metadata.media_kind == "image" and metadata.description and not transcript.strip():
        transcript = metadata.description.strip()
    if not transcript.strip() and not stills:
        raise ValueError("Nothing to summarize yet. Wait until the transcript or stills are ready.")
    attached = choose_images(stills)
    packed = build_model_prompt(user_prompt, metadata, _truncate(transcript), stills, attached)
    logger.info("Summarizing %s with %s stills (%s attached)", artifacts.video_id, len(stills), len(attached))
    raw = provider.generate(packed, [item.path for item in attached], max_tokens)
    markdown = _clean_output(raw)
    if not markdown:
        raise RuntimeError("Qwen3.8 returned an empty summary")
    return {
        "status": "completed",
        "prompt": user_prompt,
        "markdown": markdown,
        "model": provider.model_id,
        "error": "",
        "image_count": len(attached),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _plain_transcript(segments: list[NamedSegment]) -> str:
    lines: list[str] = []
    for segment in segments:
        text = (segment.text or "").strip()
        if not text:
            continue
        lines.append(f"[{format_timestamp(segment.start)}] {segment.speaker_label}: {text}")
    return "\n".join(lines)


def _truncate(text: str, limit: int = MAX_TRANSCRIPT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n…(transcript truncated)\n"


def _clean_output(text: str) -> str:
    cleaned = _THINK_BLOCK.sub("", text or "").strip()
    cleaned = cleaned.replace("```markdown", "").replace("```", "").strip()
    return cleaned


def _still_path(artifacts: ArtifactStore, verdict: FrameVerdict, frame: Keyframe | None) -> Path | None:
    candidates = []
    if frame is not None:
        candidates.append(artifacts.root / frame.image_path)
    candidates.append(artifacts.keyframe_dir / verdict.filename)
    candidates.append(artifacts.candidate_dir / verdict.filename)
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None
