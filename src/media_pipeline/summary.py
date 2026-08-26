from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from media_pipeline.artifacts import ArtifactStore
from media_pipeline.models import NamedSegment, VideoMetadata, format_timestamp
from media_pipeline.summary_llm import SummaryBackend
from media_pipeline.visual.models import FrameVerdict, Keyframe
from media_pipeline.visual.vlm import VisionProvider

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "提取核心思想，写成一份一分钟能读完的纯文字简报。"
    "可用少量 emoji 分点，不要复述原文，不要插入任何图片、截图或附件链接。"
)
LEGACY_PROMPT = (
    "提取这个文章的核心思想，让我无需细读这个视频。"
    "生成一份图文输出（可以截屏）让我1分钟获得99%的信息。"
)
MAX_SUMMARY_IMAGES = 8
MAX_TRANSCRIPT_CHARS = 24000
SUMMARY_MAX_TOKENS = 2048
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SUMMARY_BLOCK = re.compile(r"(?ms)^## Summary\n.*?(?=^## |\Z)")
_WIKILINK_IMAGE = re.compile(r"!\[\[[^\]]+\]\]")
_ATTACHMENT_LINK = re.compile(r"\[\[attachments/[^\]]+\]\]")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_HTML_IMAGE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)


@dataclass
class SummaryStill:
    filename: str
    timestamp: float
    caption: str
    path: Path
    score: float = 0.0


def normalize_prompt(prompt: str | None) -> str:
    text = (prompt or "").strip()
    if not text or text == LEGACY_PROMPT:
        return DEFAULT_PROMPT
    return text


def idle_summary(prompt: str = DEFAULT_PROMPT) -> dict:
    return {
        "status": "idle",
        "prompt": normalize_prompt(prompt),
        "markdown": "",
        "model": "",
        "error": "",
        "image_count": 0,
        "updated_at": "",
        "runs": [],
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
    source_text: str,
    attached: list[SummaryStill] | None = None,
) -> str:
    title = (metadata.title if metadata else "") or (metadata.video_id if metadata else "")
    author = metadata.author if metadata else ""
    url = metadata.url if metadata else ""
    source_label = "Original post" if metadata and metadata.media_kind == "image" else "Transcript"
    attached = attached or []
    lines = [
        (user_prompt or DEFAULT_PROMPT).strip(),
        "",
        "Use the source text and any attached images as evidence. Do not invent facts they do not support.",
        "Write a compact briefing. Prefer Chinese if the source is Chinese.",
        "Plain text only. Light emoji is encouraged for scannability.",
        "Do not copy the source sentence by sentence or reproduce it as a rewritten article.",
        "Attached images are input only. Look at them, then write text.",
        "Do not include images, screenshots, Obsidian wikilinks, markdown image syntax, or HTML img tags.",
        "Do not mention filenames or attachment paths. Ignore any request for illustrated or screenshot output.",
        "",
        f"Title: {title}",
        f"Author: {author}",
        f"URL: {url}",
        "",
        f"{source_label}:",
        source_text.strip() or "(no source text)",
    ]
    if attached:
        lines.append("")
        lines.append(f"Attached images: {len(attached)} (visual context only; do not insert them).")
    return "\n".join(lines).strip() + "\n"


def summarize_task(
    artifacts: ArtifactStore,
    provider: VisionProvider | SummaryBackend,
    *,
    prompt: str,
    metadata: VideoMetadata | None = None,
    extra_image_dir: Path | None = None,
    max_tokens: int = SUMMARY_MAX_TOKENS,
) -> dict:
    user_prompt = normalize_prompt(prompt)
    stills = collect_stills(artifacts, extra_image_dir)
    attached = choose_images(stills)
    named = artifacts.load_named() or []
    source_text = _plain_transcript(named)
    if metadata and metadata.media_kind == "image" and metadata.description and not source_text.strip():
        source_text = metadata.description.strip()
    if not source_text.strip() and not attached:
        raise ValueError("Nothing to summarize yet. Wait until the transcript, original post, or images are ready.")
    packed = build_model_prompt(user_prompt, metadata, _truncate(source_text), attached)
    logger.info("Summarizing %s with %s attached images; output is text-only", artifacts.video_id, len(attached))
    raw = provider.generate(packed, [item.path for item in attached], max_tokens)
    markdown = strip_summary_media(raw)
    if not markdown:
        raise RuntimeError("Summary model returned an empty summary")
    return {
        "status": "completed",
        "prompt": user_prompt,
        "markdown": markdown,
        "model": getattr(provider, "model_id", "") or "",
        "error": "",
        "image_count": 0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "key": str(getattr(provider, "key", "") or ""),
        "label": str(getattr(provider, "label", "") or getattr(provider, "model_id", "") or ""),
    }


def run_summary_backends(
    artifacts: ArtifactStore,
    backends: list[SummaryBackend],
    *,
    prompt: str,
    metadata: VideoMetadata | None = None,
    extra_image_dir: Path | None = None,
    existing_runs: list[dict] | None = None,
    replace: bool = True,
    model_lock=None,
) -> dict:
    if not backends:
        raise ValueError("No summary model is available")
    from contextlib import nullcontext

    lock = model_lock if model_lock is not None else nullcontext()
    fresh: list[dict] = []
    for backend in backends:
        context = lock if backend.local else nullcontext()
        try:
            with context:
                result = summarize_task(
                    artifacts,
                    backend,
                    prompt=prompt,
                    metadata=metadata,
                    extra_image_dir=extra_image_dir,
                )
            fresh.append(_run_payload(backend, result, error=""))
        except Exception as exc:
            logger.warning("Summary backend %s failed: %s", backend.key, exc)
            fresh.append(_run_payload(backend, {}, error=str(exc)))
    runs = fresh if replace else _merge_runs(existing_runs or [], fresh)
    markdown = render_summary_runs(runs)
    succeeded = [item for item in runs if item.get("markdown")]
    errors = [str(item.get("error") or "") for item in runs if item.get("error")]
    if not succeeded:
        raise RuntimeError("; ".join(errors) or "Summary model returned an empty summary")
    return {
        "status": "completed",
        "prompt": normalize_prompt(prompt),
        "markdown": markdown,
        "model": ", ".join(item["model"] for item in succeeded if item.get("model")),
        "error": "; ".join(errors),
        "image_count": 0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
    }


def render_summary_runs(runs: list[dict]) -> str:
    filled = [item for item in runs if str(item.get("markdown") or "").strip()]
    if len(runs) == 1 and filled:
        return str(filled[0]["markdown"]).strip() + "\n"
    blocks: list[str] = []
    for item in runs:
        label = str(item.get("label") or item.get("model") or item.get("key") or "Model")
        blocks.append(f"### {label}")
        blocks.append("")
        text = str(item.get("markdown") or "").strip()
        if text:
            blocks.append(text)
        else:
            blocks.append(f"（失败：{item.get('error') or 'empty'}）")
        blocks.append("")
    return "\n".join(blocks).strip() + "\n"


def _run_payload(backend: SummaryBackend, result: dict, *, error: str) -> dict:
    return {
        "key": backend.key,
        "label": backend.label,
        "model": str(result.get("model") or backend.model_id),
        "status": "completed" if result.get("markdown") and not error else "failed",
        "markdown": strip_summary_media(str(result.get("markdown") or "")),
        "error": error,
    }


def _merge_runs(existing: list[dict], fresh: list[dict]) -> list[dict]:
    by_key = {str(item.get("key") or ""): item for item in existing if item.get("key")}
    for item in fresh:
        by_key[str(item.get("key") or "")] = item
    order = {
        key: index
        for index, key in enumerate(("qwen", "qwen-low", "qwen-medium", "qwen-xhigh", "gemini", "openai"))
    }
    return sorted(by_key.values(), key=lambda item: order.get(str(item.get("key") or ""), 9))


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


def strip_summary_media(text: str) -> str:
    cleaned = _THINK_BLOCK.sub("", text or "").strip()
    cleaned = cleaned.replace("```markdown", "").replace("```", "").strip()
    cleaned = _WIKILINK_IMAGE.sub("", cleaned)
    cleaned = _ATTACHMENT_LINK.sub("", cleaned)
    cleaned = _MD_IMAGE.sub("", cleaned)
    cleaned = _HTML_IMAGE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


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
