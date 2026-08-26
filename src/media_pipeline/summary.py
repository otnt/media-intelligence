from __future__ import annotations

import logging
import re
import time
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
    "你是一位资深分析师，擅长快速提炼复杂内容的本质，并将其转化为结构清晰、有深度的中文简报。"
    "你的目标读者是希望快速理解核心思想、逻辑链条和关键细节的人，而不是只想知道表面事实的人。\n"
    "\n"
    "请基于以下内容，输出一份纯文字总结，要求如下：\n"
    "\n"
    "1. 先写出一句话或一小段核心结论：这份内容最重要的判断、冲突、问题或目的是什么。\n"
    "2. 提炼主要论证链条或信息结构，解释“为什么”和“如何”，而不仅仅罗列“是什么”。\n"
    "3. 保留支撑核心结论的必要事实、数据、条件、案例、步骤或技术细节；次要花絮、重复表述和气氛铺垫直接丢掉。\n"
    "4. 如果内容涉及多个观点、方案、主体或步骤，请对比它们的异同或逻辑关系。\n"
    "5. 如果原内容包含风险、趋势、限制条件、启示或后续值得关注的问题，请点明，不要因为篇幅而省略。\n"
    "6. 使用简洁的中文，可少量使用 emoji 作为分点符号，但不要复述原文，不要插入图片或链接。\n"
    "7. 篇幅由信息密度决定：原文短、信息少就短写；原文长、分支多或论证深，就把关键信息写完整。"
    "不要为了凑长度而重复，也不要为了写短而丢掉判断所依赖的事实。"
)
LENGTH_CAPPED_PROMPT = (
    "你是一位资深分析师，擅长快速提炼复杂内容的本质，并将其转化为结构清晰、有深度的中文简报。"
    "你的目标读者是希望快速理解核心思想、逻辑链条和关键细节的人，而不是只想知道表面事实的人。\n"
    "\n"
    "请基于以下内容，输出一份纯文字总结，要求如下：\n"
    "\n"
    "1. 先写出一句话或一小段核心结论：这份内容最重要的判断、冲突、问题或目的是什么。\n"
    "2. 提炼主要论证链条或信息结构，解释“为什么”和“如何”，而不仅仅罗列“是什么”。\n"
    "3. 保留支撑核心结论的必要事实、数据、案例或技术细节，但不要做无关堆砌。\n"
    "4. 如果内容涉及多个观点、方案、主体或步骤，请对比它们的异同或逻辑关系。\n"
    "5. 如果原内容包含风险、趋势、启示或后续值得关注的问题，请在结尾适当点明。\n"
    "6. 使用简洁的中文，可少量使用 emoji 作为分点符号，但不要复述原文，不要插入图片或链接。\n"
    "7. 篇幅控制在阅读时间约 1.5–2 分钟，信息密度要高，不要为了凑长度而重复内容。"
)
PREVIOUS_DEFAULT_PROMPT = (
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
_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
_THINK_TAG = re.compile(r"</?think>", re.IGNORECASE)
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
    if not text or text in {LEGACY_PROMPT, PREVIOUS_DEFAULT_PROMPT, LENGTH_CAPPED_PROMPT}:
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
        "Match length to information density. Prefer Chinese if the source is Chinese.",
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
    thinking, markdown = parse_summary_output(raw)
    if not markdown and not thinking:
        raise RuntimeError("Summary model returned an empty summary")
    return {
        "status": "completed",
        "prompt": user_prompt,
        "markdown": markdown,
        "thinking": thinking,
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
    replace: bool = False,
    model_lock=None,
) -> dict:
    if not backends:
        raise ValueError("No summary model is available")
    from contextlib import nullcontext

    lock = model_lock if model_lock is not None else nullcontext()
    fresh: list[dict] = []
    for backend in backends:
        context = lock if backend.local else nullcontext()
        started = time.perf_counter()
        result: dict = {}
        error = ""
        try:
            with context:
                result = summarize_task(
                    artifacts,
                    backend,
                    prompt=prompt,
                    metadata=metadata,
                    extra_image_dir=extra_image_dir,
                )
        except Exception as exc:
            logger.warning("Summary backend %s failed: %s", backend.key, exc)
            error = str(exc)
        duration_sec = round(max(0.0, time.perf_counter() - started), 3)
        fresh.append(_run_payload(backend, result, error=error, duration_sec=duration_sec))
    history = [] if replace else list(existing_runs or [])
    runs = history + fresh
    markdown = render_summary_runs(runs)
    succeeded = [item for item in runs if item.get("markdown") or item.get("thinking")]
    errors = [str(item.get("error") or "") for item in fresh if item.get("error")]
    return {
        "status": "completed" if succeeded else "failed",
        "prompt": normalize_prompt(prompt),
        "markdown": markdown,
        "model": ", ".join(item["model"] for item in succeeded if item.get("model")),
        "error": "; ".join(errors),
        "image_count": 0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
    }


def coerce_summary_runs(existing: dict | None) -> list[dict]:
    payload = existing or {}
    runs = [item for item in (payload.get("runs") or []) if isinstance(item, dict)]
    if runs:
        return [_with_thinking(item) for item in runs]
    markdown = str(payload.get("markdown") or "").strip()
    if not markdown and not str(payload.get("thinking") or "").strip():
        return []
    return [
        _with_thinking(
            {
                "key": str(payload.get("key") or ""),
                "label": str(payload.get("label") or payload.get("model") or "Previous summary"),
                "model": str(payload.get("model") or ""),
                "status": "completed",
                "markdown": markdown,
                "thinking": str(payload.get("thinking") or ""),
                "error": "",
                "duration_sec": payload.get("duration_sec"),
                "created_at": str(payload.get("updated_at") or ""),
            }
        )
    ]


def render_summary_runs(runs: list[dict]) -> str:
    if not runs:
        return ""
    blocks: list[str] = []
    for item in reversed(runs):
        blocks.append(f"### {_run_heading(item)}")
        blocks.append("")
        thinking = str(item.get("thinking") or "").strip()
        if thinking:
            blocks.append("<details>")
            blocks.append("<summary>Thinking</summary>")
            blocks.append("")
            blocks.append(thinking)
            blocks.append("")
            blocks.append("</details>")
            blocks.append("")
        text = str(item.get("markdown") or "").strip()
        if text:
            blocks.append(text)
        elif not thinking:
            blocks.append(f"（失败：{item.get('error') or 'empty'}）")
        blocks.append("")
    return "\n".join(blocks).strip() + "\n"


def format_run_duration(seconds: float | None) -> str:
    if seconds is None or seconds == "":
        return ""
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return ""
    if value < 0:
        value = 0.0
    if value < 10:
        return f"{value:.1f}s"
    if value < 60:
        return f"{int(round(value))}s"
    total = int(round(value))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s"


def _run_heading(item: dict) -> str:
    label = str(item.get("label") or item.get("model") or item.get("key") or "Model")
    duration = format_run_duration(item.get("duration_sec"))
    if duration:
        return f"{label} · {duration}"
    return label


def _run_payload(backend: SummaryBackend, result: dict, *, error: str, duration_sec: float = 0.0) -> dict:
    thinking, markdown = _payload_thinking(result)
    return {
        "key": backend.key,
        "label": backend.label,
        "model": str(result.get("model") or backend.model_id),
        "status": "completed" if (markdown or thinking) and not error else "failed",
        "markdown": markdown,
        "thinking": thinking,
        "error": error,
        "duration_sec": duration_sec,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def parse_summary_output(text: str) -> tuple[str, str]:
    thinking, answer = split_summary_thinking(text)
    return strip_summary_media(thinking), strip_summary_media(answer)


def split_summary_thinking(text: str) -> tuple[str, str]:
    raw = text or ""
    chunks: list[str] = []

    def _keep(match: re.Match[str]) -> str:
        chunks.append(match.group(1).strip())
        return "\n"

    remainder = _THINK_BLOCK.sub(_keep, raw)
    close = _THINK_CLOSE.search(remainder)
    if close:
        chunks.append(remainder[: close.start()].strip())
        remainder = remainder[close.end() :]
    remainder = _THINK_OPEN.sub("", remainder)
    thinking = "\n\n".join(part for part in chunks if part)
    return thinking.strip(), remainder.strip()


def _with_thinking(item: dict) -> dict:
    stored = str(item.get("thinking") or "").strip()
    extracted, answer = parse_summary_output(str(item.get("markdown") or ""))
    thinking = strip_summary_media(stored) or extracted
    markdown = answer if extracted else strip_summary_media(str(item.get("markdown") or ""))
    return {**item, "thinking": thinking, "markdown": markdown}


def _payload_thinking(result: dict) -> tuple[str, str]:
    item = _with_thinking(result)
    return str(item.get("thinking") or ""), str(item.get("markdown") or "")


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
    cleaned = _THINK_TAG.sub("", cleaned).strip()
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
