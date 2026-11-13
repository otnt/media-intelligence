from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from media_pipeline.config import AppConfig, DEFAULT_QWEN_8BIT_MODEL, DEFAULT_SUMMARY_PROVIDERS

logger = logging.getLogger(__name__)

BACKEND_ORDER = ("qwen", "gemini", "openai")


def qwen_summary_label(quant: str, thinking: str) -> str:
    return f"qwen-3.8-27B {quant} thinking {thinking}"


QWEN_INSTRUCT = ("qwen", qwen_summary_label("4bit", "none"))
QWEN_THINKING = (
    ("qwen-low", "low", qwen_summary_label("4bit", "low")),
    ("qwen-medium", "medium", qwen_summary_label("4bit", "medium")),
    ("qwen-xhigh", "xhigh", qwen_summary_label("4bit", "xhigh")),
)
QWEN_8BIT_INSTRUCT = ("qwen-8bit", qwen_summary_label("8bit", "none"))
QWEN_8BIT_THINKING = (
    ("qwen-8bit-low", "low", qwen_summary_label("8bit", "low")),
    ("qwen-8bit-medium", "medium", qwen_summary_label("8bit", "medium")),
    ("qwen-8bit-xhigh", "xhigh", qwen_summary_label("8bit", "xhigh")),
)
ALL_BACKEND_KEYS = (
    QWEN_INSTRUCT[0],
    *(item[0] for item in QWEN_THINKING),
    QWEN_8BIT_INSTRUCT[0],
    *(item[0] for item in QWEN_8BIT_THINKING),
    "gemini",
    "openai",
    "deepseek-v4-pro",
    "deepseek-v4.1-flash",
    "qwen3.8-max",
    "agy-gemini",
)
THINKING_MAX_TOKENS = 8192
# Roughly 100-250k tokens: well inside the 1M-token windows of the cloud summary models.
CLOUD_SOURCE_CHARS = 400_000
AGY_SOURCE_CHARS = 150_000
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# DeepSeek V4.1 Flash served by Alibaba Cloud (same DashScope key as qwen3.8-max). Unlike DeepSeek's own
# API it takes images, so slides and keyframes reach it. Off-peak (22:00-08:00 UTC+8) costs half.
DASHSCOPE_FLASH_MODEL = "deepseek-v4.1-flash"
# The same model on DeepSeek's own API (preferred when a DeepSeek key is set): vision, 1M context, and
# off-peak half price outside 01-04 and 06-10 UTC on weekdays (weekends are off-peak all day).
DEEPSEEK_FLASH_MODEL = "deepseek-flash"
# Mainland China DashScope default. Match key region:
# Singapore intl: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
# US (Virginia):  https://dashscope-us.aliyuncs.com/compatible-mode/v1
DEFAULT_QWEN_MAX_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_KEY_ALIASES = {
    "qwen3.8": "qwen",
    "local": "qwen",
    "qwen-thinking-low": "qwen-low",
    "qwen_thinking_low": "qwen-low",
    "qwen_low": "qwen-low",
    "qwen-thinking-medium": "qwen-medium",
    "qwen_thinking_medium": "qwen-medium",
    "qwen_medium": "qwen-medium",
    "qwen-thinking-xhigh": "qwen-xhigh",
    "qwen_thinking_xhigh": "qwen-xhigh",
    "qwen_xhigh": "qwen-xhigh",
    "qwen-x-high": "qwen-xhigh",
    "qwen8bit": "qwen-8bit",
    "qwen-8": "qwen-8bit",
    "qwen_8bit": "qwen-8bit",
    "qwen-8bit-thinking-low": "qwen-8bit-low",
    "qwen-8bit-thinking-medium": "qwen-8bit-medium",
    "qwen-8bit-thinking-xhigh": "qwen-8bit-xhigh",
    "deepseek": "deepseek-v4-pro",
    "deepseek-pro": "deepseek-v4-pro",
    "deepseek_v4_pro": "deepseek-v4-pro",
    "deepseek-v4": "deepseek-v4-pro",
    "flash": "deepseek-v4.1-flash",
    "deepseek-flash": "deepseek-v4.1-flash",
    "ds-flash": "deepseek-v4.1-flash",
    "deepseek-v4.1": "deepseek-v4.1-flash",
    "deepseek_v4_1_flash": "deepseek-v4.1-flash",
    "qwen-max": "qwen3.8-max",
    "qwen_max": "qwen3.8-max",
    "qwen3.8max": "qwen3.8-max",
    "qwen38-max": "qwen3.8-max",
    "agy": "agy-gemini",
    "gemini-agy": "agy-gemini",
}
_IMAGE_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
_MAX_IMAGE_BYTES = 4 * 1024 * 1024


@dataclass
class SummaryBackend:
    key: str
    label: str
    model_id: str
    local: bool = False
    generate_fn: Callable[..., str] | None = None
    # Whether the model actually receives attached images (DeepSeek V4 Pro is text-only).
    accepts_images: bool = True
    # How much transcript the backend gets. Local 27B models need the small default; the cloud models
    # here have ~1M-token windows, and cutting at 24k chars dropped the second half of long videos.
    max_source_chars: int = 24000

    def generate(self, prompt: str, images: list[Path] | None = None, max_tokens: int | None = None) -> str:
        if self.generate_fn is None:
            raise RuntimeError(f"Summary backend {self.key} is not configured")
        return self.generate_fn(prompt, images, max_tokens)


def catalog_summary_backends(config: AppConfig, vision: object | None = None) -> list[dict[str, Any]]:
    """Describe every backend, including ones that still need a key or local weights."""
    rows = []
    qwen = _qwen_backend(config, vision)
    qwen_available = qwen is not None
    qwen_detail = ""
    qwen_model = qwen.model_id if qwen else str(config.analysis.model)
    if not qwen_available:
        qwen_available, qwen_detail = _probe_qwen(config)
    rows.append(
        _catalog_row(
            QWEN_INSTRUCT[0],
            QWEN_INSTRUCT[1],
            qwen_model,
            qwen_available,
            qwen_detail,
        )
    )
    for key, _effort, label in QWEN_THINKING:
        rows.append(_catalog_row(key, label, qwen_model, qwen_available, qwen_detail))
    eight_model = _qwen_8bit_model(config)
    eight_available, eight_detail = _probe_qwen_8bit(config)
    rows.append(
        _catalog_row(
            QWEN_8BIT_INSTRUCT[0],
            QWEN_8BIT_INSTRUCT[1],
            eight_model,
            eight_available,
            eight_detail,
        )
    )
    for key, _effort, label in QWEN_8BIT_THINKING:
        rows.append(_catalog_row(key, label, eight_model, eight_available, eight_detail))
    gemini_key = _gemini_key(config)
    gemini_model = config.summary.gemini_model
    rows.append(
        _catalog_row(
            "gemini",
            f"Gemini ({gemini_model})",
            gemini_model,
            bool(gemini_key),
            "" if gemini_key else "set GEMINI_API_KEY or GOOGLE_API_KEY",
        )
    )
    openai_key = _openai_key(config)
    openai_model = config.summary.openai_model
    rows.append(
        _catalog_row(
            "openai",
            f"OpenAI ({openai_model})",
            openai_model,
            bool(openai_key),
            "" if openai_key else "set OPENAI_API_KEY",
        )
    )
    deepseek_key = _deepseek_key(config)
    deepseek_model = _deepseek_model(config)
    rows.append(
        _catalog_row(
            "deepseek-v4-pro",
            "DeepSeek V4 Pro (reasoning high)",
            deepseek_model,
            bool(deepseek_key),
            "" if deepseek_key else "set DEEPSEEK_API_KEY",
        )
    )
    qwen_max_key = _qwen_max_key(config)
    qwen_max_model = _qwen_max_model(config)
    official = bool(_deepseek_key(config))
    rows.append(
        _catalog_row(
            "deepseek-v4.1-flash",
            "DeepSeek V4.1 Flash (DeepSeek, sees images)" if official else "DeepSeek V4.1 Flash (Alibaba Cloud, sees images)",
            DEEPSEEK_FLASH_MODEL if official else DASHSCOPE_FLASH_MODEL,
            official or bool(qwen_max_key),
            "" if official or qwen_max_key else "set DEEPSEEK_API_KEY or DASHSCOPE_API_KEY",
        )
    )
    rows.append(
        _catalog_row(
            "qwen3.8-max",
            _qwen_max_label(config),
            qwen_max_model,
            bool(qwen_max_key),
            "" if qwen_max_key else "set DASHSCOPE_API_KEY or QWEN_API_KEY",
        )
    )
    agy_model = _agy_summary_model(config)
    agy_ok = _agy_available(config)
    rows.append(
        _catalog_row(
            "agy-gemini",
            f"Gemini via agy ({agy_model})",
            agy_model,
            agy_ok,
            "" if agy_ok else "install and sign in to the agy CLI",
        )
    )
    return rows


def resolve_summary_backends(
    config: AppConfig,
    vision: object | None = None,
    selection: str | None = None,
    vision_8bit: object | None = None,
) -> list[SummaryBackend]:
    wanted = _wanted_keys(config, selection)
    available = {item.key: item for item in _available_backends(config, vision, vision_8bit)}
    chosen: list[SummaryBackend] = []
    missing: list[str] = []
    for key in wanted:
        backend = available.get(key)
        if backend is None:
            missing.append(key)
            continue
        chosen.append(backend)
    if missing:
        logger.warning("Summary backends skipped: %s", ", ".join(missing))
    return chosen


def normalize_summary_key(name: str | None) -> str:
    raw = (name or "").strip().lower()
    return _KEY_ALIASES.get(raw, raw)


def wants_local_qwen(selection: str | None) -> bool:
    key = normalize_summary_key(selection)
    return key in {"", "default", "all", "qwen"} or key.startswith("qwen-")


def wants_8bit_qwen(selection: str | None) -> bool:
    key = normalize_summary_key(selection)
    return key == "qwen-8bit" or key.startswith("qwen-8bit-")


def qwen_selection_required(selection: str | None) -> bool:
    key = normalize_summary_key(selection)
    return key == "qwen" or key.startswith("qwen-")


def _wanted_keys(config: AppConfig, selection: str | None) -> list[str]:
    raw = (selection or "").strip().lower()
    if raw in {"", "default"}:
        names = list(config.summary.providers)
        if "all" in names:
            names = list(BACKEND_ORDER)
    elif raw == "all":
        names = list(BACKEND_ORDER)
    else:
        names = [raw]
    wanted: list[str] = []
    for name in names:
        key = normalize_summary_key(name)
        if key in ALL_BACKEND_KEYS and key not in wanted:
            wanted.append(key)
    return wanted or list(DEFAULT_SUMMARY_PROVIDERS)


def _available_backends(
    config: AppConfig,
    vision: object | None,
    vision_8bit: object | None = None,
) -> list[SummaryBackend]:
    backends: list[SummaryBackend] = []
    backends.extend(_qwen_backends(config, vision))
    backends.extend(_qwen_8bit_backends(config, vision_8bit))
    gemini_key = _gemini_key(config)
    if gemini_key:
        model = config.summary.gemini_model
        backends.append(
            SummaryBackend(
                key="gemini",
                label=f"Gemini ({model})",
                model_id=model,
                max_source_chars=CLOUD_SOURCE_CHARS,
                generate_fn=lambda prompt, images, max_tokens, _key=gemini_key, _model=model: _gemini_generate(
                    _key, _model, prompt, images, max_tokens
                ),
            )
        )
    openai_key = _openai_key(config)
    if openai_key:
        model = config.summary.openai_model
        backends.append(
            SummaryBackend(
                key="openai",
                label=f"OpenAI ({model})",
                model_id=model,
                max_source_chars=CLOUD_SOURCE_CHARS,
                generate_fn=lambda prompt, images, max_tokens, _key=openai_key, _model=model: _openai_generate(
                    _key, _model, prompt, images, max_tokens
                ),
            )
        )
    deepseek_key = _deepseek_key(config)
    if deepseek_key:
        model = _deepseek_model(config)
        base_url = _deepseek_base_url(config)
        backends.append(
            SummaryBackend(
                key="deepseek-v4-pro",
                label="DeepSeek V4 Pro (reasoning high)",
                model_id=model,
                accepts_images=False,
                max_source_chars=CLOUD_SOURCE_CHARS,
                generate_fn=lambda prompt, images, max_tokens, _key=deepseek_key, _model=model, _base=base_url: (
                    _openai_compatible_generate(
                        api_key=_key,
                        model=_model,
                        prompt=prompt,
                        images=None,
                        max_tokens=max(int(max_tokens or 0), THINKING_MAX_TOKENS),
                        base_url=_base,
                        extra={
                            "reasoning_effort": "high",
                            "thinking": {"type": "enabled"},
                        },
                        timeout=300,
                        stream=True,
                    )
                ),
            )
        )
    qwen_max_key = _qwen_max_key(config)
    if deepseek_key:
        base_url = _deepseek_base_url(config)
        backends.append(
            SummaryBackend(
                key="deepseek-v4.1-flash",
                label="DeepSeek V4.1 Flash (DeepSeek)",
                model_id=DEEPSEEK_FLASH_MODEL,
                max_source_chars=CLOUD_SOURCE_CHARS,
                generate_fn=lambda prompt, images, max_tokens, _key=deepseek_key, _base=base_url: strip_think(
                    _openai_compatible_generate(
                        api_key=_key,
                        model=DEEPSEEK_FLASH_MODEL,
                        prompt=prompt,
                        images=images,
                        max_tokens=max(int(max_tokens or 0), THINKING_MAX_TOKENS),
                        base_url=_base,
                        extra={"thinking": {"type": "enabled"}},
                        timeout=300,
                        stream=True,
                    )
                ),
            )
        )
    elif qwen_max_key:
        base_url = _qwen_max_base_url(config)
        backends.append(
            SummaryBackend(
                key="deepseek-v4.1-flash",
                label="DeepSeek V4.1 Flash (Alibaba Cloud)",
                model_id=DASHSCOPE_FLASH_MODEL,
                max_source_chars=CLOUD_SOURCE_CHARS,
                generate_fn=lambda prompt, images, max_tokens, _key=qwen_max_key, _base=base_url: strip_think(
                    _openai_compatible_generate(
                        api_key=_key,
                        model=DASHSCOPE_FLASH_MODEL,
                        prompt=prompt,
                        images=images,
                        max_tokens=max(int(max_tokens or 0), THINKING_MAX_TOKENS),
                        base_url=_base,
                        extra={"enable_thinking": True},
                        timeout=300,
                        stream=True,
                    )
                ),
            )
        )
    if qwen_max_key:
        model = _qwen_max_model(config)
        base_url = _qwen_max_base_url(config)
        backends.append(
            SummaryBackend(
                key="qwen3.8-max",
                label=_qwen_max_label(config),
                model_id=model,
                max_source_chars=CLOUD_SOURCE_CHARS,
                generate_fn=lambda prompt, images, max_tokens, _key=qwen_max_key, _model=model, _base=base_url: (
                    _openai_compatible_generate(
                        api_key=_key,
                        model=_model,
                        prompt=prompt,
                        images=images,
                        max_tokens=max(int(max_tokens or 0), THINKING_MAX_TOKENS),
                        base_url=_base,
                        extra=_qwen_max_thinking(config),
                        timeout=300,
                        stream=True,
                    )
                ),
            )
        )
    if _agy_available(config):
        model = _agy_summary_model(config)
        backends.append(
            SummaryBackend(
                key="agy-gemini",
                label=f"Gemini via agy ({model})",
                model_id=model,
                max_source_chars=AGY_SOURCE_CHARS,  # the prompt travels as a CLI argument (macOS ARG_MAX ~1 MB)
                generate_fn=lambda prompt, images, max_tokens, _config=config, _model=model: _agy_generate(
                    _config, _model, prompt, images
                ),
            )
        )
    return backends


def strip_think(text: str) -> str:
    """Drop a reasoning block some models (DeepSeek on DashScope) put inline as <think>...</think>."""
    if not isinstance(text, str):
        return text
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def _agy_summary_model(config: AppConfig) -> str:
    return str(getattr(getattr(config, "agy", None), "summary_model", "") or "gemini-4-argon-medium")


def _agy_available(config: AppConfig) -> bool:
    import shutil

    return bool(shutil.which(str(getattr(getattr(config, "agy", None), "binary", "") or "agy")))


def _agy_generate(config: AppConfig, model: str, prompt: str, images: list[Path] | None) -> str:
    """Summarize through the sandboxed agy CLI; keyframes are copied in and read with read_file."""
    import shutil
    import tempfile

    from media_pipeline.asr.agy import agy_settings, run_agy

    settings = agy_settings(config)
    with tempfile.TemporaryDirectory(prefix="agy-summary-") as tmp:
        workspace = Path(tmp)
        copies: list[Path] = []
        for index, (path, _mime, _blob) in enumerate(_image_blobs(images)):
            target = workspace / f"frame{index:02d}{path.suffix.lower()}"
            shutil.copyfile(path, target)
            copies.append(target)
        if copies:
            listing = "\n".join(str(item) for item in copies)
            instructions = (
                "Before answering, call read_file on each of these keyframes from the video and use what they show. "
                "Do not run commands or read any other file.\n" + listing
            )
        else:
            instructions = "Answer directly from the text below. Do not call any tools."
        text = run_agy(
            f"{instructions}\n\n{prompt}",
            workspace=workspace,
            allowed_reads=set(copies),
            settings=settings,
            model=model,
        )
    if not text.strip():
        raise RuntimeError("agy returned an empty summary")
    return text.strip()


def _probe_qwen(config: AppConfig) -> tuple[bool, str]:
    if not config.analysis.enabled:
        return False, "analysis is disabled"
    from media_pipeline.visual.vlm import probe_vlm

    ok, detail = probe_vlm(config)
    if ok:
        return True, ""
    return False, detail or "local Qwen3.8 is not available"


def _probe_qwen_8bit(config: AppConfig) -> tuple[bool, str]:
    from media_pipeline.visual.vlm import probe_vlm_model

    model_id = _qwen_8bit_model(config)
    ok, detail = probe_vlm_model(model_id)
    if ok:
        return True, ""
    return False, detail or f"weights missing. hf download {model_id}"


def qwen_8bit_model_id(config: AppConfig) -> str:
    return str(getattr(config.summary, "qwen_8bit_model", "") or DEFAULT_QWEN_8BIT_MODEL)


def _qwen_8bit_model(config: AppConfig) -> str:
    return qwen_8bit_model_id(config)


def _qwen_backend(config: AppConfig, vision: object | None) -> SummaryBackend | None:
    backends = _qwen_backends(config, vision)
    return backends[0] if backends else None


def _qwen_backends(config: AppConfig, vision: object | None) -> list[SummaryBackend]:
    return _family_backends(
        vision,
        instruct=QWEN_INSTRUCT,
        thinking=QWEN_THINKING,
        model_id=str(getattr(vision, "model_id", "") or config.analysis.model),
    )


def _qwen_8bit_backends(config: AppConfig, vision: object | None) -> list[SummaryBackend]:
    return _family_backends(
        vision,
        instruct=QWEN_8BIT_INSTRUCT,
        thinking=QWEN_8BIT_THINKING,
        model_id=str(getattr(vision, "model_id", "") or _qwen_8bit_model(config)),
    )


def _family_backends(
    vision: object | None,
    *,
    instruct: tuple[str, str],
    thinking: tuple[tuple[str, str, str], ...],
    model_id: str,
) -> list[SummaryBackend]:
    if vision is None or getattr(vision, "name", "") == "none":
        return []
    if getattr(vision, "local", True) is False:
        # A cloud keyframe judge is not the local Qwen3.8-27B; use qwen3.8-max for cloud summaries.
        return []
    generate = getattr(vision, "generate", None)
    if not callable(generate):
        return []
    key, label = instruct
    backends = [
        SummaryBackend(
            key=key,
            label=label,
            model_id=model_id,
            local=True,
            generate_fn=_wrap_qwen_generate(generate, None),
        )
    ]
    for think_key, effort, think_label in thinking:
        backends.append(
            SummaryBackend(
                key=think_key,
                label=think_label,
                model_id=model_id,
                local=True,
                generate_fn=_wrap_qwen_generate(generate, effort),
            )
        )
    return backends


def _wrap_qwen_generate(generate: Callable[..., str], effort: str | None):
    thinking = effort is not None

    def _run(prompt: str, images: list[Path] | None = None, max_tokens: int | None = None) -> str:
        tokens = int(max_tokens) if max_tokens else 0
        if thinking:
            tokens = max(tokens, THINKING_MAX_TOKENS)
        kwargs = {"enable_thinking": thinking, "reasoning_effort": effort}
        try:
            return generate(prompt, images, tokens or None, **kwargs)
        except TypeError:
            return generate(prompt, images, tokens or None)

    return _run


def _catalog_row(key: str, label: str, model: str, available: bool, detail: str) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "model": model,
        "available": available,
        "detail": detail,
    }


def _gemini_key(config: AppConfig) -> str:
    return (
        config.summary.gemini_api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or ""
    ).strip()


def _openai_key(config: AppConfig) -> str:
    return (config.summary.openai_api_key or os.environ.get("OPENAI_API_KEY") or "").strip()


def _deepseek_key(config: AppConfig) -> str:
    return (config.summary.deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY") or "").strip()


def _deepseek_model(config: AppConfig) -> str:
    return str(getattr(config.summary, "deepseek_model", "") or "deepseek-v4-pro").strip()


def _deepseek_base_url(config: AppConfig) -> str:
    return str(
        getattr(config.summary, "deepseek_base_url", "") or DEFAULT_DEEPSEEK_BASE_URL
    ).strip().rstrip("/")


def _qwen_max_reasoning(config: AppConfig) -> str:
    from media_pipeline.config import normalize_qwen_reasoning

    return normalize_qwen_reasoning(getattr(config.summary, "qwen_max_reasoning", "xhigh"))


def _qwen_max_label(config: AppConfig) -> str:
    effort = _qwen_max_reasoning(config)
    return "Qwen3.8-Max (no thinking)" if effort == "none" else f"Qwen3.8-Max (reasoning {effort})"


def _qwen_max_thinking(config: AppConfig) -> dict[str, Any]:
    effort = _qwen_max_reasoning(config)
    if effort == "none":
        return {"enable_thinking": False}
    # Only reasoning_effort: DashScope rejects it together with thinking_budget.
    return {"enable_thinking": True, "reasoning_effort": effort}


def _qwen_max_key(config: AppConfig) -> str:
    return (
        config.summary.qwen_api_key
        or str(getattr(getattr(config, "dashscope", None), "api_key", "") or "")
        or os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("QWEN_API_KEY")
        or ""
    ).strip()


def _qwen_max_model(config: AppConfig) -> str:
    return str(getattr(config.summary, "qwen_max_model", "") or "qwen3.8-max").strip()


def _qwen_max_base_url(config: AppConfig) -> str:
    dash_base = str(getattr(getattr(config, "dashscope", None), "base_url", "") or "").strip().rstrip("/")
    if dash_base:
        root = dash_base.split("/compatible-mode")[0].split("/api/v1")[0]
        return f"{root}/compatible-mode/v1"
    return str(
        getattr(config.summary, "qwen_max_base_url", "") or DEFAULT_QWEN_MAX_BASE_URL
    ).strip().rstrip("/")


def _chat_completions_url(base_url: str) -> str:
    root = (base_url or "").strip().rstrip("/")
    if root.endswith("/chat/completions"):
        return root
    if root.endswith("/v1"):
        return f"{root}/chat/completions"
    return f"{root}/v1/chat/completions"


def _gemini_generate(
    api_key: str,
    model: str,
    prompt: str,
    images: list[Path] | None,
    max_tokens: int | None,
) -> str:
    parts: list[dict[str, Any]] = [{"text": prompt}]
    for path, mime, blob in _image_blobs(images):
        parts.append({"inline_data": {"mime_type": mime, "data": blob}})
        logger.info("Gemini summary attached %s", path.name)
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "maxOutputTokens": int(max_tokens or 2048),
            "temperature": 0.3,
        },
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    data = _json_post(url, payload, {"x-goog-api-key": api_key, "Content-Type": "application/json"})
    texts: list[str] = []
    for candidate in data.get("candidates") or []:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            text = str(part.get("text") or "").strip()
            if text:
                texts.append(text)
    if not texts:
        raise RuntimeError(_gemini_error(data) or "Gemini returned an empty summary")
    return "\n".join(texts)


def _openai_generate(
    api_key: str,
    model: str,
    prompt: str,
    images: list[Path] | None,
    max_tokens: int | None,
) -> str:
    return _openai_compatible_generate(
        api_key=api_key,
        model=model,
        prompt=prompt,
        images=images,
        max_tokens=max_tokens,
        base_url="https://api.openai.com",
    )


def _openai_compatible_generate(
    *,
    api_key: str,
    model: str,
    prompt: str,
    images: list[Path] | None,
    max_tokens: int | None,
    base_url: str,
    extra: dict[str, Any] | None = None,
    timeout: int = 120,
    stream: bool = False,
) -> str:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for path, mime, blob in _image_blobs(images):
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{blob}"},
            }
        )
        logger.info("%s summary attached %s", _host(base_url), path.name)
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": int(max_tokens or 2048),
        "temperature": 0.3,
        "messages": [{"role": "user", "content": content}],
    }
    if extra:
        payload.update(extra)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if stream:
        # Reasoning models can think for several minutes on a long transcript. Streamed, the timeout
        # bounds silence between chunks rather than the whole answer.
        text, reasoning = _stream_chat(_chat_completions_url(base_url), payload, headers, timeout=timeout)
        if reasoning and text:
            return f"<think>\n{reasoning}\n</think>\n\n{text}"
        if not text:
            raise RuntimeError(f"{_host(base_url)} returned an empty summary")
        return text
    data = _json_post(
        _chat_completions_url(base_url),
        payload,
        headers,
        timeout=timeout,
    )
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(str(data.get("error") or f"{_host(base_url)} returned an empty summary"))
    message = (choices[0] or {}).get("message") or {}
    text = str(message.get("content") or "").strip()
    reasoning = str(message.get("reasoning_content") or "").strip()
    if reasoning and text:
        return f"<think>\n{reasoning}\n</think>\n\n{text}"
    if reasoning:
        return f"<think>\n{reasoning}\n</think>\n\n"
    if not text:
        raise RuntimeError(f"{_host(base_url)} returned an empty summary")
    return text


def _image_blobs(images: list[Path] | None) -> list[tuple[Path, str, str]]:
    blobs: list[tuple[Path, str, str]] = []
    for item in images or []:
        path = Path(item)
        mime = _IMAGE_TYPES.get(path.suffix.lower())
        if not path.is_file() or not mime:
            continue
        size = path.stat().st_size
        if size <= 0 or size > _MAX_IMAGE_BYTES:
            logger.warning("Skipping summary image %s (%s bytes)", path.name, size)
            continue
        blobs.append((path, mime, base64.b64encode(path.read_bytes()).decode("ascii")))
    return blobs


def _json_post(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int = 120) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    for key, value in headers.items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = _redact_secrets(exc.read().decode("utf-8", errors="replace")[:400], headers)
        raise RuntimeError(f"{_host(url)} HTTP {exc.code}: {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"{_host(url)} request failed: {_redact_secrets(str(exc), headers)}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{_host(url)} returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{_host(url)} returned a non-object JSON payload")
    return data


def _stream_chat(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    timeout: int = 300,
    max_total_sec: float = 3600.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[str, str]:
    """POST an OpenAI-style chat request with stream=true and collect (content, reasoning) from SSE."""
    import time

    body = json.dumps({**payload, "stream": True}).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    for key, value in {**headers, "Accept": "text/event-stream"}.items():
        request.add_header(key, value)
    text: list[str] = []
    reasoning: list[str] = []
    started = time.monotonic()
    try:
        with opener(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    raise RuntimeError(f"{_host(url)} error: {_redact_secrets(str(chunk['error'])[:400], headers)}")
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        text.append(str(delta["content"]))
                    if delta.get("reasoning_content"):
                        reasoning.append(str(delta["reasoning_content"]))
                if time.monotonic() - started > max_total_sec:
                    raise RuntimeError(f"{_host(url)} summary exceeded {max_total_sec:.0f}s")
    except urllib.error.HTTPError as exc:
        detail = _redact_secrets(exc.read().decode("utf-8", errors="replace")[:400], headers)
        raise RuntimeError(f"{_host(url)} HTTP {exc.code}: {detail}") from exc
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"{_host(url)} request failed: {_redact_secrets(str(exc), headers)}") from exc
    return "".join(text).strip(), "".join(reasoning).strip()


def _host(url: str) -> str:
    return urlparse(url).hostname or "llm"


def _redact_secrets(text: str, headers: dict[str, str]) -> str:
    cleaned = text
    for key, value in headers.items():
        secret = value
        if key.lower() == "authorization" and secret.lower().startswith("bearer "):
            secret = secret.split(" ", 1)[1]
        if len(secret) >= 8 and secret in cleaned:
            cleaned = cleaned.replace(secret, "[redacted]")
    return cleaned


def _gemini_error(data: dict[str, Any]) -> str:
    error = data.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error)
    prompt = data.get("promptFeedback") or {}
    if prompt:
        return str(prompt)
    return ""
