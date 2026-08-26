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

from media_pipeline.config import AppConfig

logger = logging.getLogger(__name__)

BACKEND_ORDER = ("qwen", "gemini", "openai")
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
            "qwen",
            "Qwen3.8 (local, free)",
            qwen_model,
            qwen_available,
            qwen_detail,
        )
    )
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
    return rows


def resolve_summary_backends(
    config: AppConfig,
    vision: object | None = None,
    selection: str | None = None,
) -> list[SummaryBackend]:
    wanted = _wanted_keys(config, selection)
    available = {item.key: item for item in _available_backends(config, vision)}
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
        key = "qwen" if name in {"qwen", "qwen3.8", "local"} else name
        if key in BACKEND_ORDER and key not in wanted:
            wanted.append(key)
    return wanted or ["qwen"]


def _available_backends(config: AppConfig, vision: object | None) -> list[SummaryBackend]:
    backends: list[SummaryBackend] = []
    qwen = _qwen_backend(config, vision)
    if qwen is not None:
        backends.append(qwen)
    gemini_key = _gemini_key(config)
    if gemini_key:
        model = config.summary.gemini_model
        backends.append(
            SummaryBackend(
                key="gemini",
                label=f"Gemini ({model})",
                model_id=model,
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
                generate_fn=lambda prompt, images, max_tokens, _key=openai_key, _model=model: _openai_generate(
                    _key, _model, prompt, images, max_tokens
                ),
            )
        )
    return backends


def _probe_qwen(config: AppConfig) -> tuple[bool, str]:
    if not config.analysis.enabled:
        return False, "analysis is disabled"
    from media_pipeline.visual.vlm import probe_vlm

    ok, detail = probe_vlm(config)
    if ok:
        return True, ""
    return False, detail or "local Qwen3.8 is not available"


def _qwen_backend(config: AppConfig, vision: object | None) -> SummaryBackend | None:
    if vision is None or getattr(vision, "name", "") == "none":
        return None
    generate = getattr(vision, "generate", None)
    if not callable(generate):
        return None
    model_id = str(getattr(vision, "model_id", "") or config.analysis.model)
    return SummaryBackend(
        key="qwen",
        label="Qwen3.8 (local)",
        model_id=model_id,
        local=True,
        generate_fn=generate,
    )


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
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for path, mime, blob in _image_blobs(images):
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{blob}"},
            }
        )
        logger.info("OpenAI summary attached %s", path.name)
    payload = {
        "model": model,
        "max_tokens": int(max_tokens or 2048),
        "temperature": 0.3,
        "messages": [{"role": "user", "content": content}],
    }
    data = _json_post(
        "https://api.openai.com/v1/chat/completions",
        payload,
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(str(data.get("error") or "OpenAI returned an empty summary"))
    message = (choices[0] or {}).get("message") or {}
    text = str(message.get("content") or "").strip()
    if not text:
        raise RuntimeError("OpenAI returned an empty summary")
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
