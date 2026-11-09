"""Minimal Alibaba Cloud Model Studio (DashScope) client.

Covers the three calls the pipeline needs:

* temporary file upload (``/api/v1/uploads?action=getPolicy`` → OSS form POST → ``oss://`` URL)
* async file transcription (Fun-ASR / Paraformer) with speaker diarization
* OpenAI-compatible chat completions (text + images)

Only the standard library is used, matching the other cloud backends.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_HOST = "https://dashscope.aliyuncs.com"
_TERMINAL = {"SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN"}


class DashScopeError(RuntimeError):
    pass


def api_host(base_url: str | None) -> str:
    """Reduce any DashScope URL (compatible-mode or native) to its scheme://host root."""
    raw = (base_url or "").strip()
    if not raw:
        return DEFAULT_HOST
    parsed = urllib.parse.urlparse(raw if "://" in raw else f"https://{raw}")
    if not parsed.netloc:
        return DEFAULT_HOST
    return f"{parsed.scheme or 'https'}://{parsed.netloc}"


@dataclass
class DashScopeClient:
    api_key: str
    host: str = DEFAULT_HOST
    timeout: int = 120
    sleep: Callable[[float], None] = time.sleep
    opener: Callable[..., Any] = urllib.request.urlopen

    def __post_init__(self) -> None:
        self.api_key = (self.api_key or "").strip()
        self.host = api_host(self.host)
        if not self.api_key:
            raise DashScopeError("DashScope API key is missing. Set dashscope.api_key or DASHSCOPE_API_KEY.")

    # ------------------------------------------------------------------ upload
    def upload_file(self, path: Path, model: str) -> str:
        """Upload a local file to DashScope temporary storage (48h) and return its oss:// URL."""
        query = urllib.parse.urlencode({"action": "getPolicy", "model": model})
        policy = self._request("GET", f"{self.host}/api/v1/uploads?{query}")
        data = policy.get("data") or {}
        required = ("upload_host", "upload_dir", "oss_access_key_id", "signature", "policy")
        missing = [key for key in required if not data.get(key)]
        if missing:
            raise DashScopeError(f"Upload policy is missing {', '.join(missing)}")
        size = path.stat().st_size
        max_mb = _as_int(data.get("max_file_size_mb"))
        if max_mb and size > max_mb * 1024 * 1024:
            raise DashScopeError(f"{path.name} is {size / 1e6:.0f} MB; DashScope temporary upload limit is {max_mb} MB")
        key = f"{data['upload_dir']}/{uuid.uuid4().hex[:8]}-{path.name}"
        fields = {
            "OSSAccessKeyId": str(data["oss_access_key_id"]),
            "Signature": str(data["signature"]),
            "policy": str(data["policy"]),
            "key": key,
            "success_action_status": "200",
        }
        for name in ("x_oss_object_acl", "x_oss_forbid_overwrite"):
            if data.get(name) is not None:
                fields[name.replace("_", "-")] = str(data[name])
        body, content_type = _multipart(fields, "file", path)
        request = urllib.request.Request(
            str(data["upload_host"]),
            data=body,
            method="POST",
            headers={"Content-Type": content_type, "Content-Length": str(len(body))},
        )
        try:
            with self.opener(request, timeout=max(self.timeout, 600)) as response:
                response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise DashScopeError(f"Upload to DashScope storage failed: HTTP {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise DashScopeError(f"Upload to DashScope storage failed: {exc.reason}") from exc
        logger.info("Uploaded %s (%.1f MB) to DashScope temporary storage", path.name, size / 1e6)
        return f"oss://{key}"

    # ----------------------------------------------------------- transcription
    def submit_transcription(self, model: str, file_url: str, parameters: dict[str, Any]) -> str:
        headers = {"X-DashScope-Async": "enable"}
        if file_url.startswith("oss://"):
            headers["X-DashScope-OssResourceResolve"] = "enable"
        payload = {"model": model, "input": {"file_urls": [file_url]}, "parameters": parameters or {}}
        data = self._request(
            "POST",
            f"{self.host}/api/v1/services/audio/asr/transcription",
            payload,
            headers,
        )
        task_id = str((data.get("output") or {}).get("task_id") or "")
        if not task_id:
            raise DashScopeError(f"Transcription submit returned no task_id: {_brief(data)}")
        return task_id

    def wait_task(self, task_id: str, *, poll_sec: float = 3.0, timeout_sec: float = 3 * 3600) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_sec
        delay = poll_sec
        while True:
            data = self._request("GET", f"{self.host}/api/v1/tasks/{task_id}")
            output = data.get("output") or {}
            status = str(output.get("task_status") or "").upper()
            if status in _TERMINAL:
                return output
            if time.monotonic() > deadline:
                raise DashScopeError(f"DashScope task {task_id} still {status or 'pending'} after {timeout_sec:.0f}s")
            self.sleep(delay)
            delay = min(delay * 1.5, 15.0)

    def transcribe(
        self,
        model: str,
        file_url: str,
        parameters: dict[str, Any],
        *,
        poll_sec: float = 3.0,
    ) -> dict[str, Any]:
        """Run one file transcription to completion and return the result JSON for that file."""
        task_id = self.submit_transcription(model, file_url, parameters)
        logger.info("DashScope %s transcription task %s submitted", model, task_id)
        output = self.wait_task(task_id, poll_sec=poll_sec)
        status = str(output.get("task_status") or "").upper()
        results = output.get("results") or []
        first = results[0] if results else {}
        if status != "SUCCEEDED" or str(first.get("subtask_status") or "SUCCEEDED").upper() != "SUCCEEDED":
            message = first.get("message") or output.get("message") or output.get("code") or status
            raise DashScopeError(f"DashScope transcription failed: {message}")
        url = first.get("transcription_url")
        if not url:
            raise DashScopeError(f"DashScope transcription returned no transcription_url: {_brief(output)}")
        try:
            with self.opener(urllib.request.Request(str(url)), timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            raise DashScopeError(f"Could not download DashScope transcription result: {exc}") from exc

    # -------------------------------------------------------------------- chat
    def chat(
        self,
        model: str,
        prompt: str,
        images: list[Path] | None = None,
        *,
        max_tokens: int = 1024,
        extra: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> str:
        content: list[dict[str, Any]] = []
        for path in images or []:
            mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
            blob = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{blob}"}})
        content.append({"type": "text", "text": prompt})
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": int(max_tokens),
            "messages": [{"role": "user", "content": content}],
        }
        if extra:
            payload.update(extra)
        data = self._request(
            "POST",
            f"{self.host}/compatible-mode/v1/chat/completions",
            payload,
            timeout=timeout,
        )
        choices = data.get("choices") or []
        message = (choices[0] or {}).get("message") if choices else None
        text = str((message or {}).get("content") or "").strip()
        if not text:
            raise DashScopeError(f"{model} returned an empty response: {_brief(data)}")
        return text

    # ---------------------------------------------------------------- plumbing
    def _request(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: int | None = None,
        retries: int = 3,
    ) -> dict[str, Any]:
        merged = {"Authorization": f"Bearer {self.api_key}"}
        if payload is not None:
            merged["Content-Type"] = "application/json"
        merged.update(headers or {})
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        attempt = 0
        while True:
            attempt += 1
            request = urllib.request.Request(url, data=body, method=method, headers=merged)
            try:
                with self.opener(request, timeout=timeout or self.timeout) as response:
                    return json.loads(response.read().decode("utf-8") or "{}")
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code == 429 or exc.code >= 500
                if retryable and attempt < retries:
                    self.sleep(2.0 * attempt)
                    continue
                raise DashScopeError(
                    f"DashScope HTTP {exc.code} for {urllib.parse.urlparse(url).path}: {_redact(detail, self.api_key)[:500]}"
                ) from exc
            except urllib.error.URLError as exc:
                if attempt < retries:
                    self.sleep(2.0 * attempt)
                    continue
                raise DashScopeError(f"DashScope request failed: {exc.reason}") from exc


def resolve_api_key(config: Any) -> str:
    """Shared DashScope key: dashscope.api_key → summary.qwen_api_key → env."""
    dash = getattr(config, "dashscope", None)
    summary = getattr(config, "summary", None)
    return (
        str(getattr(dash, "api_key", "") or "")
        or str(getattr(summary, "qwen_api_key", "") or "")
        or os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("QWEN_API_KEY")
        or ""
    ).strip()


def resolve_host(config: Any) -> str:
    dash = getattr(config, "dashscope", None)
    summary = getattr(config, "summary", None)
    return api_host(
        str(getattr(dash, "base_url", "") or "") or str(getattr(summary, "qwen_max_base_url", "") or "")
    )


def client_from_config(config: Any) -> DashScopeClient:
    return DashScopeClient(api_key=resolve_api_key(config), host=resolve_host(config))


def _multipart(fields: dict[str, str], file_field: str, path: Path) -> tuple[bytes, str]:
    boundary = f"----media-pipeline-{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode("utf-8")
        )
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    parts.append(
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{path.name}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode("utf-8")
    )
    parts.append(path.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _as_int(value: object) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def _brief(data: object) -> str:
    return json.dumps(data, ensure_ascii=False)[:300]


def _redact(text: str, secret: str) -> str:
    return text.replace(secret, "***") if secret else text
