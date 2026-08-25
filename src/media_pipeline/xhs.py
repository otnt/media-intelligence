from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import yaml

from media_pipeline.config import AppConfig
from media_pipeline.models import VideoMetadata

logger = logging.getLogger(__name__)

_NOTE_ID_RE = re.compile(r"^[0-9a-f]{24}$", re.IGNORECASE)
_INITIAL_STATE = re.compile(
    r"window\.__INITIAL_STATE__\s*=\s*(\{.*)\s*</script>",
    re.DOTALL | re.IGNORECASE,
)
_YAML_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass
class XhsPost:
    note_id: str
    url: str
    title: str
    author: str
    description: str
    published: str
    media_kind: str
    thumbnail_url: str = ""
    duration: float | None = None
    download_urls: list[str] = field(default_factory=list)


_POST_CACHE: dict[str, XhsPost] = {}


def is_xhs_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    return _is_xhs_host(host)


def _is_xhs_host(host: str) -> bool:
    name = (host or "").lower().removeprefix("www.")
    return (
        name in {"xiaohongshu.com", "xhslink.com", "rednote.com"}
        or name.endswith(".xiaohongshu.com")
        or name.endswith(".rednote.com")
        or name.endswith(".xhslink.com")
    )


def _page_origin(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if "rednote.com" in host:
        return "https://www.rednote.com"
    return "https://www.xiaohongshu.com"


def parse_xhs_ref(url: str) -> tuple[str, str]:
    parsed = urlparse((url or "").strip())
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if not _is_xhs_host(host):
        raise ValueError(url)
    parts = [item for item in (parsed.path or "").split("/") if item]
    if host == "xhslink.com" or host.endswith(".xhslink.com"):
        if len(parts) >= 2 and parts[0] == "o":
            return "Xiaohongshu", parts[1]
        if parts:
            return "Xiaohongshu", parts[-1]
        raise ValueError(url)
    if len(parts) >= 2 and parts[0] == "explore" and _NOTE_ID_RE.fullmatch(parts[1]):
        return "Xiaohongshu", parts[1]
    if len(parts) >= 3 and parts[0] == "discovery" and parts[1] == "item" and _NOTE_ID_RE.fullmatch(parts[2]):
        return "Xiaohongshu", parts[2]
    if len(parts) >= 2 and parts[0] == "search_result" and _NOTE_ID_RE.fullmatch(parts[1]):
        return "Xiaohongshu", parts[1]
    if len(parts) >= 4 and parts[0] == "user" and parts[1] == "profile" and _NOTE_ID_RE.fullmatch(parts[3]):
        return "Xiaohongshu", parts[3]
    raise ValueError(url)


def canonicalize_xhs_url(url: str, note_id: str = "") -> str:
    if not (note_id and _NOTE_ID_RE.fullmatch(note_id)):
        return url
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if host == "xhslink.com" or host.endswith(".xhslink.com"):
        return url
    query = parse_qs(parsed.query)
    params: dict[str, str] = {}
    if token := (query.get("xsec_token") or [""])[0]:
        params["xsec_token"] = token
    if source := (query.get("xsec_source") or [""])[0]:
        params["xsec_source"] = source
    base = f"{_page_origin(url)}/explore/{note_id}"
    if params:
        return f"{base}?{urlencode(params)}"
    return base


def fetch_xhs_metadata(url: str, asr_model: str, config: AppConfig) -> VideoMetadata:
    post = extract_xhs_post(url, config)
    return VideoMetadata(
        url=post.url,
        title=post.title or post.note_id,
        platform="Xiaohongshu",
        author=post.author,
        video_id=post.note_id,
        duration=post.duration,
        published=post.published,
        description=post.description,
        thumbnail_url=post.thumbnail_url,
        asr_model=asr_model,
        media_kind=post.media_kind,
    )


def download_xhs_post(url: str, note_id: str, dest_dir: Path, config: AppConfig) -> Path:
    post = extract_xhs_post(url, config)
    note_id = post.note_id or note_id
    if post.media_kind == "video":
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / f"{note_id}.mp4"
        if target.exists() and target.stat().st_size > 0:
            return target
        if not post.download_urls:
            raise _media_error("downloading", "Xiaohongshu video URL was not found")
        _download_first(post.download_urls, target, config, origin=_page_origin(url))
        return target
    folder = dest_dir / note_id
    existing = find_xhs_images(folder)
    if existing:
        return folder
    if not post.download_urls:
        raise _media_error("downloading", "Xiaohongshu image URLs were not found")
    folder.mkdir(parents=True, exist_ok=True)
    for index, media_url in enumerate(post.download_urls, start=1):
        dest = folder / f"{index:02d}.jpg"
        _download_file(media_url, dest, config, origin=_page_origin(url))
    return folder


def extract_xhs_post(url: str, config: AppConfig) -> XhsPost:
    cached = _cached_post(url)
    if cached is not None:
        return cached
    via_lib = _extract_with_xhs_downloader(url)
    if via_lib is not None and via_lib.download_urls:
        post = via_lib
    else:
        post = _extract_with_curl(url, config)
    _remember(url, post)
    return post


def find_xhs_images(directory: Path) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    files = [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".heic"}
    ]
    return [path for path in files if path.stat().st_size > 0]


def render_image_post(video_id: str, description: str, filenames: list[str]) -> str:
    lines = ["## Post", ""]
    text = (description or "").strip()
    if text:
        lines.append(text)
        lines.append("")
    for name in filenames:
        lines.append(f"![[attachments/{video_id}/{name}]]")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _cached_post(url: str) -> XhsPost | None:
    if url in _POST_CACHE:
        return _POST_CACHE[url]
    try:
        _, note_id = parse_xhs_ref(url)
    except ValueError:
        return None
    return _POST_CACHE.get(note_id)


def _remember(url: str, post: XhsPost) -> None:
    if not post.download_urls:
        return
    _POST_CACHE[url] = post
    if post.url:
        _POST_CACHE[post.url] = post
    if post.note_id:
        _POST_CACHE[post.note_id] = post


def _extract_with_xhs_downloader(url: str) -> XhsPost | None:
    """Use JoeanAmier's XHS-Downloader when the `source.XHS` package is importable."""
    try:
        from source import XHS  # type: ignore
    except ImportError:
        return None
    import asyncio

    async def _run() -> list[dict]:
        async with XHS(download_record=False, record_data=False, image_format="JPEG") as xhs:
            return await xhs.extract(url, download=False)

    try:
        rows = asyncio.run(_run())
    except Exception as exc:
        logger.warning("XHS-Downloader extract failed, falling back to curl-cffi: %s", exc)
        return None
    if not rows:
        return None
    row = next((item for item in rows if isinstance(item, dict) and item), None)
    if not row:
        return None
    return _post_from_xhs_row(row, url)


def _post_from_xhs_row(row: dict[str, Any], fallback_url: str) -> XhsPost:
    kind_raw = str(row.get("作品类型") or row.get("type") or "")
    media_kind = "video" if kind_raw in {"视频", "video"} else "image"
    urls = row.get("下载地址") or []
    if isinstance(urls, str):
        urls = [item for item in urls.split() if item]
    published = str(row.get("发布时间") or "").replace("_", " ")
    if " " in published:
        published = published.split(" ")[0]
    note_id = str(row.get("作品ID") or "")
    page_url = str(row.get("作品链接") or canonicalize_xhs_url(fallback_url, note_id))
    return XhsPost(
        note_id=note_id,
        url=page_url,
        title=str(row.get("作品标题") or note_id),
        author=str(row.get("作者昵称") or ""),
        description=str(row.get("作品描述") or ""),
        published=published,
        media_kind=media_kind,
        download_urls=[_format_url(str(item)) for item in urls if item],
    )


def _extract_with_curl(url: str, config: AppConfig) -> XhsPost:
    _require_xsec_token(url)
    html, final_url = _fetch_html(url, config)
    if "/404" in (final_url or "") or "error_code=300031" in (final_url or ""):
        raise _media_error(
            "fetching_metadata",
            "Xiaohongshu/RedNote returned 404 for this note. The xsec_token is missing or expired. Click ✨ Extract on the Explore card again.",
        )
    note = _note_from_html(html)
    if not note:
        raise _media_error(
            "fetching_metadata",
            "Could not parse Xiaohongshu note data. Log into Xiaohongshu or RedNote in Chrome so cookies can be read.",
        )
    note_id = str(note.get("noteId") or parse_xhs_ref(final_url)[1])
    image_list = note.get("imageList") or []
    note_type = str(note.get("type") or "")
    if note_type == "video" and len(image_list) <= 1:
        media_kind = "video"
        download_urls = _video_urls(note)
    else:
        media_kind = "image"
        download_urls = _image_urls(image_list)
    title = str(note.get("title") or "").strip() or note_id
    user = note.get("user") or {}
    author = str(user.get("nickname") or user.get("nickName") or "")
    thumbs = _image_urls(image_list[:1])
    return XhsPost(
        note_id=note_id,
        url=canonicalize_xhs_url(final_url, note_id),
        title=title,
        author=author,
        description=str(note.get("desc") or ""),
        published=_published(note.get("time")),
        media_kind=media_kind,
        thumbnail_url=thumbs[0] if thumbs else "",
        duration=_duration(note) if media_kind == "video" else None,
        download_urls=download_urls,
    )


def _fetch_html(url: str, config: AppConfig) -> tuple[str, str]:
    response = _http_get(url, config)
    if response.status_code >= 400 or not (response.text or "").strip():
        raise _media_error("fetching_metadata", f"Xiaohongshu request failed: HTTP {response.status_code}")
    return response.text, str(getattr(response, "url", "") or url)


def _http_get(
    url: str,
    config: AppConfig,
    *,
    stream: bool = False,
    stage: str = "fetching_metadata",
    origin: str = "",
):
    try:
        from curl_cffi import requests as cf_requests
    except ImportError as exc:
        raise _media_error(
            stage,
            "curl-cffi is required for Xiaohongshu. uv pip install curl-cffi",
        ) from exc
    cookies = _browser_cookies(config, url)
    site = origin or _page_origin(url)
    try:
        return cf_requests.get(
            url,
            impersonate="chrome",
            timeout=60,
            allow_redirects=True,
            cookies=cookies or None,
            stream=stream,
            headers={
                "Accept": "*/*",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Referer": f"{site}/",
                "Origin": site,
            },
        )
    except Exception as exc:
        raise _media_error(stage, f"Xiaohongshu request failed: {exc}") from exc


def _download_first(urls: list[str], dest: Path, config: AppConfig, origin: str = "") -> None:
    errors: list[str] = []
    for media_url in urls:
        try:
            _download_file(media_url, dest, config, origin=origin)
            return
        except Exception as exc:
            errors.append(str(exc))
            if dest.exists():
                dest.unlink()
    detail = errors[-1] if errors else "no URLs"
    raise _media_error("downloading", f"Failed to download Xiaohongshu video: {detail}")


def _download_file(url: str, dest: Path, config: AppConfig, origin: str = "") -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = _http_get(url, config, stream=True, stage="downloading", origin=origin)
    if response.status_code >= 400:
        raise _media_error("downloading", f"Failed to download {url}: HTTP {response.status_code}")
    ctype = str(response.headers.get("content-type") or "").lower()
    if "text/html" in ctype:
        raise _media_error("downloading", f"Xiaohongshu returned HTML instead of media for {url}")
    with dest.open("wb") as handle:
        iterator = getattr(response, "iter_content", None)
        if callable(iterator):
            for chunk in iterator(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        else:
            handle.write(response.content or b"")
    if dest.stat().st_size <= 0:
        dest.unlink(missing_ok=True)
        raise _media_error("downloading", f"Downloaded empty file from {url}")


def _require_xsec_token(url: str) -> None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if "xhslink.com" in host:
        return
    if (parse_qs(parsed.query).get("xsec_token") or [""])[0]:
        return
    raise _media_error(
        "fetching_metadata",
        "This Xiaohongshu/RedNote URL is missing xsec_token, so the note cannot be opened. Click ✨ Extract on the Explore card.",
    )


def _browser_cookies(config: AppConfig, url: str = "") -> dict[str, str]:
    browser = (config.download.cookies_from_browser or "").strip()
    if not browser:
        return {}
    try:
        from yt_dlp.cookies import extract_cookies_from_browser

        jar = extract_cookies_from_browser(browser)
    except Exception as exc:
        logger.warning("Could not read %s cookies for Xiaohongshu: %s", browser, exc)
        return {}
    cookies: dict[str, str] = {}
    for cookie in jar:
        domain = str(getattr(cookie, "domain", "") or "").lower()
        if _cookie_matches_url(domain, url):
            cookies[cookie.name] = cookie.value
    return cookies


def _cookie_matches_url(domain: str, url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if "rednote.com" in host:
        return "rednote" in domain or "xhscdn" in domain
    if "xiaohongshu.com" in host or "xhslink.com" in host:
        return "xiaohongshu" in domain or "xhslink" in domain or "xhscdn" in domain or domain.endswith(".xhs.com")
    return "xiaohongshu" in domain or "xhscdn" in domain or "xhslink" in domain or "rednote" in domain or "xhs" in domain


def _note_from_html(html: str) -> dict[str, Any] | None:
    match = _INITIAL_STATE.search(html or "")
    if not match:
        return None
    raw = match.group(1).strip().rstrip(";")
    cleaned = _YAML_ILLEGAL.sub("", raw)
    cleaned = cleaned.replace("undefined", "null").replace("new Map([])", "[]")
    data = _parse_state(cleaned)
    if not isinstance(data, dict):
        return None
    return _note_from_state(data)


def _parse_state(cleaned: str) -> Any:
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            return yaml.safe_load(cleaned)
        except yaml.YAMLError:
            return None


def _note_from_state(data: dict[str, Any]) -> dict[str, Any] | None:
    phone = ((data.get("noteData") or {}).get("data") or {}).get("noteData")
    if isinstance(phone, dict) and phone.get("noteId"):
        return phone
    detail = (data.get("note") or {}).get("noteDetailMap") or {}
    if isinstance(detail, dict) and detail:
        last = list(detail.values())[-1]
        if isinstance(last, dict):
            note = last.get("note") if isinstance(last.get("note"), dict) else last
            if isinstance(note, dict) and note.get("noteId"):
                return note
    return None


def _video_urls(note: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    video = note.get("video") or {}
    consumer = video.get("consumer") or {}
    key = consumer.get("originVideoKey")
    if key:
        urls.append(_format_url(f"https://sns-video-bd.xhscdn.com/{key}"))
    stream = (video.get("media") or {}).get("stream") or {}
    items: list[dict[str, Any]] = []
    if isinstance(stream, dict):
        for value in stream.values():
            if isinstance(value, list):
                items.extend(item for item in value if isinstance(item, dict))
    items.sort(
        key=lambda item: (
            int(item.get("height") or 0),
            int(item.get("videoBitrate") or item.get("bitrate") or 0),
        )
    )
    if items:
        best = items[-1]
        backups = best.get("backupUrls") or []
        if backups:
            urls.append(_format_url(str(backups[0])))
        master = best.get("masterUrl")
        if master:
            urls.append(_format_url(str(master)))
    seen: set[str] = set()
    unique: list[str] = []
    for item in urls:
        if item and item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _image_urls(image_list: list[Any]) -> list[str]:
    urls: list[str] = []
    for item in image_list:
        if not isinstance(item, dict):
            continue
        token = _image_token(str(item.get("urlDefault") or item.get("url") or ""))
        if token:
            urls.append(f"https://ci.xiaohongshu.com/{token}?imageView2/format/jpeg")
    return urls


def _image_token(url: str) -> str:
    url = _format_url(url)
    if not url:
        return ""
    parts = url.split("/")
    if len(parts) < 6:
        return ""
    return "/".join(parts[5:]).split("!")[0]


def _duration(note: dict[str, Any]) -> float | None:
    video = note.get("video") or {}
    capa = video.get("capa") or {}
    value = capa.get("duration") if capa.get("duration") is not None else video.get("duration")
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    if seconds > 10_000:
        seconds /= 1000
    return seconds


def _published(stamp: Any) -> str:
    try:
        millis = int(stamp)
    except (TypeError, ValueError):
        return ""
    if millis <= 0:
        return ""
    return datetime.fromtimestamp(millis / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _format_url(url: str) -> str:
    if not url:
        return ""
    try:
        return bytes(url, "utf-8").decode("unicode_escape")
    except Exception:
        return url


def _media_error(stage: str, message: str):
    from media_pipeline.media import MediaError

    return MediaError(stage, message)
