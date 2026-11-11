"""Threads (threads.com / threads.net) posts: text, images, carousels and videos.

Post pages are public. The page embeds data in ``<script type="application/json" data-sjs>`` blocks;
the post is the object whose ``code`` is the shortcode and that carries a ``caption``. Those blocks
also hold unrelated "more posts" recommendations, so the target is matched by code, never by position.
When the author continues the post by replying to themself, the continuation sits in the same
``thread_items`` list and is merged into one note.

Ids are prefixed (``th_<code>``): Threads shortcodes are 11 characters, like YouTube video ids.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

from media_pipeline.config import AppConfig
from media_pipeline.media import MediaError
from media_pipeline.models import VideoMetadata

logger = logging.getLogger(__name__)

PLATFORM = "Threads"
ID_PREFIX = "th_"
_HOSTS = {"threads.com", "threads.net"}
_POST_PATH = re.compile(r"^/@([^/]+)/post/([A-Za-z0-9_-]+)")
_SHARE_PATH = re.compile(r"^/share/([A-Za-z0-9_-]+)")
_JSON_SCRIPT = re.compile(r'<script type="application/json"[^>]*\bdata-sjs\b[^>]*>(.*?)</script>', re.S)
_CACHE: dict[str, "ThreadsPost"] = {}


class ThreadsError(MediaError):
    """Reported at its pipeline stage like any other media failure."""


@dataclass
class ThreadsPost:
    code: str
    url: str
    username: str
    author: str
    text: str
    published: str
    media_kind: str  # "video" (transcribe it) or "image" (text and images)
    image_urls: list[str] = field(default_factory=list)
    video_url: str = ""
    thumbnail_url: str = ""
    duration: float | None = None

    @property
    def video_id(self) -> str:
        return ID_PREFIX + self.code


def is_threads_host(host: str) -> bool:
    host = (host or "").lower().removeprefix("www.")
    return host in _HOSTS or any(host.endswith("." + item) for item in _HOSTS)


def parse_threads_ref(url: str) -> tuple[str, str]:
    """(platform, video_id). Share links get a provisional id until the redirect is followed."""
    parsed = urlparse((url or "").strip())
    if not is_threads_host(parsed.hostname or ""):
        raise ValueError(url)
    post = _POST_PATH.match(parsed.path or "")
    if post:
        return PLATFORM, ID_PREFIX + post.group(2)
    share = _SHARE_PATH.match(parsed.path or "")
    if share:
        return PLATFORM, f"{ID_PREFIX}share_{share.group(1)}"
    raise ValueError(url)


def canonicalize_threads_url(url: str) -> str:
    """Post URLs lose their tracking query (xmt, slof); share links are kept for the redirect."""
    parsed = urlparse((url or "").strip())
    post = _POST_PATH.match(parsed.path or "")
    if post and is_threads_host(parsed.hostname or ""):
        return f"https://www.threads.com/@{post.group(1)}/post/{post.group(2)}"
    return url


def fetch_threads_metadata(url: str, asr_model: str, config: AppConfig) -> VideoMetadata:
    post = extract_threads_post(url, config)
    first_line = next((line.strip() for line in post.text.splitlines() if line.strip()), "")
    title = first_line[:80] + ("…" if len(first_line) > 80 else "") if first_line else f"@{post.username} on Threads"
    return VideoMetadata(
        url=post.url,
        title=title,
        platform=PLATFORM,
        author=post.author,
        video_id=post.video_id,
        duration=post.duration,
        published=post.published,
        description=post.text,
        thumbnail_url=post.thumbnail_url,
        asr_model=asr_model,
        media_kind=post.media_kind,
    )


def download_threads_post(url: str, video_id: str, dest_dir: Path, config: AppConfig) -> Path:
    """A video post → <id>.mp4 for ASR; otherwise a folder of images (empty for text-only posts)."""
    post = extract_threads_post(url, config)
    video_id = post.video_id or video_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    if post.media_kind == "video":
        target = dest_dir / f"{video_id}.mp4"
        if not (target.exists() and target.stat().st_size > 0):
            _download(post.video_url, target, stage="downloading")
        return target
    folder = dest_dir / video_id
    folder.mkdir(parents=True, exist_ok=True)
    for index, image_url in enumerate(post.image_urls, start=1):
        target = folder / f"{index:02d}.jpg"
        if not (target.exists() and target.stat().st_size > 0):
            _download(image_url, target, stage="downloading")
    return folder


def extract_threads_post(url: str, config: AppConfig) -> ThreadsPost:
    cached = _CACHE.get(url)
    if cached is not None:
        return cached
    html, final_url = _fetch(url)
    post = parse_threads_page(html, final_url or url)
    _CACHE[url] = post
    _CACHE[post.url] = post
    return post


def parse_threads_page(html: str, page_url: str) -> ThreadsPost:
    parsed = urlparse(page_url)
    match = _POST_PATH.match(parsed.path or "")
    if not match:
        raise ThreadsError("fetching_metadata", f"Threads did not open a post page (ended at {page_url})")
    username, code = match.group(1), match.group(2)
    blocks = [_loads(body) for body in _JSON_SCRIPT.findall(html or "") if code in body]
    target = next((node for block in blocks for node in _posts(block) if node.get("code") == code), None)
    if target is None:
        raise ThreadsError(
            "fetching_metadata",
            "Could not find the post in the Threads page; the page format may have changed or the post needs a login.",
        )
    chain = _author_chain(blocks, target)
    user = target.get("user") or {}
    username = str(user.get("username") or username)
    text = "\n\n".join(caption for caption in (_caption(post) for post in chain) if caption)
    images, videos = [], []
    for post in chain:
        post_images, post_videos = _media(post)
        images.extend(post_images)
        videos.extend(post_videos)
    first_images, first_videos = _media(target)
    is_video = bool(first_videos) and not first_images
    return ThreadsPost(
        code=code,
        url=f"https://www.threads.com/@{username}/post/{code}",
        username=username,
        author=str(user.get("full_name") or username),
        text=text,
        published=_published(target.get("taken_at")),
        media_kind="video" if is_video else "image",
        image_urls=[] if is_video else images,
        video_url=first_videos[0][0] if is_video else "",
        thumbnail_url=images[0] if images else str(user.get("profile_pic_url") or ""),
        duration=first_videos[0][1] if is_video else None,
    )


def _posts(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        if isinstance(node.get("code"), str) and isinstance(node.get("caption"), dict):
            yield node
        for value in node.values():
            yield from _posts(value)
    elif isinstance(node, list):
        for value in node:
            yield from _posts(value)


def _thread_lists(node: Any) -> Iterator[list[Any]]:
    if isinstance(node, dict):
        if isinstance(node.get("thread_items"), list):
            yield node["thread_items"]
        for value in node.values():
            yield from _thread_lists(value)
    elif isinstance(node, list):
        for value in node:
            yield from _thread_lists(value)


def _author_chain(blocks: list[Any], target: dict[str, Any]) -> list[dict[str, Any]]:
    """The target plus the author's own consecutive continuation in the same thread."""
    author = (target.get("user") or {}).get("pk") or (target.get("user") or {}).get("username")
    for block in blocks:
        for items in _thread_lists(block):
            posts = [item.get("post") for item in items if isinstance(item, dict) and isinstance(item.get("post"), dict)]
            codes = [post.get("code") for post in posts]
            if target.get("code") not in codes:
                continue
            chain = []
            for post in posts[codes.index(target.get("code")):]:
                user = post.get("user") or {}
                if (user.get("pk") or user.get("username")) != author:
                    break
                chain.append(post)
            return chain or [target]
    return [target]


def _caption(post: dict[str, Any]) -> str:
    return str((post.get("caption") or {}).get("text") or "").strip()


def _media(post: dict[str, Any]) -> tuple[list[str], list[tuple[str, float | None]]]:
    items = post.get("carousel_media") or [post]
    images: list[str] = []
    videos: list[tuple[str, float | None]] = []
    for item in items:
        versions = item.get("video_versions") or []
        if versions:
            best = versions[0]
            duration = item.get("video_duration")
            videos.append((str(best.get("url") or ""), float(duration) if duration else None))
            continue
        candidates = (item.get("image_versions2") or {}).get("candidates") or []
        if candidates:
            best = max(candidates, key=lambda c: int(c.get("width") or 0) * int(c.get("height") or 0))
            images.append(str(best.get("url") or ""))
    return [url for url in images if url], [(url, d) for url, d in videos if url]


def _published(stamp: Any) -> str:
    try:
        return datetime.fromtimestamp(int(stamp), tz=timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return ""


def _loads(body: str) -> Any:
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def _fetch(url: str) -> tuple[str, str]:
    try:
        from curl_cffi import requests as cf_requests

        response = cf_requests.get(
            url,
            impersonate="chrome",
            timeout=60,
            allow_redirects=True,
            headers={"Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
        )
    except Exception as exc:
        raise ThreadsError("fetching_metadata", f"Threads request failed: {exc}") from exc
    if response.status_code >= 400:
        raise ThreadsError("fetching_metadata", f"Threads request failed: HTTP {response.status_code}")
    return response.text, str(getattr(response, "url", "") or url)


def _download(url: str, target: Path, *, stage: str) -> None:
    if not url:
        raise ThreadsError(stage, "Threads media URL was not found")
    from curl_cffi import requests as cf_requests

    partial = target.with_name(target.name + ".part")
    try:
        response = cf_requests.get(url, impersonate="chrome", timeout=120, stream=True)
        if response.status_code >= 400:
            raise ThreadsError(stage, f"Threads media download failed: HTTP {response.status_code}")
        with partial.open("wb") as handle:
            for chunk in response.iter_content():
                handle.write(chunk)
    except ThreadsError:
        raise
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise ThreadsError(stage, f"Threads media download failed: {exc}") from exc
    partial.replace(target)
