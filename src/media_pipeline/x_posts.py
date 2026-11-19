"""X (Twitter) posts and X Articles: text, the quoted post, images, video.

X shows nothing to logged-out readers, so the post is read through FxTwitter's public API
(``api.fxtwitter.com``, the link-preview service), which returns the whole post: text, author, date,
photos, video variants, the quoted post, and an X Article's full body (title + content blocks). If
FxTwitter is unavailable, X's own embed endpoint (``cdn.syndication.twimg.com``, used by embedded tweets)
gives the text and media, or just an article's title and preview.

Ids are ``x_<status id>``. Video posts are transcribed; everything else is an image/text post.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from media_pipeline.config import AppConfig
from media_pipeline.media import MediaError
from media_pipeline.models import VideoMetadata

logger = logging.getLogger(__name__)

PLATFORM = "X"
ID_PREFIX = "x_"
_HOSTS = {"x.com", "twitter.com", "mobile.twitter.com", "mobile.x.com"}
_STATUS = re.compile(r"^/(?:([A-Za-z0-9_]{1,30})/status(?:es)?|i(?:/web)?/status)/(\d+)")
_CACHE: dict[str, "XPost"] = {}


class XError(MediaError):
    """Reported at its pipeline stage like any other media failure."""


@dataclass
class XPost:
    status_id: str
    url: str
    author: str
    title: str
    text: str
    published: str
    media_kind: str  # "video" (transcribe it) or "image" (text and images)
    image_urls: list[str] = field(default_factory=list)
    video_url: str = ""
    duration: float | None = None

    @property
    def video_id(self) -> str:
        return ID_PREFIX + self.status_id


def is_x_host(host: str) -> bool:
    host = (host or "").lower().removeprefix("www.")
    return host in _HOSTS


def parse_x_ref(url: str) -> tuple[str, str]:
    parsed = urlparse((url or "").strip())
    if not is_x_host(parsed.hostname or ""):
        raise ValueError(url)
    match = _STATUS.match(parsed.path or "")
    if not match:
        raise ValueError(url)
    return PLATFORM, ID_PREFIX + match.group(2)


def canonicalize_x_url(url: str) -> str:
    parsed = urlparse((url or "").strip())
    match = _STATUS.match(parsed.path or "")
    if not match:
        return url
    user = match.group(1) or "i"
    return f"https://x.com/{user}/status/{match.group(2)}"


def fetch_x_metadata(url: str, asr_model: str, config: AppConfig) -> VideoMetadata:
    post = extract_x_post(url)
    return VideoMetadata(
        url=post.url,
        title=post.title,
        platform=PLATFORM,
        author=post.author,
        video_id=post.video_id,
        duration=post.duration,
        published=post.published,
        description=post.text,
        thumbnail_url=post.image_urls[0] if post.image_urls else "",
        asr_model=asr_model,
        media_kind=post.media_kind,
    )


def download_x_post(url: str, video_id: str, dest_dir: Path, config: AppConfig) -> Path:
    post = extract_x_post(url)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if post.media_kind == "video":
        target = dest_dir / f"{post.video_id}.mp4"
        if not (target.exists() and target.stat().st_size > 0):
            _download(post.video_url, target)
        return target
    folder = dest_dir / post.video_id
    folder.mkdir(parents=True, exist_ok=True)
    for index, image_url in enumerate(post.image_urls, start=1):
        target = folder / f"{index:02d}.jpg"
        if not (target.exists() and target.stat().st_size > 0):
            _download(image_url, target)
    return folder


def extract_x_post(url: str) -> XPost:
    canonical = canonicalize_x_url(url)
    cached = _CACHE.get(canonical)
    if cached is not None:
        return cached
    _, video_id = parse_x_ref(canonical)
    status_id = video_id.removeprefix(ID_PREFIX)
    try:
        post = parse_fxtwitter(_get_json(f"https://api.fxtwitter.com/i/status/{status_id}"), canonical)
    except Exception as exc:  # noqa: BLE001 - fall back to X's own embed endpoint
        logger.warning("FxTwitter failed for %s (%s); using X's embed endpoint", status_id, exc)
        post = parse_syndication(_get_json(
            f"https://cdn.syndication.twimg.com/tweet-result?id={status_id}&token={_embed_token(status_id)}&lang=en"),
            canonical)
    _CACHE[canonical] = post
    return post


def parse_fxtwitter(data: dict[str, Any], url: str) -> XPost:
    tweet = data.get("tweet") if isinstance(data, dict) else None
    if not tweet:
        raise XError("fetching_metadata", str((data or {}).get("message") or "X post not found"))
    author = tweet.get("author") or {}
    name = f"{author.get('name') or ''} (@{author.get('screen_name') or ''})".strip()
    images = [p.get("url") for p in (tweet.get("media") or {}).get("photos") or [] if p.get("url")]
    videos, silent_thumbs = [], []
    for v in (tweet.get("media") or {}).get("videos") or []:
        if not v.get("url"):
            continue
        if has_audio(v):
            videos.append((v.get("url"), v.get("duration")))
        elif v.get("thumbnail_url"):
            silent_thumbs.append(v["thumbnail_url"])  # a silent clip: nothing to transcribe, keep its still
    images = images + silent_thumbs
    article = tweet.get("article")
    if isinstance(article, dict) and article.get("content"):
        body = article_markdown(article)
        title = str(article.get("title") or "").strip() or "X Article"
        cover = ((article.get("cover_media") or {}).get("media_info") or {}).get("original_img_url")
        images = ([cover] if cover else []) + images
        text = body
    else:
        text = str(((tweet.get("raw_text") or {}).get("text")) or tweet.get("text") or "").strip()
        title = _first_line(text) or f"{name} on X"
    quote = tweet.get("quote")
    if isinstance(quote, dict) and (quote.get("text") or "").strip():
        quoted_author = (quote.get("author") or {}).get("screen_name") or ""
        text = f"{text}\n\n引用 @{quoted_author}：\n{quote['text'].strip()}"
    video_url, duration = (videos[0] if videos and not images else ("", None))
    return XPost(
        status_id=str(tweet.get("id") or parse_x_ref(url)[1].removeprefix(ID_PREFIX)),
        url=url,
        author=name,
        title=title,
        text=text.strip(),
        published=_date(tweet.get("created_timestamp")),
        media_kind="video" if video_url else "image",
        image_urls=[] if video_url else images,
        video_url=video_url or "",
        duration=float(duration) if duration else None,
    )


def has_audio(video: dict[str, Any]) -> bool:
    """X's mp4 files carry no audio when the clip is silent; its HLS playlist then lists no audio group."""
    if video.get("type") == "gif":
        return False
    playlist = next((f.get("url") for f in video.get("formats") or video.get("variants") or []
                     if f.get("container") == "m3u8" or ".m3u8" in str(f.get("url") or "")), "")
    if not playlist:
        return True  # can't tell; let ASR try
    try:
        return "TYPE=AUDIO" in _get_text(playlist) or "mp4a" in _get_text(playlist)
    except Exception:  # noqa: BLE001 - a failed check should not block the post
        return True


def _get_text(url: str) -> str:
    from curl_cffi import requests as cf_requests

    return cf_requests.get(url, impersonate="chrome", timeout=20).text


def article_markdown(article: dict[str, Any]) -> str:
    """X Article content blocks (Draft.js) → Markdown: headings, lists, quotes, paragraphs."""
    lines: list[str] = [f"# {article.get('title')}".strip()] if article.get("title") else []
    for block in (article.get("content") or {}).get("blocks") or []:
        text = str(block.get("text") or "").rstrip()
        kind = block.get("type") or "unstyled"
        if kind == "atomic" or not text.strip():
            continue
        prefix = {"header-one": "## ", "header-two": "### ", "header-three": "#### ", "unordered-list-item": "- ",
                  "ordered-list-item": "1. ", "blockquote": "> ", "code-block": "    "}.get(kind, "")
        lines.append(prefix + text)
    return "\n\n".join(lines)


def parse_syndication(data: dict[str, Any], url: str) -> XPost:
    if not isinstance(data, dict) or not data.get("id_str"):
        raise XError("fetching_metadata", "X post not found (it may be private or removed)")
    user = data.get("user") or {}
    name = f"{user.get('name') or ''} (@{user.get('screen_name') or ''})".strip()
    article = data.get("article") or {}
    if article.get("title"):
        title = article["title"].strip()
        text = f"# {title}\n\n{article.get('preview_text') or ''}\n\n（X 长文的正文需要登录才能看到，这里只有标题和预览。）"
    else:
        text = str(data.get("text") or "").strip()
        title = _first_line(text) or f"{name} on X"
    images, video_url, duration = [], "", None
    for media in data.get("mediaDetails") or []:
        if media.get("type") == "photo" and media.get("media_url_https"):
            images.append(media["media_url_https"])
        elif media.get("type") in ("video", "animated_gif"):
            variants = [v for v in (media.get("video_info") or {}).get("variants") or [] if v.get("content_type") == "video/mp4"]
            if variants and not video_url:
                video_url = max(variants, key=lambda v: v.get("bitrate") or 0)["url"]
                millis = (media.get("video_info") or {}).get("duration_millis")
                duration = float(millis) / 1000 if millis else None
    return XPost(
        status_id=str(data["id_str"]), url=url, author=name, title=title, text=text,
        published=str(data.get("created_at") or "")[:10], media_kind="video" if video_url and not images else "image",
        image_urls=[] if video_url and not images else images, video_url=video_url if not images else "",
        duration=duration,
    )


def _embed_token(status_id: str) -> str:
    """The token X's embed endpoint expects: (id / 1e15 * π) in base 36, without zeros and the dot."""
    value = (int(status_id) / 1e15) * math.pi
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    whole, frac = int(value), value - int(value)
    out = ""
    while whole:
        out = digits[whole % 36] + out
        whole //= 36
    out = (out or "0") + "."
    for _ in range(12):
        frac *= 36
        out += digits[int(frac)]
        frac -= int(frac)
    return out.replace("0", "").replace(".", "")


def _first_line(text: str) -> str:
    line = next((part.strip() for part in (text or "").splitlines() if part.strip() and not part.strip().startswith("http")), "")
    return line[:80] + ("…" if len(line) > 80 else "")


def _date(stamp: Any) -> str:
    try:
        return datetime.fromtimestamp(int(stamp), tz=timezone(timedelta(hours=8))).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return ""


def _get_json(url: str) -> dict[str, Any]:
    from curl_cffi import requests as cf_requests

    response = cf_requests.get(url, impersonate="chrome", timeout=30)
    if response.status_code >= 400:
        raise XError("fetching_metadata", f"X request failed: HTTP {response.status_code}")
    try:
        return json.loads(response.text)
    except ValueError as exc:
        raise XError("fetching_metadata", "X returned something that is not JSON") from exc


def _download(url: str, target: Path) -> None:
    from curl_cffi import requests as cf_requests

    partial = target.with_name(target.name + ".part")
    try:
        response = cf_requests.get(url, impersonate="chrome", timeout=300, stream=True)
        if response.status_code >= 400:
            raise XError("downloading", f"X media download failed: HTTP {response.status_code}")
        with partial.open("wb") as handle:
            for chunk in response.iter_content():
                handle.write(chunk)
    except XError:
        raise
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise XError("downloading", f"X media download failed: {exc}") from exc
    partial.replace(target)
