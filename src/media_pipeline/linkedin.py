"""LinkedIn posts: text, images, videos, and the post they reshare.

Public post pages (``linkedin.com/posts/...``) carry the whole post as JSON-LD (``SocialMediaPosting``, or
``VideoObject`` for video posts): the text (``articleBody``), author, date, images, and ``sharedContent``
when the post reshares another one. Video sources are in the player's ``data-sources`` attribute. Share
links from the app (``lnkd.in/p/...``) redirect to the post page; links inside the text (``lnkd.in/...``)
are resolved to where they point, so the note shows the real destination.

Ids are ``li_<activity id>``. Before a link has been opened the id is provisional (``li_share_<id>``,
``li_lnkd_<code>``), like Threads share links. Articles (``/pulse/``) are not handled.
"""

from __future__ import annotations

import html as html_lib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from media_pipeline.config import AppConfig
from media_pipeline.media import MediaError
from media_pipeline.models import VideoMetadata

logger = logging.getLogger(__name__)

PLATFORM = "LinkedIn"
ID_PREFIX = "li_"
_POST_PATH = re.compile(r"^/posts/[^/]+-(activity|share|ugcPost)-(\d+)(?:-[A-Za-z0-9_-]+)?/?$")
_FEED_PATH = re.compile(r"^/feed/update/urn:li:(activity|share|ugcPost):(\d+)/?$")
_SHORT_PATH = re.compile(r"^/(?:p/)?([A-Za-z0-9_-]+)/?$")
_LD = re.compile(r'<script type="application/ld\+json">(.*?)</script>', re.S)
_META = re.compile(r'<meta (?:property|name)="(og:[a-z:]+)" content="([^"]*)"')
_SOURCES = re.compile(r'data-sources="([^"]+)"')
_FEED_IMAGE = re.compile(r'(https://media\.licdn\.com/dms/image/[^"\s<>]*feedshare[^"\s<>]*)')
_SHORT_LINK = re.compile(r"https?://lnkd\.in/[A-Za-z0-9_-]+")
_CACHE: dict[str, "LinkedInPost"] = {}


class LinkedInError(MediaError):
    """Reported at its pipeline stage like any other media failure."""


@dataclass
class LinkedInPost:
    activity_id: str
    url: str
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
        return ID_PREFIX + self.activity_id


def is_linkedin_host(host: str) -> bool:
    host = (host or "").lower().removeprefix("www.")
    return host in {"linkedin.com", "lnkd.in"} or host.endswith(".linkedin.com")


def parse_linkedin_ref(url: str) -> tuple[str, str]:
    """(platform, video_id); share and short links get a provisional id until the page is opened."""
    parsed = urlparse((url or "").strip())
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if not is_linkedin_host(host):
        raise ValueError(url)
    path = parsed.path or ""
    if host == "lnkd.in":
        short = _SHORT_PATH.match(path)
        if short:
            return PLATFORM, f"{ID_PREFIX}lnkd_{short.group(1)}"
        raise ValueError(url)
    for pattern in (_POST_PATH, _FEED_PATH):
        match = pattern.match(path)
        if match:
            kind, number = match.groups()
            return PLATFORM, f"{ID_PREFIX}{number}" if kind == "activity" else f"{ID_PREFIX}{kind.lower()}_{number}"
    raise ValueError(url)


def canonicalize_linkedin_url(url: str) -> str:
    """Post URLs lose their tracking query (utm_*, rcm); short links are kept for the redirect."""
    parsed = urlparse((url or "").strip())
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if host.endswith("linkedin.com") and (_POST_PATH.match(parsed.path or "") or _FEED_PATH.match(parsed.path or "")):
        return f"https://www.linkedin.com{parsed.path}"
    return url


def fetch_linkedin_metadata(url: str, asr_model: str, config: AppConfig) -> VideoMetadata:
    post = extract_linkedin_post(url, config)
    first_line = next((line.strip() for line in post.text.splitlines() if line.strip()), "")
    title = first_line[:80] + ("…" if len(first_line) > 80 else "") if first_line else f"{post.author} on LinkedIn"
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


def download_linkedin_post(url: str, video_id: str, dest_dir: Path, config: AppConfig) -> Path:
    """A video post → <id>.mp4 for ASR; otherwise a folder of images (empty for text-only posts)."""
    post = extract_linkedin_post(url, config)
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


def extract_linkedin_post(url: str, config: AppConfig) -> LinkedInPost:
    cached = _CACHE.get(url)
    if cached is not None:
        return cached
    page, final_url = _fetch(url)
    post = parse_linkedin_page(page, final_url or url)
    post.text = _resolve_short_links(post.text)
    _CACHE[url] = post
    _CACHE[post.url] = post
    return post


def parse_linkedin_page(page: str, page_url: str) -> LinkedInPost:
    meta = {key: html_lib.unescape(value) for key, value in _META.findall(page or "")}
    blocks = [_loads(body) for body in _LD.findall(page or "")]
    post = next((b for b in blocks if isinstance(b, dict) and b.get("@type") in ("SocialMediaPosting", "VideoObject")),
                None)
    if post is None:
        if "/posts/" not in page_url and "/feed/update/" not in page_url:
            raise LinkedInError("fetching_metadata", f"LinkedIn did not open a post (ended at {page_url})")
        raise LinkedInError("fetching_metadata",
                            "LinkedIn did not show this post publicly (it may need a login, or was removed).")
    canonical = meta.get("og:url") or page_url
    activity = re.search(r"activity-(\d+)", canonical) or re.search(r"urn:li:activity:(\d+)", page) \
        or re.search(r"(?:activity|share|ugcPost)[-:](\d+)", page_url)
    if activity is None:
        raise LinkedInError("fetching_metadata", "Could not find the LinkedIn post id")
    author = _name(post.get("author")) or _name(post.get("creator")) or "LinkedIn"
    body = str(post.get("articleBody") or post.get("text") or post.get("description") or meta.get("og:description") or "")
    shared = post.get("sharedContent") if isinstance(post.get("sharedContent"), dict) else None
    if shared:
        shared_text = str(shared.get("articleBody") or shared.get("text") or shared.get("headline") or "").strip()
        if shared_text:
            body = f"{body.strip()}\n\n转发自 {_name(shared.get('author')) or '另一条帖子'}：\n{shared_text}"
    images = _image_urls(post.get("image")) or _feed_images(page)
    video_url, duration = _video(post, page)
    clean_url = canonical.split("?")[0]
    return LinkedInPost(
        activity_id=activity.group(1),
        url=clean_url if "/posts/" in clean_url or "/feed/update/" in clean_url else page_url.split("?")[0],
        author=author,
        text=body.strip(),
        published=str(post.get("datePublished") or post.get("uploadDate") or "")[:10],
        media_kind="video" if video_url else "image",
        image_urls=[] if video_url else images,
        video_url=video_url,
        thumbnail_url=(images[0] if images else str(post.get("thumbnailUrl") or "")),
        duration=duration,
    )


def _name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or "").strip()
    if isinstance(value, list) and value:
        return _name(value[0])
    return str(value or "").strip() if isinstance(value, str) else ""


def _image_urls(value: Any) -> list[str]:
    items = value if isinstance(value, list) else [value] if value else []
    urls = []
    for item in items:
        url = item.get("url") or item.get("contentUrl") if isinstance(item, dict) else item if isinstance(item, str) else None
        if url and "static.licdn.com" not in url:  # skip the generic LinkedIn card image
            urls.append(str(url))
    return list(dict.fromkeys(urls))


def _feed_images(page: str) -> list[str]:
    return list(dict.fromkeys(html_lib.unescape(url) for url in _FEED_IMAGE.findall(page or "")))


def _video(post: dict[str, Any], page: str) -> tuple[str, float | None]:
    duration = _iso_duration(str(post.get("duration") or ""))
    best, best_rate = "", -1
    for raw in _SOURCES.findall(page or ""):
        try:
            sources = json.loads(html_lib.unescape(raw))
        except ValueError:
            continue
        for source in sources if isinstance(sources, list) else []:
            src = str(source.get("src") or "")
            rate = int(source.get("data-bitrate") or source.get("bitrate") or 0)
            if src and "mp4" in (str(source.get("type") or "") + src) and rate > best_rate:
                best, best_rate = src, rate
    if not best and post.get("@type") == "VideoObject":
        best = str(post.get("contentUrl") or "")
    return best, duration


def _iso_duration(value: str) -> float | None:
    match = re.fullmatch(r"P(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)", value or "")
    if not match or not any(match.groups()):
        return None
    hours, minutes, seconds = (float(part or 0) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def _resolve_short_links(text: str, limit: int = 5) -> str:
    """'https://lnkd.in/x' → 'https://lnkd.in/x（→ https://real.example/page）' for the first few links."""
    seen: dict[str, str] = {}
    for short in list(dict.fromkeys(_SHORT_LINK.findall(text or "")))[:limit]:
        try:
            target = _redirect_target(short)
        except Exception:  # noqa: BLE001 - a dead link must not fail the post
            continue
        if target and target != short and "lnkd.in" not in urlparse(target).netloc:
            seen[short] = target
    for short, target in seen.items():
        text = text.replace(short, f"{short}（→ {target}）", 1)
    return text


def _redirect_target(url: str) -> str:
    from curl_cffi import requests as cf_requests

    response = cf_requests.get(url, impersonate="chrome", timeout=20, allow_redirects=False)
    location = response.headers.get("location") or ""
    if location:
        return location
    # lnkd.in often answers with an interstitial page whose "external_url_click" link is the destination.
    for tag in re.findall(r"<a\b[^>]*>", response.text or ""):
        if "external_url_click" in tag:
            href = re.search(r'href="(https?://[^"]+)"', tag)
            if href:
                return html_lib.unescape(href.group(1))
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
            headers={"Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8"},
        )
    except Exception as exc:
        raise LinkedInError("fetching_metadata", f"LinkedIn request failed: {exc}") from exc
    if response.status_code >= 400:
        raise LinkedInError("fetching_metadata", f"LinkedIn request failed: HTTP {response.status_code}")
    return response.text, str(getattr(response, "url", "") or url)


def _download(url: str, target: Path, *, stage: str) -> None:
    if not url:
        raise LinkedInError(stage, "LinkedIn media URL was not found")
    from curl_cffi import requests as cf_requests

    partial = target.with_name(target.name + ".part")
    try:
        response = cf_requests.get(url, impersonate="chrome", timeout=120, stream=True)
        if response.status_code >= 400:
            raise LinkedInError(stage, f"LinkedIn media download failed: HTTP {response.status_code}")
        with partial.open("wb") as handle:
            for chunk in response.iter_content():
                handle.write(chunk)
    except LinkedInError:
        raise
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise LinkedInError(stage, f"LinkedIn media download failed: {exc}") from exc
    partial.replace(target)
