"""Xiaoyuzhou (小宇宙) podcast episodes: the audio is transcribed like a video, the show notes become the
description.

Episode pages (``xiaoyuzhoufm.com/episode/<eid>``) are public and carry everything in ``__NEXT_DATA__``:
title, podcast, author, date, duration, the show notes (HTML with the timeline, guests and links) and a
public audio URL (``media.xyzcdn.net/...m4a``). Xiaoyuzhou's own transcript needs the app's login, so ASR
does the transcript. Ids are ``xyz_<eid>``.
"""

from __future__ import annotations

import html as html_lib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from media_pipeline.config import AppConfig
from media_pipeline.media import MediaError
from media_pipeline.models import VideoMetadata

logger = logging.getLogger(__name__)

PLATFORM = "Xiaoyuzhou"
ID_PREFIX = "xyz_"
_EPISODE_PATH = re.compile(r"^/episode/([0-9a-f]{24})/?$")
_NEXT_DATA = re.compile(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S)
_CACHE: dict[str, "Episode"] = {}


class XiaoyuzhouError(MediaError):
    """Reported at its pipeline stage like any other media failure."""


@dataclass
class Episode:
    eid: str
    url: str
    title: str
    podcast: str
    author: str
    published: str
    duration: float | None
    notes: str
    audio_url: str
    image_url: str

    @property
    def video_id(self) -> str:
        return ID_PREFIX + self.eid


def is_xiaoyuzhou_host(host: str) -> bool:
    host = (host or "").lower().removeprefix("www.")
    return host == "xiaoyuzhoufm.com" or host.endswith(".xiaoyuzhoufm.com")


def parse_xiaoyuzhou_ref(url: str) -> tuple[str, str]:
    parsed = urlparse((url or "").strip())
    if not is_xiaoyuzhou_host(parsed.hostname or ""):
        raise ValueError(url)
    match = _EPISODE_PATH.match(parsed.path or "")
    if not match:
        raise ValueError(url)
    return PLATFORM, ID_PREFIX + match.group(1)


def canonicalize_xiaoyuzhou_url(url: str) -> str:
    _, video_id = parse_xiaoyuzhou_ref(url)
    return f"https://www.xiaoyuzhoufm.com/episode/{video_id.removeprefix(ID_PREFIX)}"


def fetch_xiaoyuzhou_metadata(url: str, asr_model: str, config: AppConfig) -> VideoMetadata:
    episode = extract_episode(url)
    return VideoMetadata(
        url=episode.url,
        title=episode.title,
        platform=PLATFORM,
        author=" · ".join(part for part in (episode.podcast, episode.author) if part) or "小宇宙",
        video_id=episode.video_id,
        duration=episode.duration,
        published=episode.published,
        description=episode.notes,
        thumbnail_url=episode.image_url,
        asr_model=asr_model,
    )


def download_xiaoyuzhou_episode(url: str, video_id: str, dest_dir: Path, config: AppConfig) -> Path:
    episode = extract_episode(url)
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(urlparse(episode.audio_url).path).suffix.lower() or ".m4a"
    target = dest_dir / f"{episode.video_id}{suffix if suffix in ('.m4a', '.mp3', '.aac') else '.m4a'}"
    if not (target.exists() and target.stat().st_size > 0):
        _download(episode.audio_url, target)
    return target


def extract_episode(url: str) -> Episode:
    canonical = canonicalize_xiaoyuzhou_url(url)
    cached = _CACHE.get(canonical)
    if cached is not None:
        return cached
    episode = parse_episode_page(_fetch(canonical), canonical)
    _CACHE[canonical] = episode
    return episode


def parse_episode_page(page: str, page_url: str) -> Episode:
    match = _NEXT_DATA.search(page or "")
    try:
        data = json.loads(match.group(1)) if match else {}
    except ValueError:
        data = {}
    episode = ((data.get("props") or {}).get("pageProps") or {}).get("episode") or {}
    if not episode.get("eid"):
        raise XiaoyuzhouError("fetching_metadata", "小宇宙页面里没有找到这一集（可能已下架或是付费内容）")
    media = episode.get("media") or {}
    audio = ((media.get("source") or {}).get("url") or (episode.get("enclosure") or {}).get("url") or "")
    if not audio:
        raise XiaoyuzhouError("fetching_metadata", "这一集没有公开的音频（可能是付费内容）")
    podcast = episode.get("podcast") or {}
    image = (episode.get("image") or {}).get("picUrl") or (podcast.get("image") or {}).get("picUrl") or ""
    duration = episode.get("duration")
    return Episode(
        eid=str(episode["eid"]),
        url=page_url,
        title=str(episode.get("title") or episode["eid"]).strip(),
        podcast=str(podcast.get("title") or "").strip(),
        author=str(podcast.get("author") or "").strip(),
        published=_date(episode.get("pubDate")),
        duration=float(duration) if duration else None,
        notes=shownotes_text(str(episode.get("shownotes") or "")) or str(episode.get("description") or "").strip(),
        audio_url=str(audio),
        image_url=str(image),
    )


def shownotes_text(markup: str) -> str:
    """Show notes HTML → text with paragraphs and links kept (the timeline and guest list live here)."""
    text = re.sub(r'<a\b[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                  lambda m: m.group(2) if m.group(1) in m.group(2) else f"{m.group(2)} ({m.group(1)})", markup, flags=re.S)
    text = re.sub(r"<img\b[^>]*>", "", text)
    text = re.sub(r"<br\s*/?>|</p>|</li>|</h\d>", "\n", text)
    text = re.sub(r"<li\b[^>]*>", "- ", text)
    text = re.sub(r"<[^>]+>", "", text)
    lines = [" ".join(html_lib.unescape(line).split()) for line in text.splitlines()]
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):
            out.append(line)
    return "\n".join(out).strip()


def _date(value: Any) -> str:
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return ""
    return moment.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")


def _fetch(url: str) -> str:
    try:
        from curl_cffi import requests as cf_requests

        response = cf_requests.get(url, impersonate="chrome", timeout=60)
    except Exception as exc:
        raise XiaoyuzhouError("fetching_metadata", f"小宇宙请求失败：{exc}") from exc
    if response.status_code >= 400:
        raise XiaoyuzhouError("fetching_metadata", f"小宇宙请求失败：HTTP {response.status_code}")
    return response.text


def _download(url: str, target: Path) -> None:
    from curl_cffi import requests as cf_requests

    partial = target.with_name(target.name + ".part")
    try:
        response = cf_requests.get(url, impersonate="chrome", timeout=600, stream=True)
        if response.status_code >= 400:
            raise XiaoyuzhouError("downloading", f"小宇宙音频下载失败：HTTP {response.status_code}")
        with partial.open("wb") as handle:
            for chunk in response.iter_content():
                handle.write(chunk)
    except XiaoyuzhouError:
        raise
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise XiaoyuzhouError("downloading", f"小宇宙音频下载失败：{exc}") from exc
    partial.replace(target)
