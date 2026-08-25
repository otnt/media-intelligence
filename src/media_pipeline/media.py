from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from media_pipeline.config import AppConfig
from media_pipeline.models import VideoMetadata

logger = logging.getLogger(__name__)

_BV_RE = re.compile(r"BV[0-9A-Za-z]+")
_AV_RE = re.compile(r"(?:av|AV)(\d+)")
_YT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


class UnsupportedURLError(ValueError):
    pass


class MediaError(RuntimeError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


def parse_video_ref(url: str) -> tuple[str, str]:
    """Return (platform, video_id) from a supported URL."""
    raw = (url or "").strip()
    if not raw:
        raise UnsupportedURLError("URL is empty")
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if host in {"youtu.be"} or host.endswith("youtube.com"):
        video_id = _youtube_id(parsed)
        if not video_id:
            raise UnsupportedURLError(f"Could not parse a YouTube video id from {url}")
        return "YouTube", video_id
    if host.endswith("bilibili.com") or host in {"b23.tv"}:
        video_id = _bilibili_id(parsed)
        if not video_id:
            raise UnsupportedURLError(f"Could not parse a Bilibili video id from {url}")
        return "Bilibili", video_id
    if host.endswith("xiaohongshu.com") or host in {"xhslink.com"}:
        from media_pipeline.xhs import parse_xhs_ref

        try:
            return parse_xhs_ref(raw)
        except ValueError as exc:
            raise UnsupportedURLError(f"Could not parse a Xiaohongshu note id from {url}") from exc
    raise UnsupportedURLError("Only Bilibili, YouTube, and Xiaohongshu URLs are supported")


def canonicalize_url(url: str) -> str:
    """Collapse playlist/watchlater wrappers to a single-video URL."""
    try:
        platform, video_id = parse_video_ref(url)
    except UnsupportedURLError:
        return url
    if platform == "Bilibili" and _BV_RE.fullmatch(video_id):
        return f"https://www.bilibili.com/video/{video_id}"
    if platform == "Bilibili" and video_id.lower().startswith("av") and video_id[2:].isdigit():
        return f"https://www.bilibili.com/video/{video_id}"
    if platform == "YouTube" and video_id:
        return f"https://www.youtube.com/watch?v={video_id}"
    if platform == "Xiaohongshu":
        from media_pipeline.xhs import canonicalize_xhs_url

        return canonicalize_xhs_url(url, video_id)
    return url


def _youtube_id(parsed) -> str:
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    if "youtu.be" in host:
        candidate = path.strip("/").split("/")[0]
        return candidate if _YT_ID_RE.match(candidate) else ""
    query = parse_qs(parsed.query or "")
    if query.get("v"):
        candidate = query["v"][0]
        return candidate if _YT_ID_RE.match(candidate) else candidate
    parts = [item for item in path.split("/") if item]
    if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"}:
        return parts[1]
    return ""


def _bilibili_id(parsed) -> str:
    query = parse_qs(parsed.query or "")
    for key in ("bvid", "bvid[]"):
        if query.get(key):
            match = _BV_RE.search(query[key][0])
            if match:
                return match.group(0)
    path = parsed.path or ""
    match = _BV_RE.search(path)
    if match:
        return match.group(0)
    query_match = _BV_RE.search(parsed.query or "")
    if query_match:
        return query_match.group(0)
    av_match = _AV_RE.search(path) or _AV_RE.search(parsed.query or "")
    if av_match:
        return f"av{av_match.group(1)}"
    # Short links such as b23.tv/xxxx are resolved later by yt-dlp.
    tail = path.strip("/").split("/")[-1] if path.strip("/") else ""
    return tail


def fetch_metadata(url: str, asr_model: str, config: AppConfig) -> VideoMetadata:
    page_url = canonicalize_url(url)
    platform, fallback_id = _safe_parse(page_url)
    if platform == "Xiaohongshu":
        from media_pipeline.xhs import fetch_xhs_metadata

        return fetch_xhs_metadata(page_url, asr_model, config)
    info = _yt_dlp_info(page_url, config, download=False)
    video_id = str(info.get("id") or fallback_id)
    extractor = str(info.get("extractor_key") or info.get("extractor") or "").lower()
    if "bili" in extractor:
        platform = "Bilibili"
    elif "youtube" in extractor:
        platform = "YouTube"
    upload_date = str(info.get("upload_date") or "")
    published = ""
    if len(upload_date) == 8 and upload_date.isdigit():
        published = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"
    author = (
        info.get("uploader")
        or info.get("channel")
        or info.get("creator")
        or info.get("uploader_id")
        or ""
    )
    return VideoMetadata(
        url=str(info.get("webpage_url") or page_url),
        title=str(info.get("title") or video_id),
        platform=platform or "Unknown",
        author=str(author),
        video_id=video_id,
        duration=info.get("duration"),
        published=published,
        description=str(info.get("description") or ""),
        thumbnail_url=str(info.get("thumbnail") or ""),
        asr_model=asr_model,
    )


def download_video(url: str, video_id: str, dest_dir: Path, config: AppConfig) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = find_media_file(dest_dir, video_id)
    if existing:
        logger.info("Reusing downloaded video %s", existing)
        return existing
    page_url = canonicalize_url(url)
    platform, _fallback = _safe_parse(page_url)
    if platform == "Xiaohongshu":
        from media_pipeline.xhs import download_xhs_post

        return download_xhs_post(page_url, video_id, dest_dir, config)
    opts = _base_ydl_opts(config)
    opts.update(
        {
            "format": config.download.format,
            "merge_output_format": "mp4",
            "outtmpl": str(dest_dir / f"{video_id}.%(ext)s"),
            "noprogress": True,
        }
    )
    try:
        with _ydl(opts) as ydl:
            ydl.download([page_url])
    except Exception as exc:
        if config.download.cookies_from_browser:
            logger.warning("Download with browser cookies failed, retrying without cookies: %s", exc)
            fallback = dict(opts)
            fallback.pop("cookiesfrombrowser", None)
            try:
                with _ydl(fallback) as ydl:
                    ydl.download([page_url])
            except Exception as retry_exc:
                raise MediaError("downloading", f"yt-dlp download failed: {retry_exc}") from retry_exc
        else:
            raise MediaError("downloading", f"yt-dlp download failed: {exc}") from exc
    found = find_media_file(dest_dir, video_id)
    if not found:
        raise MediaError("downloading", "yt-dlp finished but the video file was not found")
    return found


def extract_audio(video_path: Path, audio_path: Path) -> Path:
    if audio_path.exists() and audio_path.stat().st_size > 0:
        logger.info("Reusing extracted audio %s", audio_path)
        return audio_path
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise MediaError("extracting_audio", "ffmpeg is not installed or not on PATH")
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(audio_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise MediaError("extracting_audio", f"ffmpeg failed to start: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()[-1:] or ["unknown error"]
        raise MediaError("extracting_audio", f"ffmpeg audio extraction failed: {detail[-1]}")
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise MediaError("extracting_audio", "ffmpeg produced an empty audio file")
    return audio_path


def probe_duration(path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe or not path.exists():
        return None
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    try:
        return float(completed.stdout.strip())
    except ValueError:
        return None


def find_media_file(directory: Path, video_id: str) -> Path | None:
    if not directory.exists():
        return None
    preferred = [".mp4", ".webm", ".mkv", ".m4a", ".mp3"]
    for suffix in preferred:
        candidate = directory / f"{video_id}{suffix}"
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    matches = sorted(directory.glob(f"{video_id}.*"))
    for match in matches:
        if match.is_file() and match.stat().st_size > 0:
            return match
    folder = directory / video_id
    if folder.is_dir():
        from media_pipeline.xhs import find_xhs_images

        if find_xhs_images(folder):
            return folder
    return None


def _safe_parse(url: str) -> tuple[str, str]:
    try:
        return parse_video_ref(url)
    except UnsupportedURLError:
        return "", ""


def _yt_dlp_info(url: str, config: AppConfig, download: bool) -> dict[str, Any]:
    info = _yt_dlp_extract(url, config, download=download)
    if info is None:
        raise MediaError("fetching_metadata", "yt-dlp returned no metadata")
    if info.get("is_live"):
        raise MediaError("fetching_metadata", "Live streams are not supported")
    return info


def _yt_dlp_extract(url: str, config: AppConfig, download: bool) -> dict[str, Any] | None:
    attempts: list[AppConfig] = [config]
    if config.download.cookies_from_browser:
        attempts.append(replace(config, download=replace(config.download, cookies_from_browser="")))
    errors: list[Exception] = []
    for attempt in attempts:
        opts = _base_ydl_opts(attempt)
        opts["skip_download"] = not download
        try:
            with _ydl(opts) as ydl:
                return ydl.extract_info(url, download=False)
        except Exception as exc:
            errors.append(exc)
            logger.warning("yt-dlp metadata attempt failed: %s", exc)
    raise MediaError("fetching_metadata", f"yt-dlp metadata failed: {errors[-1]}") from errors[-1]


def _base_ydl_opts(config: AppConfig) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "retries": 5,
        "fragment_retries": 5,
        "ignoreconfig": True,
        "overwrites": False,
    }
    browser = (config.download.cookies_from_browser or "").strip()
    if browser:
        opts["cookiesfrombrowser"] = (browser,)
    return opts


def _ydl(opts: dict[str, Any]):
    from yt_dlp import YoutubeDL

    return YoutubeDL(opts)
