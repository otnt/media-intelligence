"""YouTube's own captions and chapters, taken from the same yt-dlp extraction that downloads a video.

Most talks have automatic captions (``<lang>-orig``, YouTube's ASR, with punctuation) and some have
uploaded ones; either is a free transcript, just without speaker turns. Captions become a normal
:class:`Transcript` (model ``youtube-captions``, one speaker), saved where the pipeline looks for
transcripts, so a summary can be written without paying for ASR. Chapters are kept with the metadata.
"""

from __future__ import annotations

import json
import re
from typing import Any

from media_pipeline.models import Transcript, TranscriptSegment

CAPTIONS_MODEL = "youtube-captions"
SPEAKER = "SPEAKER_00"
_NOISE = re.compile(r"^\[[^\]]*\]$")  # [Music], [Applause]
_SENTENCE_END = (".", "?", "!", "。", "？", "！")


def pick_track(info: dict[str, Any]) -> tuple[str, str, str] | None:
    """(language, "manual"|"auto", json3 URL) for the video's own language, uploaded captions first."""
    language = str(info.get("language") or "en")
    base = language.split("-")[0]
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    candidates: list[tuple[str, str, list]] = []
    for key in sorted(manual, key=lambda k: (k != language, k != base)):
        if key in (language, base) or key.startswith(base + "-"):
            candidates.append((key, "manual", manual[key]))
    for key in (f"{language}-orig", f"{base}-orig", language, base):
        if key in auto and all(key != c[0] for c in candidates):
            candidates.append((key, "auto", auto[key]))
    for key, kind, formats in candidates:
        url = next((f.get("url") for f in formats or [] if f.get("ext") == "json3" and f.get("url")), None)
        if url:
            return key.removesuffix("-orig"), kind, url
    return None


def parse_json3(data: dict[str, Any], language: str = "") -> Transcript:
    """Caption events → sentence-sized segments (8–30 s), one speaker."""
    pieces: list[tuple[float, float, str]] = []
    for event in data.get("events") or []:
        segs = event.get("segs")
        if not segs:
            continue
        text = " ".join("".join(str(seg.get("utf8") or "") for seg in segs).split())
        if not text or _NOISE.match(text):
            continue
        start = float(event.get("tStartMs") or 0) / 1000.0
        end = start + float(event.get("dDurationMs") or 0) / 1000.0
        pieces.append((start, end, text))
    segments: list[TranscriptSegment] = []
    words: list[str] = []
    seg_start = seg_end = 0.0
    for index, (start, end, text) in enumerate(pieces):
        if not words:
            seg_start = start
        words.append(text)
        following = pieces[index + 1][0] if index + 1 < len(pieces) else end
        seg_end = max(seg_end, min(end, following) if following > start else end)
        joined = " ".join(words)
        long_enough = seg_end - seg_start >= 8.0
        if (joined.endswith(_SENTENCE_END) and long_enough) or seg_end - seg_start >= 30.0 or len(joined) >= 500:
            segments.append(TranscriptSegment(seg_start, max(seg_end, seg_start), joined, speaker=SPEAKER))
            words = []
            seg_end = 0.0
    if words:
        segments.append(TranscriptSegment(seg_start, max(seg_end, seg_start), " ".join(words), speaker=SPEAKER))
    return Transcript(language=language or None, segments=segments, provider="youtube", model=CAPTIONS_MODEL)


def fetch_captions(ydl: Any, info: dict[str, Any]) -> tuple[Transcript | None, str]:
    """(transcript, kind) using the extraction's own session; (None, "") when the video has none."""
    track = pick_track(info)
    if track is None:
        return None, ""
    language, kind, url = track
    with ydl.urlopen(url) as response:
        data = json.loads(response.read().decode("utf-8") or "{}")
    transcript = parse_json3(data, language)
    return (transcript if transcript.segments else None), kind


def chapters(info: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"start": float(item.get("start_time") or 0), "end": float(item.get("end_time") or 0),
         "title": str(item.get("title") or "").strip()}
        for item in info.get("chapters") or []
        if str(item.get("title") or "").strip()
    ]
