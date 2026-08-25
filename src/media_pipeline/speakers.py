from __future__ import annotations

import re
from collections import defaultdict

from media_pipeline.models import AlignedSegment, NamedSegment, VideoMetadata

_SELF_INTRO_EN = re.compile(
    r"(?i:i(?:['’]m| am)|my name is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b"
)
_SELF_INTRO_ZH = re.compile(r"(?:我是|我叫|我的名字是)\s*([^\s，。！？,.]{2,20})")
_HOST_PHRASE_EN = re.compile(
    r"\bwelcome to (?:the |my )?(?:show|podcast|channel)\b",
    re.IGNORECASE,
)
_HOST_PHRASE_ZH = re.compile(r"(欢迎来到|欢迎收看|欢迎收听|我是主持人)")
_NAME_TOKEN = re.compile(r"^[A-Z][a-z]+(?:[ '-][A-Z][a-z]+)*$")
_TITLE_WITH = re.compile(
    r"(?:interview(?:s)?(?: with)?|with|feat(?:uring)?\.?|x |vs\.? |对谈|对话|专访|feat\.?)\s+(.+)$",
    re.IGNORECASE,
)


def name_speakers(segments: list[AlignedSegment], metadata: VideoMetadata | None) -> list[NamedSegment]:
    speaker_ids = _ordered_speaker_ids(segments)
    labels = {speaker_id: _fallback_label(index) for index, speaker_id in enumerate(speaker_ids)}
    if not segments:
        return []

    evidence: dict[str, list[str]] = defaultdict(list)
    for speaker_id in speaker_ids:
        text = " ".join(item.text for item in segments if item.speaker_id == speaker_id)
        for match in _SELF_INTRO_EN.finditer(text):
            evidence[speaker_id].append(match.group(1).strip())
        for match in _SELF_INTRO_ZH.finditer(text):
            evidence[speaker_id].append(match.group(1).strip())

    title_names = _names_from_title(metadata.title if metadata else "")
    author = (metadata.author if metadata else "") or ""
    description = metadata.description if metadata else ""
    known_names = _unique([*title_names, author] if author else title_names)
    known_names.extend(_names_from_description(description))
    known_names = _unique(known_names)

    assigned: dict[str, str] = {}
    used: set[str] = set()

    for speaker_id, names in evidence.items():
        resolved = _best_supported_name(names, known_names) or _confident_self_name(names)
        if resolved and resolved.lower() not in used:
            assigned[speaker_id] = resolved
            used.add(resolved.lower())

    host_id = _detect_host(segments, speaker_ids, author)
    if host_id and host_id not in assigned:
        assigned[host_id] = "Host"
        used.add("host")
    elif host_id and _names_similar(assigned.get(host_id, ""), author):
        assigned[host_id] = "Host"

    for speaker_id in speaker_ids:
        if speaker_id in assigned:
            labels[speaker_id] = assigned[speaker_id]

    return [
        NamedSegment(
            start=segment.start,
            end=segment.end,
            speaker_id=segment.speaker_id,
            speaker_label=labels.get(segment.speaker_id, "Speaker 1"),
            text=segment.text,
        )
        for segment in segments
    ]


def _ordered_speaker_ids(segments: list[AlignedSegment]) -> list[str]:
    ordered: list[str] = []
    for segment in segments:
        if segment.speaker_id not in ordered:
            ordered.append(segment.speaker_id)
    return ordered


def _fallback_label(index: int) -> str:
    return f"Speaker {index + 1}"


def _detect_host(segments: list[AlignedSegment], speaker_ids: list[str], author: str) -> str | None:
    if not speaker_ids:
        return None
    scores: dict[str, float] = {speaker_id: 0.0 for speaker_id in speaker_ids}
    for speaker_id in speaker_ids:
        text = " ".join(item.text for item in segments if item.speaker_id == speaker_id)
        if _HOST_PHRASE_EN.search(text) or _HOST_PHRASE_ZH.search(text):
            scores[speaker_id] += 2.0
        if author and author.lower() in text.lower():
            scores[speaker_id] += 1.5
    best_id = max(scores, key=lambda key: scores[key])
    if scores[best_id] >= 2.0:
        return best_id
    return None


def _names_from_title(title: str) -> list[str]:
    if not title:
        return []
    names: list[str] = []
    match = _TITLE_WITH.search(title)
    if match:
        tail = re.split(r"[|:/\-–—]", match.group(1))[0]
        candidate = tail.strip(" .")
        if 1 < len(candidate.split()) <= 5 or _CJK_NAME(candidate):
            names.append(candidate)
    for token in re.split(r"[|:/\-–—]", title):
        token = token.strip()
        if _NAME_TOKEN.match(token) and " " in token:
            names.append(token)
    return names


def _CJK_NAME(value: str) -> bool:
    return bool(re.fullmatch(r"[\u3400-\u9fff]{2,6}", value.strip()))


def _names_from_description(description: str) -> list[str]:
    if not description:
        return []
    names: list[str] = []
    for match in re.finditer(r"\bGuest:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})", description):
        names.append(match.group(1).strip())
    for match in re.finditer(r"(?:嘉宾|主讲)[:：]\s*([^\s，。,]+)", description):
        names.append(match.group(1).strip())
    return names


def _best_supported_name(claimed: list[str], known: list[str]) -> str | None:
    for claim in claimed:
        for candidate in known:
            if _names_similar(claim, candidate):
                return candidate if len(candidate) >= len(claim) else claim
    return None


def _confident_self_name(claimed: list[str]) -> str | None:
    """Accept a self-introduction only when it looks like a real name, not a role word."""
    blocked = {
        "host",
        "guest",
        "speaker",
        "you",
        "me",
        "the host",
        "主持人",
        "嘉宾",
        "the",
        "a",
        "an",
        "this",
        "that",
    }
    for claim in claimed:
        lowered = claim.strip().lower()
        if lowered in blocked:
            continue
        if _NAME_TOKEN.match(claim) or _CJK_NAME(claim):
            return claim.strip()
    return None


def _names_similar(left: str, right: str) -> bool:
    if not left or not right:
        return False
    a = _normalize_name(left)
    b = _normalize_name(right)
    return a == b or a in b or b in a


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", value.lower())


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.strip().lower()
        if not value.strip() or key in seen:
            continue
        seen.add(key)
        result.append(value.strip())
    return result
