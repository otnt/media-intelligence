from __future__ import annotations

import re

from media_pipeline.models import NamedSegment, Transcript, TranscriptSegment, WordSpan, format_timestamp

_WHISPER_TAGS = re.compile(r"\[(?:BLANK_AUDIO|MUSIC|SOUND|NOISE|INAUDIBLE)\]", re.IGNORECASE)
_MULTISPACE = re.compile(r"[ \t]{2,}")
_REPEATED_WORD = re.compile(r"\b(\S+)(?:\s+\1){7,}\b", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.!?。！？；;]$")
_CJK_CHAR = re.compile(r"[\u3400-\u9fff]")


def clean_text(text: str) -> str:
    cleaned = _WHISPER_TAGS.sub("", text or "")
    cleaned = collapse_repeated_ngrams(cleaned)
    cleaned = _REPEATED_WORD.sub(r"\1", cleaned)
    cleaned = _MULTISPACE.sub(" ", cleaned)
    return cleaned.strip()


def collapse_repeated_ngrams(text: str) -> str:
    """Collapse obvious ASR loops such as a phrase repeated many times in a row."""
    if not text:
        return ""
    tokens = text.split()
    if len(tokens) < 8:
        return text
    for ngram in (2, 3, 4, 5, 6, 8):
        min_repeats = 4 if ngram == 2 else 3
        tokens = _collapse_token_ngrams(tokens, ngram, min_repeats=min_repeats)
    return " ".join(tokens)


def _collapse_token_ngrams(tokens: list[str], ngram: int, min_repeats: int) -> list[str]:
    if len(tokens) < ngram * min_repeats:
        return tokens
    result: list[str] = []
    index = 0
    while index < len(tokens):
        chunk = tokens[index : index + ngram]
        if len(chunk) < ngram:
            result.extend(tokens[index:])
            break
        repeats = 1
        cursor = index + ngram
        while tokens[cursor : cursor + ngram] == chunk:
            repeats += 1
            cursor += ngram
        if repeats >= min_repeats:
            result.extend(chunk)
            index = cursor
        else:
            result.append(tokens[index])
            index += 1
    return result


def clean_transcript(transcript: Transcript) -> Transcript:
    segments: list[TranscriptSegment] = []
    previous_text = None
    repeat_count = 0
    for segment in transcript.segments:
        text = clean_text(segment.text)
        if not text:
            continue
        if text == previous_text:
            repeat_count += 1
            if repeat_count >= 3:
                continue
        else:
            previous_text = text
            repeat_count = 1
        words = None
        if segment.words:
            words = [
                WordSpan(start=word.start, end=word.end, text=clean_text(word.text) or word.text)
                for word in segment.words
            ]
        segments.append(TranscriptSegment(start=segment.start, end=segment.end, text=text, words=words))
    return Transcript(
        language=transcript.language,
        segments=segments,
        provider=transcript.provider,
        model=transcript.model,
    )


def group_word_spans(words: list[WordSpan], max_duration: float = 12.0, pause: float = 0.8) -> list[TranscriptSegment]:
    if not words:
        return []
    segments: list[TranscriptSegment] = []
    current: list[WordSpan] = []
    for word in words:
        if not current:
            current.append(word)
            continue
        duration = word.end - current[0].start
        gap = word.start - current[-1].end
        current_text = _join_words(current)
        should_flush = False
        if gap >= pause and len(current) >= 3:
            should_flush = True
        elif duration >= max_duration and len(current) >= 4:
            should_flush = True
        elif _SENTENCE_END.search(current_text.strip()) and gap >= 0.25:
            should_flush = True
        if should_flush:
            segments.append(_segment_from_words(current))
            current = [word]
        else:
            current.append(word)
    if current:
        segments.append(_segment_from_words(current))
    return segments


def _segment_from_words(words: list[WordSpan]) -> TranscriptSegment:
    return TranscriptSegment(
        start=words[0].start,
        end=words[-1].end,
        text=_join_words(words).strip(),
        words=words,
    )


def _join_words(words: list[WordSpan]) -> str:
    parts: list[str] = []
    for word in words:
        token = word.text or ""
        if not token:
            continue
        if not parts:
            parts.append(token.lstrip())
            continue
        previous = parts[-1]
        if token.startswith(" ") or _needs_space(previous, token):
            parts.append(token if token.startswith(" ") else f" {token}")
        else:
            parts.append(token)
    return "".join(parts)


def _needs_space(previous: str, token: str) -> bool:
    left = previous[-1]
    right = token[0]
    if right in " \t,.;:!?)]}%。，、！？；：”’…":
        return False
    if left in "([{“‘":
        return False
    if _CJK_CHAR.match(left) and _CJK_CHAR.match(right):
        return False
    return True


def render_transcript(
    segments: list[NamedSegment],
    *,
    video_id: str = "",
    frames: list[tuple[float, str] | tuple[float, str, str]] | None = None,
) -> str:
    """Render the transcript, inserting keyframe wikilinks before covering blocks.

    A frame is placed immediately above the first speech block whose [start, end]
    contains its timestamp. If no block covers it, it sits above the next later
    block, or after the last block if speech has already ended.
    """
    usable = [segment for segment in segments if segment.text.strip()]
    placed = _place_frames(usable, _normalize_frames(frames or []))
    lines = ["## Transcript", ""]
    if not usable and not (frames or []):
        lines.append("No speech detected.")
        lines.append("")
        return "\n".join(lines)
    for index, segment in enumerate(usable):
        lines.extend(_frame_lines(video_id, placed[index]))
        lines.append(f"### [{format_timestamp(segment.start)}] {segment.speaker_label}")
        lines.append("")
        lines.append(segment.text.strip())
        lines.append("")
    lines.extend(_frame_lines(video_id, placed[-1] if placed else []))
    return "\n".join(lines).rstrip() + "\n"


def _normalize_frames(frames: list[tuple[float, str] | tuple[float, str, str]]) -> list[tuple[float, str, str]]:
    normalized: list[tuple[float, str, str]] = []
    for item in frames:
        stamp = float(item[0])
        path = str(item[1])
        caption = str(item[2]).strip() if len(item) > 2 else ""
        normalized.append((stamp, path, caption))
    return normalized


def _place_frames(
    segments: list[NamedSegment],
    frames: list[tuple[float, str, str]],
) -> list[list[tuple[float, str, str]]]:
    buckets: list[list[tuple[float, str, str]]] = [[] for _ in range(len(segments) + 1)]
    ordered = sorted(frames, key=lambda item: (item[0], item[1]))
    for stamp, filename, caption in ordered:
        covering = next(
            (index for index, segment in enumerate(segments) if segment.start <= stamp <= segment.end),
            None,
        )
        if covering is not None:
            buckets[covering].append((stamp, filename, caption))
            continue
        later = next(
            (index for index, segment in enumerate(segments) if segment.start > stamp),
            len(segments),
        )
        buckets[later].append((stamp, filename, caption))
    return buckets


def _frame_lines(video_id: str, frames: list[tuple[float, str, str]]) -> list[str]:
    if not video_id or not frames:
        return []
    lines: list[str] = []
    for _stamp, filename, caption in frames:
        name = filename.rsplit("/", 1)[-1]
        lines.append(f"![[attachments/{video_id}/{name}]]")
        if caption:
            lines.append("")
            lines.append(f"*{caption}*")
        lines.append("")
    return lines
