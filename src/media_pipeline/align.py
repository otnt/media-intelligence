from __future__ import annotations

from media_pipeline.models import AlignedSegment, DiarizationSegment, Transcript, TranscriptSegment, WordSpan


def align_transcript(
    transcript: Transcript,
    diarization: list[DiarizationSegment] | None,
) -> list[AlignedSegment]:
    speakers = diarization or []
    aligned: list[AlignedSegment] = []
    for segment in transcript.segments:
        if not segment.text.strip():
            continue
        if not speakers:
            aligned.append(
                AlignedSegment(
                    start=segment.start,
                    end=segment.end,
                    speaker_id="SPEAKER_00",
                    text=segment.text.strip(),
                )
            )
            continue
        if segment.words:
            aligned.extend(_align_with_words(segment, speakers))
        else:
            speaker = majority_speaker(segment.start, segment.end, speakers)
            aligned.append(
                AlignedSegment(
                    start=segment.start,
                    end=segment.end,
                    speaker_id=speaker,
                    text=segment.text.strip(),
                )
            )
    return _drop_empty(aligned)


def majority_speaker(start: float, end: float, speakers: list[DiarizationSegment]) -> str:
    best = "SPEAKER_00"
    best_overlap = -1.0
    for item in speakers:
        overlap = _overlap(start, end, item.start, item.end)
        if overlap > best_overlap:
            best_overlap = overlap
            best = item.speaker
    return best


def speaker_at(time: float, speakers: list[DiarizationSegment]) -> str:
    for item in speakers:
        if item.start <= time < item.end:
            return item.speaker
    nearest = min(speakers, key=lambda item: min(abs(time - item.start), abs(time - item.end)))
    return nearest.speaker


def _align_with_words(segment: TranscriptSegment, speakers: list[DiarizationSegment]) -> list[AlignedSegment]:
    groups: list[tuple[str, list[WordSpan]]] = []
    current_speaker: str | None = None
    current_words: list[WordSpan] = []
    for word in segment.words or []:
        midpoint = (word.start + word.end) / 2.0
        speaker = speaker_at(midpoint, speakers)
        if current_speaker is None:
            current_speaker = speaker
        if speaker != current_speaker and current_words:
            groups.append((current_speaker, current_words))
            current_words = [word]
            current_speaker = speaker
        else:
            current_words.append(word)
            current_speaker = speaker
    if current_speaker is not None and current_words:
        groups.append((current_speaker, current_words))

    if not groups:
        return [
            AlignedSegment(
                start=segment.start,
                end=segment.end,
                speaker_id=majority_speaker(segment.start, segment.end, speakers),
                text=segment.text.strip(),
            )
        ]

    from media_pipeline.transcript import _join_words

    result: list[AlignedSegment] = []
    for speaker, words in groups:
        text = _join_words(words).strip() or segment.text.strip()
        result.append(
            AlignedSegment(
                start=words[0].start,
                end=words[-1].end,
                speaker_id=speaker,
                text=text,
            )
        )
    return result


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _drop_empty(segments: list[AlignedSegment]) -> list[AlignedSegment]:
    return [segment for segment in segments if segment.text.strip()]
