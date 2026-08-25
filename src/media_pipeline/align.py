from __future__ import annotations

from media_pipeline.models import AlignedSegment, DiarizationSegment, Transcript


def align_transcript(
    transcript: Transcript,
    diarization: list[DiarizationSegment] | None,
) -> list[AlignedSegment]:
    """Assign a speaker to each transcript segment by timestamp overlap.

    Transcript segments and diarization turns share the source-audio timeline.
    Word-level timestamps are not used.
    """
    speakers = diarization or []
    aligned: list[AlignedSegment] = []
    for segment in transcript.segments:
        if not segment.text.strip():
            continue
        speaker = majority_speaker(segment.start, segment.end, speakers) if speakers else "SPEAKER_00"
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


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _drop_empty(segments: list[AlignedSegment]) -> list[AlignedSegment]:
    return [segment for segment in segments if segment.text.strip()]
