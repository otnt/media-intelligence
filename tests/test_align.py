from media_pipeline.align import align_transcript, majority_speaker
from media_pipeline.models import DiarizationSegment, Transcript, TranscriptSegment


def test_assigns_majority_speaker_without_splitting_text():
    transcript = Transcript(
        language="en",
        provider="test",
        model="test",
        segments=[TranscriptSegment(start=0.0, end=6.0, text="Welcome to today's discussion.")],
    )
    diarization = [
        DiarizationSegment(0.0, 8.5, "SPEAKER_00"),
        DiarizationSegment(8.5, 16.4, "SPEAKER_01"),
    ]
    aligned = align_transcript(transcript, diarization)
    assert len(aligned) == 1
    assert aligned[0].speaker_id == "SPEAKER_00"
    assert aligned[0].text == "Welcome to today's discussion."
    assert aligned[0].start == 0.0
    assert aligned[0].end == 6.0


def test_keeps_one_segment_when_speakers_overlap_coarse_bounds():
    transcript = Transcript(
        language="en",
        provider="test",
        model="test",
        segments=[TranscriptSegment(start=0.0, end=4.0, text="Hello thanks")],
    )
    diarization = [
        DiarizationSegment(0.0, 1.8, "SPEAKER_00"),
        DiarizationSegment(1.8, 5.0, "SPEAKER_01"),
    ]
    aligned = align_transcript(transcript, diarization)
    assert len(aligned) == 1
    assert aligned[0].speaker_id == "SPEAKER_01"
    assert aligned[0].text == "Hello thanks"


def test_majority_speaker_prefers_longest_overlap():
    speakers = [
        DiarizationSegment(0.0, 1.0, "SPEAKER_00"),
        DiarizationSegment(1.0, 10.0, "SPEAKER_01"),
    ]
    assert majority_speaker(0.5, 8.0, speakers) == "SPEAKER_01"
