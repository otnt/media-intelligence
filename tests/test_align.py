from media_pipeline.align import align_transcript, majority_speaker
from media_pipeline.models import DiarizationSegment, Transcript, TranscriptSegment, WordSpan


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


def test_splits_on_word_level_speaker_change():
    transcript = Transcript(
        language="en",
        provider="test",
        model="test",
        segments=[
            TranscriptSegment(
                start=0.0,
                end=4.0,
                text="Hello thanks",
                words=[
                    WordSpan(0.0, 1.5, "Hello"),
                    WordSpan(2.0, 3.8, "thanks"),
                ],
            )
        ],
    )
    diarization = [
        DiarizationSegment(0.0, 1.8, "SPEAKER_00"),
        DiarizationSegment(1.8, 5.0, "SPEAKER_01"),
    ]
    aligned = align_transcript(transcript, diarization)
    assert [item.speaker_id for item in aligned] == ["SPEAKER_00", "SPEAKER_01"]
    assert aligned[0].text == "Hello"
    assert aligned[1].text == "thanks"


def test_majority_speaker_prefers_longest_overlap():
    speakers = [
        DiarizationSegment(0.0, 1.0, "SPEAKER_00"),
        DiarizationSegment(1.0, 10.0, "SPEAKER_01"),
    ]
    assert majority_speaker(0.5, 8.0, speakers) == "SPEAKER_01"
