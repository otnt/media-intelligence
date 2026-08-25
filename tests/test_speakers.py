from media_pipeline.models import AlignedSegment, VideoMetadata
from media_pipeline.speakers import name_speakers


def _meta(**kwargs) -> VideoMetadata:
    payload = {
        "url": "https://www.youtube.com/watch?v=abc",
        "title": "A conversation",
        "platform": "YouTube",
        "author": "Example Channel",
        "video_id": "abc",
        "duration": 60.0,
        "published": "2026-08-01",
        "description": "",
        "thumbnail_url": "",
        "asr_model": "whisper-large-v3-turbo",
    }
    payload.update(kwargs)
    return VideoMetadata(**payload)


def test_self_introduction_is_used_as_name():
    segments = [
        AlignedSegment(0, 5, "SPEAKER_00", "Welcome to the podcast."),
        AlignedSegment(5, 10, "SPEAKER_01", "I'm Andrej Karpathy and I work on this problem."),
    ]
    named = name_speakers(segments, _meta(title="Interview with Andrej Karpathy"))
    assert named[1].speaker_label == "Andrej Karpathy"


def test_does_not_assign_title_name_without_transcript_evidence():
    segments = [
        AlignedSegment(0, 5, "SPEAKER_00", "Let's start with the basics."),
        AlignedSegment(5, 10, "SPEAKER_01", "The first thing we should talk about is alignment."),
    ]
    named = name_speakers(segments, _meta(title="Interview with Andrej Karpathy"))
    assert named[0].speaker_label == "Speaker 1"
    assert named[1].speaker_label == "Speaker 2"


def test_host_label_requires_host_evidence():
    segments = [
        AlignedSegment(0, 8, "SPEAKER_00", "Welcome to the podcast. Today we have a special guest."),
        AlignedSegment(8, 14, "SPEAKER_01", "Thanks, I'm Alice Johnson."),
    ]
    named = name_speakers(segments, _meta(author="Lex Fridman", title="Lex Fridman Podcast"))
    assert named[0].speaker_label == "Host"
    assert named[1].speaker_label == "Alice Johnson"
