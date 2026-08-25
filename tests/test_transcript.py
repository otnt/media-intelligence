from media_pipeline.transcript import clean_text, collapse_repeated_ngrams, render_transcript
from media_pipeline.models import NamedSegment


def test_collapses_whisper_phrase_loops():
    looped = "thank you thank you thank you thank you thank you thank you thank you thank you"
    assert clean_text(looped) == "thank you"


def test_preserves_natural_repetition():
    text = "yeah yeah yeah I know"
    assert "yeah yeah yeah" in clean_text(text)


def test_removes_blank_audio_tags():
    assert clean_text("[BLANK_AUDIO] Hello there") == "Hello there"


def test_render_transcript_keeps_speaker_changes():
    markdown = render_transcript(
        [
            NamedSegment(0, 6, "SPEAKER_00", "Host", "Welcome to today's discussion."),
            NamedSegment(8, 13, "SPEAKER_01", "Alice", "Thanks for having me."),
        ]
    )
    assert "### [00:00:00] Host" in markdown
    assert "### [00:00:08] Alice" in markdown
    assert markdown.index("Host") < markdown.index("Alice")


def test_collapse_does_not_summarize_unique_content():
    text = "The first thing we should talk about is alignment."
    assert collapse_repeated_ngrams(text) == text
