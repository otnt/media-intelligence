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


def test_render_transcript_inserts_frames_before_covering_block():
    markdown = render_transcript(
        [
            NamedSegment(0, 10, "s0", "Host", "hello"),
            NamedSegment(20, 30, "s0", "Host", "now look at the slide"),
        ],
        video_id="BVtest",
        frames=[
            (5.0, "keyframes/00-00-05.000.jpg"),
            (12.0, "keyframes/00-00-12.000.jpg"),
            (25.0, "00-00-25.000.jpg"),
            (40.0, "00-00-40.000.jpg"),
        ],
    )
    assert "## Visual Timeline" not in markdown
    assert "Nearby transcript:" not in markdown
    before_first = markdown.split("### [00:00:00] Host")[0]
    assert "![[attachments/BVtest/00-00-05.000.jpg]]" in before_first
    between = markdown.split("### [00:00:00] Host")[1].split("### [00:00:20] Host")[0]
    assert "![[attachments/BVtest/00-00-12.000.jpg]]" in between
    assert "![[attachments/BVtest/00-00-25.000.jpg]]" in between
    assert between.index("00-00-12.000.jpg") < between.index("00-00-25.000.jpg")
    after_second = markdown.split("### [00:00:20] Host")[1]
    assert "![[attachments/BVtest/00-00-40.000.jpg]]" in after_second
    assert markdown.index("hello") < markdown.index("00-00-12.000.jpg")
    assert markdown.index("00-00-25.000.jpg") < markdown.index("now look at the slide")


def test_render_transcript_includes_caption_under_frame():
    markdown = render_transcript(
        [NamedSegment(0, 10, "s0", "Host", "hello")],
        video_id="BVtest",
        frames=[(5.0, "keyframes/00-00-05.000.jpg", "Cover of three recommended books")],
    )
    assert "![[attachments/BVtest/00-00-05.000.jpg]]" in markdown
    assert "*Cover of three recommended books*" in markdown
    assert markdown.index("00-00-05.000.jpg") < markdown.index("Cover of three recommended books")
    assert markdown.index("Cover of three recommended books") < markdown.index("hello")


def test_collapse_does_not_summarize_unique_content():
    text = "The first thing we should talk about is alignment."
    assert collapse_repeated_ngrams(text) == text
