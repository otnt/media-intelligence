from pathlib import Path

from media_pipeline.models import Task, TaskStatus, VideoMetadata, asr_label
from media_pipeline.notes import NoteDocument, NoteWriter, sanitize_filename


def test_sanitize_filename_keeps_cjk_and_strips_illegal_chars():
    assert sanitize_filename('How to / Quickly: "Break" Ice?.md') == "How to Quickly Break Ice.md"
    assert "关系" in sanitize_filename("如何快速破冰并建立关系")


def test_note_round_trip(tmp_path: Path):
    metadata = VideoMetadata(
        url="https://www.bilibili.com/video/BVxxxx",
        title="How to Quickly Break the Ice and Build Relationships",
        platform="Bilibili",
        author="Example Author",
        video_id="BVxxxx",
        duration=1471,
        published="2026-08-01",
        description="A talk about social skills.",
        thumbnail_url="https://example.com/thumb.jpg",
        asr_model="whisper-large-v3-turbo",
    )
    task = Task(
        id="t1",
        url=metadata.url,
        asr_model="whisper-large-v3-turbo",
        status=TaskStatus.downloading,
        video_id=metadata.video_id,
        video_path="/Users/me/AIContent/videos/BVxxxx.mp4",
    )
    writer = NoteWriter(tmp_path)
    path = writer.create_or_open(metadata, task)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# How to Quickly Break the Ice and Build Relationships")
    assert "Status: downloading" in text
    assert "ASR Model: Whisper large-v3-turbo" in text
    assert "Language Mode: auto" in text
    assert asr_label(task.asr_model) in text

    task.status = TaskStatus.failed
    task.error_stage = "downloading"
    task.error = "yt-dlp download failed: network"
    writer.update_progress(path, task, metadata)
    failed = path.read_text(encoding="utf-8")
    assert "Status: failed" in failed
    assert "Error Stage: downloading" in failed

    parsed = NoteDocument.parse(failed)
    assert parsed.fields["Author"] == "Example Author"
    assert parsed.fields["Duration"] == "24:31"
