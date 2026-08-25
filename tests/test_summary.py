from pathlib import Path

from media_pipeline.artifacts import ArtifactStore
from media_pipeline.models import NamedSegment, Task, TaskStatus, VideoMetadata
from media_pipeline.notes import NoteDocument, NoteWriter, load_note
from media_pipeline.summary import (
    DEFAULT_PROMPT,
    LEGACY_PROMPT,
    SummaryStill,
    build_model_prompt,
    choose_images,
    collect_stills,
    normalize_prompt,
    summarize_task,
    upsert_summary_section,
)
from media_pipeline.visual.models import FrameVerdict, Keyframe


class FakeVision:
    name = "fake"
    model_id = "fake-vlm"

    def __init__(self) -> None:
        self.prompt = ""
        self.images: list[Path] = []
        self.max_tokens = 0

    def generate(self, prompt: str, images=None, max_tokens=None) -> str:
        self.prompt = prompt
        self.images = list(images or [])
        self.max_tokens = int(max_tokens or 0)
        return "<think>skip</think>## 核心思想\n\n这是要点。\n\n![[attachments/vid/keep.jpg]]\n"


def _metadata() -> VideoMetadata:
    return VideoMetadata(
        url="https://www.bilibili.com/video/BVxxxx",
        title="How to listen",
        platform="Bilibili",
        author="Ada",
        video_id="BVxxxx",
        duration=12.0,
        published="2026-08-01",
        description="",
        thumbnail_url="",
        asr_model="qwen3-asr-1.7b",
    )


def test_choose_images_keeps_highest_scores_in_time_order():
    stills = [
        SummaryStill("c.jpg", 30, "", Path("c.jpg"), score=0.2),
        SummaryStill("a.jpg", 10, "", Path("a.jpg"), score=0.9),
        SummaryStill("b.jpg", 20, "", Path("b.jpg"), score=0.8),
    ]
    chosen = choose_images(stills, limit=2)
    assert [item.filename for item in chosen] == ["a.jpg", "b.jpg"]


def test_build_prompt_is_text_only():
    text = build_model_prompt("请总结", _metadata(), "[00:00:00] Speaker 1: hello")
    assert "请总结" in text
    assert "How to listen" in text
    assert "Speaker 1: hello" in text
    assert "Transcript:" in text
    assert "keep.jpg" not in text
    assert "![[attachments" not in text
    assert "Plain text only" in text


def test_build_prompt_labels_image_post_as_original():
    metadata = VideoMetadata(
        url="https://www.xiaohongshu.com/explore/note1",
        title="币圈大佬",
        platform="Xiaohongshu",
        author="产联社",
        video_id="note1",
        duration=0.0,
        published="2026-08-13",
        description="8月7日凌晨4点30分，叶俊德坠亡。",
        thumbnail_url="",
        asr_model="",
        media_kind="image",
    )
    text = build_model_prompt(DEFAULT_PROMPT, metadata, metadata.description)
    assert "Original post:" in text
    assert "Transcript:" not in text
    assert metadata.description in text
    assert "Do not copy the source sentence by sentence" in text


def test_normalize_prompt_replaces_legacy_illustrated_default():
    assert normalize_prompt("") == DEFAULT_PROMPT
    assert normalize_prompt(LEGACY_PROMPT) == DEFAULT_PROMPT
    assert normalize_prompt("自定义摘要") == "自定义摘要"


def test_upsert_summary_replaces_existing_block():
    body = "## Summary\n\nold\n\n## Transcript\n\nhello\n"
    updated = upsert_summary_section(body, "new briefing")
    assert updated.startswith("## Summary\n\nnew briefing\n")
    assert "## Transcript" in updated
    assert "old" not in updated


def test_summarize_task_uses_transcript_without_images(tmp_path: Path):
    artifacts = ArtifactStore(tmp_path / "artifacts", "BVxxxx")
    artifacts.save_metadata(_metadata())
    frame_dir = artifacts.keyframe_dir
    frame_dir.mkdir(parents=True, exist_ok=True)
    keep = frame_dir / "keep.jpg"
    drop = frame_dir / "drop.jpg"
    keep.write_bytes(b"keep")
    drop.write_bytes(b"drop")
    artifacts.save_keyframes(
        [
            Keyframe(timestamp=1.5, image_path="keyframes/keep.jpg"),
            Keyframe(timestamp=2.5, image_path="keyframes/drop.jpg"),
        ]
    )
    artifacts.save_frame_analysis(
        [
            FrameVerdict("keep.jpg", 1.5, True, 0.9, "slide", "", "a slide", True),
            FrameVerdict("drop.jpg", 2.5, False, 0.1, "talking_head", "", "", False),
        ]
    )
    artifacts.save_named(
        [NamedSegment(start=0.0, end=4.0, speaker_id="A", speaker_label="Speaker 1", text="Welcome.")]
    )
    vision = FakeVision()
    result = summarize_task(artifacts, vision, prompt="", metadata=_metadata())
    assert result["status"] == "completed"
    assert result["prompt"] == DEFAULT_PROMPT
    assert result["image_count"] == 0
    assert "核心思想" in result["markdown"]
    assert "<think>" not in result["markdown"]
    assert "![[" not in result["markdown"]
    assert vision.images == []
    assert "Welcome." in vision.prompt
    assert "Plain text only" in vision.prompt
    stills = collect_stills(artifacts)
    assert [item.filename for item in stills] == ["keep.jpg"]


def test_summarize_image_post_uses_caption_not_photos(tmp_path: Path):
    metadata = VideoMetadata(
        url="https://www.xiaohongshu.com/explore/note1",
        title="币圈大佬",
        platform="Xiaohongshu",
        author="产联社",
        video_id="note1",
        duration=0.0,
        published="2026-08-13",
        description="8月7日凌晨4点30分，叶俊德坠亡。管理资产超24亿美元。",
        thumbnail_url="",
        asr_model="",
        media_kind="image",
    )
    artifacts = ArtifactStore(tmp_path / "artifacts", "note1")
    artifacts.save_metadata(metadata)
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    (photo_dir / "01.jpg").write_bytes(b"photo")
    vision = FakeVision()
    result = summarize_task(artifacts, vision, prompt=LEGACY_PROMPT, metadata=metadata)
    assert result["status"] == "completed"
    assert result["prompt"] == DEFAULT_PROMPT
    assert result["image_count"] == 0
    assert vision.images == []
    assert "Original post:" in vision.prompt
    assert metadata.description in vision.prompt
    assert "![[" not in result["markdown"]


def test_note_writer_inserts_summary_above_transcript(tmp_path: Path):
    metadata = _metadata()
    task = Task(
        id="t1",
        url=metadata.url,
        asr_model="qwen3-asr-1.7b",
        status=TaskStatus.completed,
        video_id=metadata.video_id,
    )
    writer = NoteWriter(tmp_path)
    path = writer.create_or_open(metadata, task)
    writer.update_progress(path, task, metadata, body="## Transcript\n\nhello\n")
    writer.update_summary(path, "一分钟读完")
    text = path.read_text(encoding="utf-8")
    assert text.index("## Summary") < text.index("URL:")
    assert text.index("## Summary") < text.index("## Transcript")
    assert "一分钟读完" in text
    parsed = NoteDocument.parse(text)
    assert parsed.summary_markdown == "一分钟读完"
    assert parsed.transcript_markdown.startswith("## Transcript")
    writer.update_progress(path, task, metadata, body="## Transcript\n\nreplaced\n")
    kept = path.read_text(encoding="utf-8")
    assert "一分钟读完" in kept
    assert "replaced" in kept


def test_load_note_moves_legacy_summary_above_metadata(tmp_path: Path):
    path = tmp_path / "legacy.md"
    path.write_text(
        "# Title\n\nURL: https://example.com/v\nPlatform: Bilibili\nKind: Video\n"
        "Author: A\nStatus: completed\nVideo Path: /tmp/v\nCreated: now\nUpdated: now\n"
        "\n---\n\n## Summary\n\nbriefing\n\n## Transcript\n\nhello\n",
        encoding="utf-8",
    )
    document = load_note(path, rewrite_layout=True)
    text = path.read_text(encoding="utf-8")
    assert text.index("## Summary") < text.index("URL:")
    assert document.summary_markdown == "briefing"
    assert document.transcript_markdown.startswith("## Transcript")
    assert "## Summary" not in document.transcript_markdown
