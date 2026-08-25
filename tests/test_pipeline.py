from pathlib import Path

from media_pipeline.artifacts import ArtifactStore
from media_pipeline.config import AppConfig, PathsConfig
from media_pipeline.diarization import NullDiarizationProvider
from media_pipeline.models import (
    ASROptions,
    Task,
    TaskStatus,
    Transcript,
    TranscriptSegment,
    VideoMetadata,
)
from media_pipeline.pipeline import RERUN_STAGES, Pipeline
from media_pipeline.store import TaskStore


class FakeVisual:
    def run(self, video_path, artifacts, metadata, named, settings, from_stage="", progress=None):
        from media_pipeline.visual.align import align_keyframes, build_multimodal_document
        from media_pipeline.visual.models import Keyframe, SceneSpan

        if progress:
            progress("detecting_scenes", {})
            progress("sampling_frames", {"scene_count": 1})
            progress("deduplicating_frames", {"candidate_count": 0})
            progress("aligning_multimodal", {"keyframe_count": 1})
        artifacts.save_scenes([SceneSpan(0.0, float(metadata.duration or 0.0))])
        artifacts.save_candidates([])
        keyframes = [
            Keyframe(timestamp=1.0, image_path="keyframes/00-00-01.000.jpg", sources=["periodic"]),
        ]
        artifacts.save_keyframes(keyframes)
        timeline = align_keyframes(keyframes, named, before_sec=10, after_sec=20)
        document = build_multimodal_document(metadata, named, keyframes, timeline)
        artifacts.save_multimodal(document)
        return {
            "scenes": [],
            "candidates": [],
            "keyframes": keyframes,
            "selected": keyframes,
            "analysis": [],
            "timeline": timeline,
            "document": document,
        }


class FakeASR:
    def __init__(self) -> None:
        self.calls = 0
        self.last_options: ASROptions | None = None

    def transcribe(self, audio_path: Path, options: ASROptions | None = None) -> Transcript:
        self.calls += 1
        self.last_options = options
        return Transcript(
            language="en",
            provider="fake",
            model="whisper-large-v3-turbo",
            segments=[TranscriptSegment(start=0.0, end=4.0, text="Welcome to today's discussion.")],
        )


def _config(tmp_path: Path) -> AppConfig:
    config = AppConfig(
        paths=PathsConfig(
            videos=tmp_path / "videos",
            audio=tmp_path / "audio",
            artifacts=tmp_path / "artifacts",
            logs=tmp_path / "logs",
            vault=tmp_path / "vault",
            db=tmp_path / "tasks.sqlite3",
        )
    )
    config.ensure_directories()
    return config


def test_pipeline_reuses_download_and_audio(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    store = TaskStore(config.paths.db)
    asr = FakeASR()
    metadata = VideoMetadata(
        url="https://www.bilibili.com/video/BVxxxx",
        title="How to Quickly Break the Ice and Build Relationships",
        platform="Bilibili",
        author="Example Author",
        video_id="BVxxxx",
        duration=24.0,
        published="2026-08-01",
        description="",
        thumbnail_url="",
        asr_model="whisper-large-v3-turbo",
    )

    downloads = {"count": 0}
    extracts = {"count": 0}

    def fake_fetch(url, asr_model, cfg):
        return metadata

    def fake_download(url, video_id, dest_dir, cfg):
        downloads["count"] += 1
        path = dest_dir / f"{video_id}.mp4"
        path.write_bytes(b"video")
        return path

    def fake_extract(video_path, audio_path):
        extracts["count"] += 1
        audio_path.write_bytes(b"audio")
        return audio_path

    monkeypatch.setattr("media_pipeline.pipeline.fetch_metadata", fake_fetch)
    monkeypatch.setattr("media_pipeline.pipeline.download_video", fake_download)
    monkeypatch.setattr("media_pipeline.pipeline.extract_audio", fake_extract)
    monkeypatch.setattr("media_pipeline.pipeline.require_provider", lambda model_id: asr)

    pipeline = Pipeline(config, store, diarization=NullDiarizationProvider(), visual=FakeVisual())
    first = store.insert(
        Task(
            id="task-1",
            url=metadata.url,
            asr_model="whisper-large-v3-turbo",
            video_id="BVxxxx",
        )
    )
    result = pipeline.run(first)
    assert result.status == TaskStatus.completed
    assert downloads["count"] == 1
    assert extracts["count"] == 1
    assert asr.calls == 1
    assert asr.last_options is not None
    assert asr.last_options.language is None
    note = Path(result.note_path).read_text(encoding="utf-8")
    assert "Status: completed" in note
    assert "Language Mode: auto" in note
    assert "Languages: en" in note
    assert "### [00:00:00] Speaker 1" in note
    assert "## Visual Timeline" not in note
    assert note.index("![[attachments/BVxxxx/00-00-01.000.jpg]]") < note.index("### [00:00:00] Speaker 1")
    assert ArtifactStore(config.paths.artifacts, "BVxxxx").asr_path("whisper-large-v3-turbo").exists()
    assert ArtifactStore(config.paths.artifacts, "BVxxxx").multimodal_path.exists()
    timings = result.extra.get("stage_timings") or {}
    assert timings["transcribing"]["status"] == "succeeded"
    assert "started_at" in timings["transcribing"]
    assert timings["transcribing"]["duration_sec"] >= 0
    persisted = ArtifactStore(config.paths.artifacts, "BVxxxx").load_stage_timings()
    assert persisted["transcribing"]["status"] == "succeeded"
    assert persisted["writing_outputs"]["status"] == "succeeded"

    second = store.insert(
        Task(
            id="task-2",
            url=metadata.url,
            asr_model="whisper-large-v3-turbo",
            video_id="BVxxxx",
        )
    )
    reused = pipeline.run(second)
    assert reused.status == TaskStatus.completed
    assert downloads["count"] == 1
    assert extracts["count"] == 1
    assert asr.calls == 1


def test_pipeline_writes_failure_note_when_metadata_fails(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    store = TaskStore(config.paths.db)
    monkeypatch.setattr(
        "media_pipeline.pipeline.fetch_metadata",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("yt-dlp exploded")),
    )
    pipeline = Pipeline(config, store, diarization=NullDiarizationProvider(), visual=FakeVisual())
    task = store.insert(
        Task(
            id="task-fail",
            url="https://www.bilibili.com/video/BV181KNeuEi2",
            asr_model="whisper-large-v3-turbo",
        )
    )
    result = pipeline.run(task)
    assert result.status == TaskStatus.failed
    assert result.error_stage == "fetching_metadata"
    failed = (result.extra.get("stage_timings") or {}).get("fetching_metadata") or {}
    assert failed.get("status") == "failed"
    assert "duration_sec" not in failed
    assert failed.get("started_at")
    assert result.note_path
    text = Path(result.note_path).read_text(encoding="utf-8")
    assert "Status: failed" in text
    assert "Error Stage: fetching_metadata" in text


def test_rerun_stages_include_frame_filter():
    assert "filtering_frames" in RERUN_STAGES
    assert "deduplicating_frames" in RERUN_STAGES

