import shutil
import subprocess
import time
from pathlib import Path

import pytest

from media_pipeline.config import AppConfig, PathsConfig, StorageConfig
from media_pipeline.models import Task, TaskStatus, Transcript, TranscriptSegment, VideoMetadata
from media_pipeline.storage import cleanup, compact_audio_path, compact_video_media
from media_pipeline.store import TaskStore

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is required")


def _config(tmp_path: Path, **storage) -> AppConfig:
    config = AppConfig(
        paths=PathsConfig(
            videos=tmp_path / "videos", audio=tmp_path / "audio", artifacts=tmp_path / "artifacts",
            logs=tmp_path / "logs", vault=tmp_path / "vault", db=tmp_path / "tasks.sqlite3",
        ),
        storage=StorageConfig(**storage),
    )
    config.ensure_directories()
    return config


def _real_wav(path: Path, seconds: float = 3.0) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(path)],
        check=True,
    )
    return path


@needs_ffmpeg
@pytest.mark.parametrize("mode", ["archive", "keep", "delete"])
def test_compact_replaces_wav_and_handles_the_original_per_mode(tmp_path: Path, mode: str):
    config = _config(tmp_path, mode=mode, archive_dir=tmp_path / "icloud" / "videos")
    wav = _real_wav(config.paths.audio / "vid.wav")
    video = config.paths.videos / "vid.mp4"
    video.write_bytes(b"x" * 50_000)
    result = compact_video_media(config, "vid")
    m4a = compact_audio_path(config, "vid")
    assert not wav.exists() and result.audio == m4a and m4a.stat().st_size > 0
    archived = config.storage.archive_dir / "vid.mp4"
    if mode == "archive":
        assert not video.exists() and archived.read_bytes() == b"x" * 50_000 and result.archived == [archived]
    elif mode == "keep":
        assert video.exists() and not archived.exists()
    else:
        assert not video.exists() and not archived.exists() and video in result.removed


@needs_ffmpeg
def test_keep_audio_days_zero_and_delete_mode_leave_nothing(tmp_path: Path):
    config = _config(tmp_path, keep_audio_days=0, mode="delete")
    _real_wav(config.paths.audio / "vid.wav")
    (config.paths.videos / "vid.mp4").write_bytes(b"x")
    result = compact_video_media(config, "vid")
    assert result.audio is None
    assert list(config.paths.audio.iterdir()) == [] and list(config.paths.videos.iterdir()) == []


def test_unencodable_wav_is_kept_but_video_still_goes(tmp_path: Path):
    config = _config(tmp_path, mode="delete")
    wav = config.paths.audio / "vid.wav"
    wav.write_bytes(b"not really audio")
    video = config.paths.videos / "vid.mp4"
    video.write_bytes(b"x")
    result = compact_video_media(config, "vid")
    assert wav.exists() and result.audio == wav
    assert not video.exists()


@needs_ffmpeg
def test_cleanup_respects_activity_retention_orphans_and_dry_run(tmp_path: Path):
    config = _config(tmp_path, keep_audio_days=30, mode="delete")
    store = TaskStore(config.paths.db)
    for vid, status in [("done", TaskStatus.completed), ("busy", TaskStatus.transcribing), ("bad", TaskStatus.failed)]:
        store.insert(Task(id=f"t-{vid}", url=f"https://www.youtube.com/watch?v={vid}", asr_model="fun-asr",
                          video_id=vid, status=status, extra={"media_kind": "video"}))
        _real_wav(config.paths.audio / f"{vid}.wav")
        (config.paths.videos / f"{vid}.mp4").write_bytes(b"x" * 1000)
    (config.paths.videos / "stray.mp4").write_bytes(b"x")

    dry = cleanup(config, store, dry_run=True)
    assert dry.freed > 0 and (config.paths.videos / "done.mp4").exists()

    report = cleanup(config, store)
    assert not (config.paths.videos / "done.mp4").exists() and compact_audio_path(config, "done").exists()
    assert (config.paths.videos / "busy.mp4").exists() and "busy" in report.skipped_active
    assert (config.paths.videos / "bad.mp4").exists()  # failed, still inside the retry window
    assert (config.paths.videos / "stray.mp4").exists() and any(p.name == "stray.mp4" for p in report.orphans)
    assert store.get("t-done").video_path == ""

    later = cleanup(config, store, now=time.time() + 31 * 86400)
    assert not compact_audio_path(config, "done").exists()
    assert not (config.paths.videos / "bad.mp4").exists() and not (config.paths.audio / "bad.wav").exists()
    assert later.freed > 0


@needs_ffmpeg
def test_pipeline_compacts_after_success_and_rebuilds_wav_without_downloading(tmp_path: Path, monkeypatch):
    from tests.test_pipeline import FakeVision, FakeVisual, _config as pipeline_config
    from media_pipeline.diarization import NullDiarizationProvider
    from media_pipeline.pipeline import Pipeline

    config = pipeline_config(tmp_path)
    store = TaskStore(config.paths.db)
    metadata = VideoMetadata(url="https://www.youtube.com/watch?v=vidCompact1", title="t", platform="YouTube",
                             author="a", video_id="vidCompact1", duration=3.0, published="", description="",
                             thumbnail_url="", asr_model="x")
    downloads, sources = [], []

    def fake_download(url, video_id, dest_dir, cfg):
        downloads.append(video_id)
        path = dest_dir / f"{video_id}.mp4"
        _real_wav(tmp_path / "src.wav")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_path / "src.wav"), "-c:a", "aac", str(path)],
                       check=True)
        return path

    def real_extract(source, audio_path):
        sources.append(Path(source).suffix)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-ac", "1", "-ar", "16000",
                        "-c:a", "pcm_s16le", str(audio_path)], check=True)
        return audio_path

    class ASR:
        def transcribe(self, audio_path, options=None):
            assert Path(audio_path).exists()
            return Transcript(language="en", provider="p", model="m",
                              segments=[TranscriptSegment(0.0, 2.0, "hello")])

    monkeypatch.setattr("media_pipeline.pipeline.fetch_metadata", lambda url, model, cfg: metadata)
    monkeypatch.setattr("media_pipeline.pipeline.download_video", fake_download)
    monkeypatch.setattr("media_pipeline.pipeline.extract_audio", real_extract)
    monkeypatch.setattr("media_pipeline.pipeline.require_provider", lambda model_id: ASR())
    pipeline = Pipeline(config, store, diarization=NullDiarizationProvider(), visual=FakeVisual(), vision=FakeVision())

    first = pipeline.run(store.insert(Task(id="a", url=metadata.url, asr_model="whisper-large-v3-turbo",
                                           video_id="vidCompact1", extra={})))
    assert first.status == TaskStatus.completed
    archived = config.storage.archive_dir / "vidCompact1.mp4"
    assert config.storage.mode == "archive" and archived.exists()  # the original is kept, in "iCloud"
    assert not (config.paths.videos / "vidCompact1.mp4").exists()
    assert not (config.paths.audio / "vidCompact1.wav").exists()
    assert compact_audio_path(config, "vidCompact1").exists()
    assert first.video_path == str(archived) and first.audio_path.endswith(".m4a")

    # Re-sharing the finished video: everything is cached, nothing is downloaded or extracted.
    again = pipeline.run(store.insert(Task(id="b", url=metadata.url, asr_model="whisper-large-v3-turbo",
                                           video_id="vidCompact1", extra={})))
    assert again.status == TaskStatus.completed and downloads == ["vidCompact1"] and sources == [".mp4"]

    # A different ASR model needs audio: rebuilt from the compact copy, still no download.
    other = pipeline.run(store.insert(Task(id="c", url=metadata.url, asr_model="whisper-large-v3",
                                           video_id="vidCompact1", extra={})))
    assert other.status == TaskStatus.completed
    assert downloads == ["vidCompact1"] and sources == [".mp4", ".m4a"]

    # Keyframes later need the video itself: it comes back from the archive, not from YouTube.
    fetched = []
    monkeypatch.setattr("media_pipeline.pipeline.materialize", lambda path: fetched.append(path) or path)
    frames = pipeline.run(store.insert(Task(id="d", url=metadata.url, asr_model="whisper-large-v3",
                                            video_id="vidCompact1", extra={"rerun_stage": "detecting_scenes"})))
    assert frames.status == TaskStatus.completed
    assert downloads == ["vidCompact1"] and fetched == [archived]


def test_eviction_sweep_frees_only_what_icloud_accepted(tmp_path: Path, monkeypatch):
    import media_pipeline.storage as storage

    config = _config(tmp_path, mode="archive", archive_dir=tmp_path / "icloud" / "videos")
    config.storage.archive_dir.mkdir(parents=True)
    uploaded = config.storage.archive_dir / "up.mp4"
    pending = config.storage.archive_dir / "pending.mp4"
    uploaded.write_bytes(b"x" * 10)
    pending.write_bytes(b"y" * 10)
    evicted = set()

    def fake_run(command, **kwargs):
        if command[:2] == ["brctl", "evict"] and Path(command[2]).name == "up.mp4":
            evicted.add(Path(command[2]).name)
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 1, "", "cannot be evicted")  # still uploading

    monkeypatch.setattr(storage.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(storage.subprocess, "run", fake_run)
    monkeypatch.setattr(storage, "is_dataless", lambda path: path.name in evicted)
    report = cleanup(config, TaskStore(config.paths.db))
    assert report.evicted == [uploaded] and report.evicted_bytes == 10
    assert report.awaiting_upload == [pending]
