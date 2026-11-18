from pathlib import Path

from media_pipeline.config import AppConfig, PathsConfig, load_config


def _config(tmp_path: Path) -> AppConfig:
    config = AppConfig(paths=PathsConfig(videos=tmp_path / "v", audio=tmp_path / "a", artifacts=tmp_path / "art",
                                         logs=tmp_path / "l", vault=None, db=tmp_path / "t.sqlite3"))
    config.ensure_directories()
    return config


def test_vault_off_in_the_config_means_no_obsidian(tmp_path: Path):
    path = tmp_path / "c.yaml"
    path.write_text("paths:\n  vault: off\n", encoding="utf-8")
    assert load_config(path).paths.vault is None and load_config(path).notes_dir() is None


def test_the_pipeline_and_reader_work_without_a_vault(tmp_path: Path, monkeypatch):
    from fastapi.testclient import TestClient

    from media_pipeline.api import create_app
    from media_pipeline.diarization import NullDiarizationProvider
    from media_pipeline.models import Task, TaskStatus, VideoMetadata
    from media_pipeline.pipeline import Pipeline
    from media_pipeline.store import TaskStore
    from tests.test_api import DummyWorker
    from tests.test_pipeline import FakeVision, FakeVisual

    config = _config(tmp_path)
    store = TaskStore(config.paths.db)
    meta = VideoMetadata(url="https://www.threads.com/@a/post/ABC", title="A post", platform="Threads", author="a",
                         video_id="th_ABC", duration=None, published="", description="Just text, no images.",
                         thumbnail_url="", asr_model="fun-asr", media_kind="image")
    monkeypatch.setattr("media_pipeline.pipeline.fetch_metadata", lambda url, model, cfg: meta)
    monkeypatch.setattr("media_pipeline.pipeline.download_video",
                        lambda url, vid, dest, cfg: (dest / vid).mkdir(parents=True, exist_ok=True) or dest / vid)
    monkeypatch.setattr("media_pipeline.summary_llm.resolve_summary_backends", lambda *a, **k: [])
    pipeline = Pipeline(config, store, diarization=NullDiarizationProvider(), visual=FakeVisual(), vision=FakeVision())
    task = pipeline.run(store.insert(Task(id="t", url=meta.url, asr_model="fun-asr", video_id="th_ABC")))
    assert task.status == TaskStatus.completed and task.note_path == ""
    note = TestClient(create_app(config, store=store, worker=DummyWorker())).get("/v1/tasks/t/note").json()
    assert "Just text, no images." in str(note)
