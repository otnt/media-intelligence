from pathlib import Path

from fastapi.testclient import TestClient

from media_pipeline.api import create_app
from media_pipeline.config import AppConfig, PathsConfig
from media_pipeline.models import Task
from media_pipeline.store import TaskStore


class DummyWorker:
    def __init__(self) -> None:
        self.submitted: list[str] = []

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def submit(self, task: Task) -> None:
        self.submitted.append(task.id)

    def retry(self, task: Task) -> Task:
        self.submitted.append(task.id)
        return task


def test_create_task_accepts_bilibili_url(tmp_path: Path):
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
    store = TaskStore(config.paths.db)
    worker = DummyWorker()
    client = TestClient(create_app(config, store=store, worker=worker))
    response = client.post(
        "/v1/tasks",
        json={"url": "https://www.bilibili.com/video/BV181KNeuEi2", "asr_model": "whisper-large-v3-turbo"},
    )
    assert response.status_code == 202
    payload = response.json()
    assert payload["message"] == "Added to queue"
    assert payload["asr_label"] == "Whisper large-v3-turbo"
    assert worker.submitted == [payload["id"]]


def test_rejects_unknown_model_and_platform(tmp_path: Path):
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
    client = TestClient(create_app(config, store=TaskStore(config.paths.db), worker=DummyWorker()))
    bad_model = client.post(
        "/v1/tasks",
        json={"url": "https://www.youtube.com/watch?v=dQw4w9wgGcQ", "asr_model": "whisper-tiny"},
    )
    assert bad_model.status_code == 400
    bad_site = client.post(
        "/v1/tasks",
        json={"url": "https://twitter.com/x/status/1", "asr_model": "whisper-large-v3-turbo"},
    )
    assert bad_site.status_code == 400
