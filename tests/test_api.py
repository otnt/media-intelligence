from pathlib import Path

from fastapi.testclient import TestClient

from media_pipeline.api import create_app
from media_pipeline.config import AppConfig, PathsConfig
from media_pipeline.models import Task
from media_pipeline.store import TaskStore


class DummyWorker:
    def __init__(self) -> None:
        self.submitted: list[str] = []
        self.summarized: list[tuple[str, str]] = []
        self.limits = {"youtube": 10, "bilibili": 10, "other": 10, "model_jobs": 1}
        self._summarizing: set[str] = set()

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def submit(self, task: Task) -> None:
        self.submitted.append(task.id)

    def retry(self, task: Task, stage: str | None = None, visual: dict | None = None) -> Task:
        self.submitted.append(task.id)
        return task

    def summarize(self, task: Task, prompt: str = "") -> dict:
        from media_pipeline.summary import DEFAULT_PROMPT

        self.summarized.append((task.id, prompt))
        self._summarizing.add(task.id)
        return {
            "status": "running",
            "prompt": prompt or DEFAULT_PROMPT,
            "markdown": "",
            "model": "",
            "error": "",
            "image_count": 0,
            "updated_at": "",
        }

    def set_limits(self, **kwargs: int) -> None:
        for key, value in kwargs.items():
            if value is not None:
                self.limits[key] = value

    def snapshot(self) -> dict[str, dict[str, int]]:
        return {key: {"limit": value, "running": 0} for key, value in self.limits.items()}


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
    assert payload["language"] == "auto"
    assert worker.submitted == [payload["id"]]
    stored = store.get(payload["id"])
    assert stored is not None
    assert stored.extra["language"] == "auto"


def test_create_task_accepts_xiaohongshu_url(tmp_path: Path):
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
        json={"url": "http://xhslink.com/o/2x5jqGA2hr6", "asr_model": "qwen3-asr-1.7b"},
    )
    assert response.status_code == 202
    stored = store.get(response.json()["id"])
    assert stored is not None
    assert stored.platform == "Xiaohongshu"
    assert stored.video_id == "2x5jqGA2hr6"


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


def test_models_mark_qwen_as_multilingual(tmp_path: Path):
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
    payload = client.get("/v1/models").json()
    assert payload["default"] == "qwen3-asr-1.7b"
    health = client.get("/v1/health").json()
    assert health["default_asr_model"] == "qwen3-asr-1.7b"
    assert health["default_language"] == "auto"
    assert health["worker"]["youtube"]["limit"] == 10
    assert health["worker"]["bilibili"]["limit"] == 10
    assert health["worker"]["model_jobs"]["limit"] == 1
    by_id = {item["id"]: item for item in payload["models"]}
    assert by_id["qwen3-asr-1.7b"]["code_switching"] is True
    assert by_id["whisper-large-v3-turbo"]["code_switching"] is False


def test_create_task_accepts_explicit_language(tmp_path: Path):
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
    client = TestClient(create_app(config, store=store, worker=DummyWorker()))
    response = client.post(
        "/v1/tasks",
        json={
            "url": "https://www.bilibili.com/video/BV181KNeuEi2",
            "asr_model": "qwen3-asr-1.7b",
            "language": "auto",
        },
    )
    assert response.status_code == 202
    assert response.json()["language"] == "auto"
    stored = store.get(response.json()["id"])
    assert stored is not None
    assert stored.extra["language"] == "auto"
    assert stored.asr_model == "qwen3-asr-1.7b"


def test_dashboard_and_frame_override(tmp_path: Path):
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
    home = client.get("/")
    assert home.status_code == 200
    assert "Extraction dashboard" in home.text
    assert "Started" in home.text
    assert "class=\"help\"" in home.text
    assert "Sample interval" in home.text
    assert "Qwen3.8 frame filter" in home.text
    assert "vlm_keep_threshold" in home.text
    assert "Generate summary" in home.text
    assert "提取这个文章的核心思想" in home.text
    visual = client.get("/v1/visual/config").json()
    assert visual["visual"]["vlm_keep_threshold"] == 0.45
    assert "filtering_frames" in visual["stages"]
    assert "Concurrency" in home.text
    assert "YouTube max" in home.text
    assert "Bilibili max" in home.text
    created = client.post(
        "/v1/tasks",
        json={"url": "https://www.bilibili.com/video/BV181KNeuEi2", "asr_model": "qwen3-asr-1.7b"},
    )
    task_id = created.json()["id"]
    rerun = client.post(
        f"/v1/tasks/{task_id}/retry",
        json={"stage": "filtering_frames", "visual": {"vlm_keep_threshold": 0.5}},
    )
    assert rerun.status_code == 200
    assert worker.submitted[-1] == task_id
    decision = client.post(
        "/v1/videos/BV181KNeuEi2/frames/00-00-01.000.jpg/decision",
        json={"decision": "keep"},
    )
    assert decision.status_code == 200
    assert decision.json()["overrides"]["00-00-01.000.jpg"] == "keep"
    debug = client.get(f"/v1/tasks/{task_id}/debug")
    assert debug.status_code == 200
    assert "debug" in debug.json()
    assert debug.json()["summary"]["status"] == "idle"
    started = client.post(f"/v1/tasks/{task_id}/summary", json={"prompt": "一句话概括"})
    assert started.status_code == 202
    assert started.json()["status"] == "running"
    assert started.json()["prompt"] == "一句话概括"
    assert worker.summarized == [(task_id, "一句话概括")]


def test_update_worker_limits(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("# keep this comment\nserver:\n  host: 127.0.0.1\n", encoding="utf-8")
    config = AppConfig(
        paths=PathsConfig(
            videos=tmp_path / "videos",
            audio=tmp_path / "audio",
            artifacts=tmp_path / "artifacts",
            logs=tmp_path / "logs",
            vault=tmp_path / "vault",
            db=tmp_path / "tasks.sqlite3",
        ),
        source_path=config_path,
    )
    config.ensure_directories()
    worker = DummyWorker()
    client = TestClient(create_app(config, store=TaskStore(config.paths.db), worker=worker))
    empty = client.put("/v1/worker", json={})
    assert empty.status_code == 400
    response = client.put("/v1/worker", json={"youtube": 3, "bilibili": 7, "model_jobs": 2})
    assert response.status_code == 200
    payload = response.json()["worker"]
    assert payload["youtube"]["limit"] == 3
    assert payload["bilibili"]["limit"] == 7
    assert payload["model_jobs"]["limit"] == 2
    assert worker.limits["youtube"] == 3
    assert config.worker.youtube == 3
    saved = config_path.read_text(encoding="utf-8")
    assert "# keep this comment" in saved
    assert "youtube: 3" in saved
    assert "bilibili: 7" in saved
    status = client.get("/v1/worker").json()
    assert status["youtube"]["limit"] == 3
    invalid = client.put("/v1/worker", json={"youtube": -1})
    assert invalid.status_code == 422


