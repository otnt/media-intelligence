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

    def summarize(self, task: Task, prompt: str = "", model: str = "") -> dict:
        from media_pipeline.summary import DEFAULT_PROMPT

        self.summarized.append((task.id, prompt, model))
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


def test_create_task_extracts_url_from_share_text_and_defaults_asr(tmp_path: Path):
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
        json={"url": "标题\nhttp://xhslink.com/o/2x5jqGA2hr6\n复制后打开"},
    )
    assert response.status_code == 202
    payload = response.json()
    assert payload["asr_model"] == "qwen3-asr-1.7b"
    assert payload["url"] == "http://xhslink.com/o/2x5jqGA2hr6"
    stored = store.get(payload["id"])
    assert stored is not None
    assert stored.video_id == "2x5jqGA2hr6"


def test_create_task_accepts_form_post(tmp_path: Path):
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
        data={"url": "http://xhslink.com/o/2x5jqGA2hr6"},
    )
    assert response.status_code == 202
    assert response.json()["url"] == "http://xhslink.com/o/2x5jqGA2hr6"


def test_inbox_get_queues_share_text(tmp_path: Path):
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
    response = client.get(
        "/v1/inbox",
        params={"url": "标题 http://xhslink.com/o/2x5jqGA2hr6"},
    )
    assert response.status_code == 202
    payload = response.json()
    assert payload["message"] == "Added to queue"
    assert payload["url"] == "http://xhslink.com/o/2x5jqGA2hr6"
    stored = store.get(payload["id"])
    assert stored is not None
    assert stored.video_id == "2x5jqGA2hr6"


def test_create_task_requires_ingest_token_for_remote_clients(tmp_path: Path):
    from media_pipeline.config import ServerConfig

    config = AppConfig(
        server=ServerConfig(ingest_token="secret-token"),
        paths=PathsConfig(
            videos=tmp_path / "videos",
            audio=tmp_path / "audio",
            artifacts=tmp_path / "artifacts",
            logs=tmp_path / "logs",
            vault=tmp_path / "vault",
            db=tmp_path / "tasks.sqlite3",
        ),
    )
    config.ensure_directories()
    client = TestClient(create_app(config, store=TaskStore(config.paths.db), worker=DummyWorker()))
    denied = client.post("/v1/tasks", json={"url": "http://xhslink.com/o/2x5jqGA2hr6"})
    assert denied.status_code == 401
    allowed = client.post(
        "/v1/tasks",
        json={"url": "http://xhslink.com/o/2x5jqGA2hr6"},
        headers={"X-Ingest-Token": "secret-token"},
    )
    assert allowed.status_code == 202


def test_create_task_accepts_rednote_url(tmp_path: Path):
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
            "url": "https://www.rednote.com/explore/64aaaaaaaaaaaaaaaaaaaaaa?xsec_token=abc",
            "asr_model": "qwen3-asr-1.7b",
        },
    )
    assert response.status_code == 202
    stored = store.get(response.json()["id"])
    assert stored is not None
    assert stored.platform == "Xiaohongshu"
    assert stored.video_id == "64aaaaaaaaaaaaaaaaaaaaaa"


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


def test_create_task_defaults_keyframes_off(tmp_path: Path):
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
        json={"url": "https://www.bilibili.com/video/BV181KNeuEi2", "asr_model": "whisper-large-v3-turbo"},
    )
    assert response.status_code == 202
    payload = response.json()
    assert payload["extract_keyframes"] is False
    stored = store.get(payload["id"])
    assert stored is not None
    assert stored.extra["extract_keyframes"] is False
    listed = client.get("/v1/tasks").json()["tasks"][0]
    assert listed["extract_keyframes"] is False


def test_create_task_accepts_extract_keyframes(tmp_path: Path):
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
            "asr_model": "whisper-large-v3-turbo",
            "extract_keyframes": True,
        },
    )
    assert response.status_code == 202
    payload = response.json()
    assert payload["extract_keyframes"] is True
    stored = store.get(payload["id"])
    assert stored is not None
    assert stored.extra["extract_keyframes"] is True


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
    assert "Regenerate summary" in home.text
    assert "Compare all available" in home.text
    assert "summary-model" in home.text
    assert "/v1/summary/models" in home.text
    assert "Extracted contents" in home.text
    assert "fold-frames" in home.text
    assert "Extract keyframes" in home.text
    assert "Transcript only" in home.text
    assert "Keyframes on" in home.text
    assert "提取核心思想" in home.text
    assert "不要插入任何图片" in home.text
    visual = client.get("/v1/visual/config").json()
    assert visual["visual"]["vlm_keep_threshold"] == 0.45
    assert "filtering_frames" in visual["stages"]
    assert "summarizing" in visual["stages"]
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
    assert worker.summarized == [(task_id, "一句话概括", "")]
    compared = client.post(f"/v1/tasks/{task_id}/summary", json={"prompt": "一句话概括", "model": "all"})
    assert compared.status_code == 202
    assert worker.summarized[-1] == (task_id, "一句话概括", "all")


def test_summary_models_endpoint(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
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
    payload = client.get("/v1/summary/models").json()
    assert payload["default"] == ["qwen"]
    by_key = {item["key"]: item for item in payload["providers"]}
    assert set(by_key) == {"qwen", "gemini", "openai"}
    assert by_key["gemini"]["available"] is False
    assert by_key["openai"]["available"] is False
    assert "GEMINI_API_KEY" in by_key["gemini"]["detail"]
    assert "OPENAI_API_KEY" in by_key["openai"]["detail"]


def test_attachment_endpoint_serves_vault_and_video_images(tmp_path: Path):
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
    note_dir = config.notes_dir()
    assert note_dir is not None
    attachment_dir = note_dir / "attachments" / "note123"
    attachment_dir.mkdir(parents=True)
    image = attachment_dir / "01.jpg"
    image.write_bytes(b"jpeg-bytes")
    client = TestClient(create_app(config, store=TaskStore(config.paths.db), worker=DummyWorker()))
    response = client.get("/v1/videos/note123/attachments/01.jpg")
    assert response.status_code == 200
    assert response.content == b"jpeg-bytes"
    missing = client.get("/v1/videos/note123/attachments/missing.jpg")
    assert missing.status_code == 404


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


