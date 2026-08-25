"""End-to-end: Explore Extract click → CREATE_TASK → API queue → download."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from media_pipeline.api import create_app
from media_pipeline.config import AppConfig, PathsConfig
from media_pipeline.diarization import NullDiarizationProvider
from media_pipeline.models import Task, TaskStatus, VideoMetadata
from media_pipeline.pipeline import Pipeline
from media_pipeline.store import TaskStore
from media_pipeline.xhs import _POST_CACHE, download_xhs_post, extract_xhs_post

from tests.test_pipeline import FakeVisual

ROOT = Path(__file__).resolve().parents[1]
CLICKED_NOTE_ID = "64bbbbbbbbbbbbbbbbbbbbbb"
JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n"
    b"\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d"
    b"\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342"
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
    b"\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x08\xff\xda\x00\x08\x01\x01\x00\x00?\x00\x54\xbf"
    b"\xff\xd9"
)


def _chrome_bin() -> str | None:
    env = os.environ.get("CHROME_BIN")
    candidates = [
        env,
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return path
    return None


class _FixtureHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, posted=None, **kwargs):
        self._posted = posted
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format, *args):
        return None

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            payload = {"raw": raw.decode("utf-8", errors="replace")}
        self._posted.append(payload)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()


def _serve_fixture():
    posted: list[dict] = []
    ready = threading.Event()

    def factory(*args, **kwargs):
        return _FixtureHandler(*args, posted=posted, **kwargs)

    server = ThreadingHTTPServer(("127.0.0.1", 0), factory)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    ready.set()
    url = f"http://127.0.0.1:{server.server_address[1]}/tests/fixtures/rednote_explore.html?autotest=1"
    return server, posted, url


def _wait_for_post(posted: list[dict], proc: subprocess.Popen, kind: str, timeout: float = 20.0) -> dict:
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        for item in list(posted):
            if item.get("type") == kind:
                return item
            messages = item.get("messages") or []
            for message in messages:
                if message.get("type") == kind:
                    return message
            extra = item.get("extra") if item.get("type") == "TEST_STATUS" else None
            if kind == "CREATE_TASK" and extra and extra.get("url"):
                return {"type": "CREATE_TASK", "url": extra["url"], "asr_model": extra.get("asr_model")}
            if item.get("type") == "TEST_STATUS" and item.get("status") == "fail":
                raise AssertionError(item.get("extra") or item)
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    raise AssertionError(f"{kind} never arrived; posts={posted!r} exit={proc.poll()}")


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
    (config.paths.vault / "Transcripts").mkdir(parents=True, exist_ok=True)
    return config


def _html_for_image_note(note_id: str) -> str:
    note = {
        "noteId": note_id,
        "type": "normal",
        "title": "Street photos",
        "desc": "a walk",
        "time": 1700000000000,
        "user": {"nickname": "Nana"},
        "imageList": [
            {"urlDefault": "http://sns-webpic-qc.xhscdn.com/20240101/h/notes_pre_post/one!nd"},
            {"urlDefault": "http://sns-webpic-qc.xhscdn.com/20240101/h/notes_pre_post/two!nd"},
        ],
    }
    state = {"note": {"noteDetailMap": {note_id: {"note": note}}}}
    return f"<html><script>window.__INITIAL_STATE__={json.dumps(state)}</script></html>"


class _FakeResp:
    def __init__(self, url: str, text: str = "", content: bytes = b"", status_code: int = 200, content_type: str = "text/html"):
        self.url = url
        self.text = text
        self.content = content
        self.status_code = status_code
        self.headers = {"content-type": content_type}

    def iter_content(self, chunk_size=1024 * 1024):
        yield self.content


def test_extract_click_queues_and_downloads_rednote_post(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    chrome = _chrome_bin()
    if not chrome:
        pytest.skip("Chrome is required to simulate the Explore Extract click")

    server, posted, page_url = _serve_fixture()
    try:
        profile = tmp_path / "chrome-profile"
        profile.mkdir()
        proc = subprocess.Popen(
            [
                chrome,
                "--headless=new",
                "--disable-gpu",
                "--disable-extensions",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-background-networking",
                f"--user-data-dir={profile}",
                "--window-size=1280,900",
                page_url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            layout = _wait_for_post(posted, proc, "LAYOUT")
            clicked = _wait_for_post(posted, proc, "CREATE_TASK")
        finally:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.terminate()
    finally:
        server.shutdown()
        server.server_close()

    assert layout["cardCount"] == 3
    assert layout["buttonCount"] == 3
    assert not layout.get("overlay")
    assert all(row["buttons"] == 1 and not row["insideCover"] and row["noteId"] for row in layout["perCard"])
    assert clicked["type"] == "CREATE_TASK"
    assert clicked["url"].startswith(f"https://www.rednote.com/explore/{CLICKED_NOTE_ID}")
    assert "xsec_token=feed-token-def" in clicked["url"]
    assert clicked["asr_model"] == "qwen3-asr-1.7b"
    assert clicked.get("extract_keyframes") is False

    config = _config(tmp_path)
    store = TaskStore(config.paths.db)

    class RecordingWorker:
        def __init__(self) -> None:
            self.submitted: list[str] = []
            self.limits = {"youtube": 10, "bilibili": 10, "other": 10, "model_jobs": 1}

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
            return {}

        def set_limits(self, **kwargs: int) -> None:
            return None

        def snapshot(self) -> dict:
            return {key: {"limit": value, "running": 0} for key, value in self.limits.items()}

    worker = RecordingWorker()
    client = TestClient(create_app(config, store=store, worker=worker))
    response = client.post(
        "/v1/tasks",
        json={
            "url": clicked["url"],
            "asr_model": clicked["asr_model"],
            "extract_keyframes": bool(clicked.get("extract_keyframes")),
        },
    )
    assert response.status_code == 202, response.text
    payload = response.json()
    stored = store.get(payload["id"])
    assert stored is not None
    assert stored.platform == "Xiaohongshu"
    assert stored.video_id == CLICKED_NOTE_ID
    assert worker.submitted == [payload["id"]]

    _POST_CACHE.clear()
    html = _html_for_image_note(CLICKED_NOTE_ID)

    def fake_get(url, config, stream=False, stage="fetching_metadata", origin=""):
        if "rednote.com" in url or "xiaohongshu.com/explore" in url:
            return _FakeResp(url, text=html)
        return _FakeResp(url, content=JPEG, content_type="image/jpeg")

    monkeypatch.setattr("media_pipeline.xhs._http_get", fake_get)
    monkeypatch.setattr("media_pipeline.xhs._extract_with_xhs_downloader", lambda url: None)
    post = extract_xhs_post(clicked["url"], config)
    assert post.media_kind == "image"
    assert post.note_id == CLICKED_NOTE_ID
    dest = download_xhs_post(clicked["url"], CLICKED_NOTE_ID, config.paths.videos, config)
    images = sorted(path.name for path in dest.iterdir())
    assert images == ["01.jpg", "02.jpg"]
    assert (dest / "01.jpg").stat().st_size > 0

    metadata = VideoMetadata(
        url=clicked["url"],
        title="Street photos",
        platform="Xiaohongshu",
        author="Nana",
        video_id=CLICKED_NOTE_ID,
        duration=None,
        published="2023-11-14",
        description="a walk",
        thumbnail_url="",
        asr_model="qwen3-asr-1.7b",
        media_kind="image",
    )

    def fake_fetch(url, asr_model, cfg):
        return metadata

    def fake_download(url, video_id, dest_dir, cfg):
        folder = dest_dir / video_id
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dest / "01.jpg", folder / "01.jpg")
        shutil.copy2(dest / "02.jpg", folder / "02.jpg")
        return folder

    monkeypatch.setattr("media_pipeline.pipeline.fetch_metadata", fake_fetch)
    monkeypatch.setattr("media_pipeline.pipeline.download_video", fake_download)
    pipeline = Pipeline(config, store, diarization=NullDiarizationProvider(), visual=FakeVisual())
    result = pipeline.run(stored)
    assert result.status == TaskStatus.completed
    assert result.extra.get("media_kind") == "image"
    assert result.extra.get("image_count") == 2
    assert (config.notes_dir() / "attachments" / CLICKED_NOTE_ID / "01.jpg").exists()
    note = Path(result.note_path).read_text(encoding="utf-8")
    assert f"![[attachments/{CLICKED_NOTE_ID}/01.jpg]]" in note
    _POST_CACHE.clear()
