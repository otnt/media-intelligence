import threading
import time
from pathlib import Path

from media_pipeline.config import AppConfig, PathsConfig, WorkerConfig
from media_pipeline.models import Task, TaskStatus
from media_pipeline.store import TaskStore
from media_pipeline.worker import DomainSlots, TaskWorker, platform_key


def wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _config(tmp_path: Path, **worker_kwargs: int) -> AppConfig:
    config = AppConfig(
        paths=PathsConfig(
            videos=tmp_path / "videos",
            audio=tmp_path / "audio",
            artifacts=tmp_path / "artifacts",
            logs=tmp_path / "logs",
            vault=tmp_path / "vault",
            db=tmp_path / "tasks.sqlite3",
        ),
        worker=WorkerConfig(**worker_kwargs),
    )
    config.ensure_directories()
    return config


def _task(task_id: str, platform: str) -> Task:
    return Task(
        id=task_id,
        url=f"https://example.test/{task_id}",
        asr_model="qwen3-asr-1.7b",
        status=TaskStatus.queued,
        platform=platform,
        video_id=task_id,
    )


class HoldingPipeline:
    def __init__(self, started: list[str], gate: threading.Event, lock: threading.Lock) -> None:
        self.started = started
        self.gate = gate
        self.lock = lock

    def __call__(self, *args, **kwargs):
        return self

    def run(self, task: Task) -> Task:
        with self.lock:
            self.started.append(task.id)
        self.gate.wait(timeout=3)
        return task


def test_platform_key_normalizes_known_sites():
    assert platform_key("YouTube") == "youtube"
    assert platform_key("Bilibili") == "bilibili"
    assert platform_key("unknown") == "other"


def test_domain_slots_are_independent():
    slots = DomainSlots(WorkerConfig(youtube=1, bilibili=1, other=1))
    assert slots.try_acquire("YouTube")
    assert slots.try_acquire("Bilibili")
    assert not slots.try_acquire("YouTube")
    slots.release("YouTube")
    assert slots.try_acquire("YouTube")


def test_worker_runs_youtube_and_bilibili_in_parallel(tmp_path: Path, monkeypatch):
    started: list[str] = []
    gate = threading.Event()
    lock = threading.Lock()
    monkeypatch.setattr("media_pipeline.worker.Pipeline", HoldingPipeline(started, gate, lock))
    config = _config(tmp_path, youtube=2, bilibili=2)
    store = TaskStore(config.paths.db)
    worker = TaskWorker(config, store)
    worker.start()
    try:
        for index, platform in enumerate(["YouTube", "YouTube", "Bilibili"]):
            task = _task(f"t{index}", platform)
            store.insert(task)
            worker.submit(task)
        assert wait_until(lambda: len(started) == 3)
        ids = set(started)
        assert ids == {"t0", "t1", "t2"}
        snap = worker.snapshot()
        assert snap["youtube"]["running"] == 2
        assert snap["bilibili"]["running"] == 1
    finally:
        gate.set()
        worker.stop()


def test_youtube_cap_holds_the_next_job(tmp_path: Path, monkeypatch):
    started: list[str] = []
    gate = threading.Event()
    lock = threading.Lock()
    monkeypatch.setattr("media_pipeline.worker.Pipeline", HoldingPipeline(started, gate, lock))
    config = _config(tmp_path, youtube=2)
    store = TaskStore(config.paths.db)
    worker = TaskWorker(config, store)
    worker.start()
    try:
        for index in range(3):
            task = _task(f"yt{index}", "YouTube")
            store.insert(task)
            worker.submit(task)
        assert wait_until(lambda: len(started) == 2)
        time.sleep(0.4)
        assert len(started) == 2
        assert worker.snapshot()["youtube"] == {"limit": 2, "running": 2}
        gate.set()
        assert wait_until(lambda: len(started) == 3)
    finally:
        gate.set()
        worker.stop()


def test_raising_youtube_limit_starts_waiting_job(tmp_path: Path, monkeypatch):
    started: list[str] = []
    gate = threading.Event()
    lock = threading.Lock()
    monkeypatch.setattr("media_pipeline.worker.Pipeline", HoldingPipeline(started, gate, lock))
    config = _config(tmp_path, youtube=1, bilibili=1)
    store = TaskStore(config.paths.db)
    worker = TaskWorker(config, store)
    worker.start()
    try:
        for index in range(2):
            task = _task(f"yt{index}", "YouTube")
            store.insert(task)
            worker.submit(task)
        assert wait_until(lambda: len(started) == 1)
        time.sleep(0.3)
        assert len(started) == 1
        worker.set_limits(youtube=2)
        assert wait_until(lambda: len(started) == 2)
        assert worker.snapshot()["youtube"]["limit"] == 2
    finally:
        gate.set()
        worker.stop()


def test_retry_visual_stage_enables_keyframes(tmp_path: Path):
    config = _config(tmp_path)
    store = TaskStore(config.paths.db)
    worker = TaskWorker(config, store)
    task = store.insert(_task("t-visual", "YouTube"))
    assert not task.extra.get("extract_keyframes")
    worker.retry(task, stage="detecting_scenes")
    stored = store.get("t-visual")
    assert stored is not None
    assert stored.extra["extract_keyframes"] is True
    assert stored.extra["rerun_stage"] == "detecting_scenes"
    assert stored.status == TaskStatus.queued


def test_retry_asr_stage_keeps_keyframes_off(tmp_path: Path):
    config = _config(tmp_path)
    store = TaskStore(config.paths.db)
    worker = TaskWorker(config, store)
    task = store.insert(_task("t-asr", "YouTube"))
    worker.retry(task, stage="transcribing")
    stored = store.get("t-asr")
    assert stored is not None
    assert not stored.extra.get("extract_keyframes")
    assert stored.extra["rerun_stage"] == "transcribing"


def test_worker_unloads_idle_vision(tmp_path: Path):
    class FakeVision:
        name = "fake"
        loaded = True
        last_used = 0.0

        def unload(self) -> None:
            self.loaded = False

    config = _config(tmp_path)
    config.analysis.idle_unload_sec = 1
    worker = TaskWorker(config, TaskStore(config.paths.db))
    provider = FakeVision()
    provider.last_used = time.monotonic() - 30
    worker._vision = provider
    worker._maybe_unload_vision()
    assert provider.loaded is False
