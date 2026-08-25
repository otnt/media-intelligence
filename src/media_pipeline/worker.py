from __future__ import annotations

import logging
import threading
import time
from collections import deque

from media_pipeline.artifacts import ArtifactStore
from media_pipeline.config import AppConfig, WorkerConfig
from media_pipeline.diarization import DiarizationProvider
from media_pipeline.models import Task, TaskStatus
from media_pipeline.pipeline import VISUAL_RERUN_STAGES, Pipeline
from media_pipeline.stage_timing import clear_invalidated_timings
from media_pipeline.store import TaskStore
from media_pipeline.visual.vlm import VisionProvider, build_vision_provider

logger = logging.getLogger(__name__)

DOMAIN_KEYS = ("youtube", "bilibili", "other")


def platform_key(platform: str | None) -> str:
    name = (platform or "").strip().lower()
    if name == "youtube":
        return "youtube"
    if name == "bilibili":
        return "bilibili"
    return "other"


class ResizableSlots:
    """Counting limiter whose max can change while jobs are running."""

    def __init__(self, maximum: int) -> None:
        self._cond = threading.Condition()
        self._max = max(0, int(maximum))
        self._used = 0

    def set_max(self, maximum: int) -> None:
        with self._cond:
            self._max = max(0, int(maximum))
            self._cond.notify_all()

    def try_acquire(self) -> bool:
        with self._cond:
            if self._max <= 0 or self._used >= self._max:
                return False
            self._used += 1
            return True

    def acquire(self) -> None:
        with self._cond:
            while self._max <= 0 or self._used >= self._max:
                self._cond.wait()
            self._used += 1

    def release(self) -> None:
        with self._cond:
            self._used = max(0, self._used - 1)
            self._cond.notify_all()

    def snapshot(self) -> dict[str, int]:
        with self._cond:
            return {"limit": self._max, "running": self._used}

    def __enter__(self) -> ResizableSlots:
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


class DomainSlots:
    def __init__(self, worker: WorkerConfig) -> None:
        self._slots = {
            "youtube": ResizableSlots(worker.youtube),
            "bilibili": ResizableSlots(worker.bilibili),
            "other": ResizableSlots(worker.other),
        }

    def try_acquire(self, platform: str | None) -> bool:
        return self._slots[platform_key(platform)].try_acquire()

    def release(self, platform: str | None) -> None:
        self._slots[platform_key(platform)].release()

    def set_limits(self, worker: WorkerConfig) -> None:
        self._slots["youtube"].set_max(worker.youtube)
        self._slots["bilibili"].set_max(worker.bilibili)
        self._slots["other"].set_max(worker.other)

    def snapshot(self) -> dict[str, dict[str, int]]:
        return {key: self._slots[key].snapshot() for key in DOMAIN_KEYS}


class TaskWorker:
    def __init__(self, config: AppConfig, store: TaskStore) -> None:
        self.config = config
        self.store = store
        self._pending: deque[str] = deque()
        self._inflight: set[str] = set()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._dispatcher: threading.Thread | None = None
        self._jobs: list[threading.Thread] = []
        self._slots = DomainSlots(config.worker)
        self._model_slots = ResizableSlots(config.worker.model_jobs)
        self._diarization: DiarizationProvider | None = None
        self._diarization_lock = threading.Lock()
        self._vision: VisionProvider | None = None
        self._summarizing: set[str] = set()

    def start(self) -> None:
        if self._dispatcher and self._dispatcher.is_alive():
            return
        self._stop.clear()
        self._recover()
        self._dispatcher = threading.Thread(
            target=self._loop,
            name="media-pipeline-dispatcher",
            daemon=True,
        )
        self._dispatcher.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._dispatcher:
            self._dispatcher.join(timeout=2)
        deadline = time.monotonic() + 30
        for thread in list(self._jobs):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(timeout=max(0.0, remaining))
        self._unload_vision()

    def submit(self, task: Task) -> None:
        with self._lock:
            if task.id in self._inflight or task.id in self._pending:
                return
            self._pending.append(task.id)
        self._wake.set()

    def retry(self, task: Task, stage: str | None = None, visual: dict | None = None) -> Task:
        if visual:
            merged = dict(task.extra.get("visual") or {})
            merged.update(visual)
            task.extra["visual"] = merged
        if stage:
            task.extra["rerun_stage"] = stage
            if stage in VISUAL_RERUN_STAGES:
                task.extra["extract_keyframes"] = True
            clear_invalidated_timings(task.extra, stage)
            if task.video_id:
                ArtifactStore(self.config.paths.artifacts, task.video_id).clear_invalidated_timings(stage)
        if task.status not in {TaskStatus.failed, TaskStatus.completed}:
            self.store.update(task)
            self.submit(task)
            return task
        task.status = TaskStatus.queued
        task.error = ""
        task.error_stage = ""
        self.store.update(task)
        self.submit(task)
        return task

    def summarize(self, task: Task, prompt: str = "") -> dict:
        from datetime import datetime, timezone

        from media_pipeline.summary import idle_summary, normalize_prompt

        if not task.video_id:
            raise ValueError("Task has no video id yet")
        artifacts = ArtifactStore(self.config.paths.artifacts, task.video_id)
        existing = artifacts.load_summary() or idle_summary(prompt)
        with self._lock:
            if task.id in self._summarizing:
                return existing
            self._summarizing.add(task.id)
        payload = {
            "status": "running",
            "prompt": normalize_prompt(prompt or existing.get("prompt")),
            "markdown": str(existing.get("markdown") or ""),
            "model": str(existing.get("model") or ""),
            "error": "",
            "image_count": int(existing.get("image_count") or 0),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        artifacts.save_summary(payload)
        thread = threading.Thread(
            target=self._run_summary,
            args=(task.id, payload["prompt"]),
            name=f"media-pipeline-summary-{task.id[:8]}",
            daemon=True,
        )
        thread.start()
        self._wake.set()
        return payload

    def set_limits(
        self,
        youtube: int | None = None,
        bilibili: int | None = None,
        other: int | None = None,
        model_jobs: int | None = None,
    ) -> None:
        self.config.worker.set_limits(
            youtube=youtube,
            bilibili=bilibili,
            other=other,
            model_jobs=model_jobs,
        )
        self._slots.set_limits(self.config.worker)
        self._model_slots.set_max(self.config.worker.model_jobs)
        self._wake.set()

    def snapshot(self) -> dict[str, dict[str, int]]:
        payload = self._slots.snapshot()
        payload["model_jobs"] = self._model_slots.snapshot()
        return payload

    def _recover(self) -> None:
        for task in self.store.list_incomplete():
            if task.status != TaskStatus.queued:
                logger.info("Re-queueing interrupted task %s from %s", task.id, task.status.value)
                task.status = TaskStatus.queued
                self.store.update(task)
            self.submit(task)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._dispatch()
            self._maybe_unload_vision()
            self._wake.wait(timeout=0.2)
            self._wake.clear()

    def _dispatch(self) -> None:
        if self._stop.is_set():
            return
        self._reap()
        with self._lock:
            pending = list(self._pending)
            self._pending.clear()
            leftover: list[str] = []
            for task_id in pending:
                if task_id in self._inflight:
                    continue
                task = self.store.get(task_id)
                if task is None:
                    continue
                if not self._slots.try_acquire(task.platform):
                    leftover.append(task_id)
                    continue
                self._inflight.add(task_id)
                thread = threading.Thread(
                    target=self._run_acquired,
                    args=(task_id, task.platform),
                    name=f"media-pipeline-job-{task_id[:8]}",
                    daemon=True,
                )
                self._jobs.append(thread)
                thread.start()
            self._pending.extend(leftover)

    def _run_acquired(self, task_id: str, platform: str) -> None:
        try:
            task = self.store.get(task_id)
            if task is None:
                return
            with self._diarization_lock:
                pipeline = Pipeline(
                    self.config,
                    self.store,
                    diarization=self._diarization,
                    vision=self._vision,
                    model_lock=self._model_slots,
                )
                provider = getattr(pipeline, "diarization", None)
                if provider is not None:
                    self._diarization = provider
                vision = getattr(pipeline, "vision", None)
                if vision is not None:
                    self._vision = vision
            pipeline.run(task)
        except Exception:
            logger.exception("Worker crashed while running task %s", task_id)
        finally:
            self._slots.release(platform)
            with self._lock:
                self._inflight.discard(task_id)
            self._wake.set()

    def _reap(self) -> None:
        self._jobs = [thread for thread in self._jobs if thread.is_alive()]

    def _maybe_unload_vision(self) -> None:
        idle = float(self.config.analysis.idle_unload_sec)
        if idle <= 0:
            return
        with self._lock:
            busy = bool(self._inflight) or bool(self._pending) or bool(self._summarizing)
        if busy:
            return
        self._unload_vision(idle_only=True, idle_sec=idle)

    def _unload_vision(self, *, idle_only: bool = False, idle_sec: float = 0.0) -> None:
        with self._diarization_lock:
            provider = self._vision
            if provider is None:
                return
            if idle_only:
                if not provider.loaded:
                    return
                last = float(getattr(provider, "last_used", 0.0) or 0.0)
                if last <= 0 or time.monotonic() - last < idle_sec:
                    return
                provider.unload()
                return
            if provider.loaded:
                provider.unload()
            closer = getattr(provider, "close", None)
            if callable(closer):
                closer()

    def _run_summary(self, task_id: str, prompt: str) -> None:
        from pathlib import Path

        from media_pipeline.notes import NoteWriter
        from media_pipeline.summary import summarize_task

        task = self.store.get(task_id)
        if task is None or not task.video_id:
            with self._lock:
                self._summarizing.discard(task_id)
            return
        artifacts = ArtifactStore(self.config.paths.artifacts, task.video_id)
        try:
            provider = self._ensure_vision()
            metadata = artifacts.load_metadata()
            with self._model_slots:
                result = summarize_task(
                    artifacts,
                    provider,
                    prompt=prompt,
                    metadata=metadata,
                )
            artifacts.save_summary(result)
            if task.note_path:
                try:
                    NoteWriter(Path(task.note_path).parent).update_summary(Path(task.note_path), result["markdown"])
                except Exception:
                    logger.exception("Could not write summary into note for task %s", task_id)
        except Exception as exc:
            logger.exception("Summary failed for task %s", task_id)
            current = artifacts.load_summary() or {}
            artifacts.save_summary(
                {
                    **current,
                    "status": "failed",
                    "prompt": prompt,
                    "error": str(exc),
                }
            )
        finally:
            with self._lock:
                self._summarizing.discard(task_id)
            self._wake.set()

    def _ensure_vision(self) -> VisionProvider:
        with self._diarization_lock:
            if self._vision is None:
                self._vision = build_vision_provider(self.config)
            return self._vision
