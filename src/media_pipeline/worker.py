from __future__ import annotations

import logging
import queue
import threading

from media_pipeline.config import AppConfig
from media_pipeline.models import Task, TaskStatus
from media_pipeline.pipeline import Pipeline
from media_pipeline.store import TaskStore

logger = logging.getLogger(__name__)


class TaskWorker:
    def __init__(self, config: AppConfig, store: TaskStore) -> None:
        self.config = config
        self.store = store
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pipeline = Pipeline(config, store)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._recover()
        self._thread = threading.Thread(target=self._loop, name="media-pipeline-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._queue.put("")
        if self._thread:
            self._thread.join(timeout=5)

    def submit(self, task: Task) -> None:
        self._queue.put(task.id)

    def retry(self, task: Task) -> Task:
        if task.status not in {TaskStatus.failed, TaskStatus.completed}:
            self.submit(task)
            return task
        task.status = TaskStatus.queued
        task.error = ""
        task.error_stage = ""
        self.store.update(task)
        self.submit(task)
        return task

    def _recover(self) -> None:
        for task in self.store.list_incomplete():
            if task.status == TaskStatus.queued:
                self._queue.put(task.id)
                continue
            logger.info("Re-queueing interrupted task %s from %s", task.id, task.status.value)
            task.status = TaskStatus.queued
            self.store.update(task)
            self._queue.put(task.id)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                task_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if not task_id:
                continue
            task = self.store.get(task_id)
            if task is None:
                continue
            try:
                self._pipeline.run(task)
            except Exception:
                logger.exception("Worker crashed while running task %s", task_id)
            finally:
                self._queue.task_done()
