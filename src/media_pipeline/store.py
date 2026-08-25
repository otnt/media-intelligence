from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from media_pipeline.models import Task, TaskStatus, now_stamp


class TaskStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create()

    def _create(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                asr_model TEXT NOT NULL,
                status TEXT NOT NULL,
                video_id TEXT,
                platform TEXT,
                title TEXT,
                note_path TEXT,
                video_path TEXT,
                audio_path TEXT,
                error_stage TEXT,
                error TEXT,
                created_at TEXT,
                updated_at TEXT,
                extra TEXT
            )
            """
        )
        self._conn.commit()

    def insert(self, task: Task) -> Task:
        if not task.created_at:
            task.created_at = now_stamp()
        task.updated_at = now_stamp()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO tasks (
                    id, url, asr_model, status, video_id, platform, title,
                    note_path, video_path, audio_path, error_stage, error,
                    created_at, updated_at, extra
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _row_values(task),
            )
            self._conn.commit()
        return task

    def update(self, task: Task) -> Task:
        task.updated_at = now_stamp()
        with self._lock:
            self._conn.execute(
                """
                UPDATE tasks SET
                    url=?, asr_model=?, status=?, video_id=?, platform=?, title=?,
                    note_path=?, video_path=?, audio_path=?, error_stage=?, error=?,
                    created_at=?, updated_at=?, extra=?
                WHERE id=?
                """,
                (*_row_values(task)[1:], task.id),
            )
            self._conn.commit()
        return task

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return _task_from_row(row) if row else None

    def list_recent(self, limit: int = 50) -> list[Task]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM tasks ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_task_from_row(row) for row in rows]

    def list_incomplete(self) -> list[Task]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM tasks WHERE status NOT IN (?, ?)",
                (TaskStatus.completed.value, TaskStatus.failed.value),
            ).fetchall()
        return [_task_from_row(row) for row in rows]


def _row_values(task: Task) -> tuple[Any, ...]:
    return (
        task.id,
        task.url,
        task.asr_model,
        task.status.value,
        task.video_id,
        task.platform,
        task.title,
        task.note_path,
        task.video_path,
        task.audio_path,
        task.error_stage,
        task.error,
        task.created_at,
        task.updated_at,
        json.dumps(task.extra, ensure_ascii=False),
    )


def _task_from_row(row: sqlite3.Row) -> Task:
    extra_raw = row["extra"] or "{}"
    try:
        extra = json.loads(extra_raw)
    except json.JSONDecodeError:
        extra = {}
    return Task(
        id=row["id"],
        url=row["url"],
        asr_model=row["asr_model"],
        status=TaskStatus(row["status"]),
        video_id=row["video_id"] or "",
        platform=row["platform"] or "",
        title=row["title"] or "",
        note_path=row["note_path"] or "",
        video_path=row["video_path"] or "",
        audio_path=row["audio_path"] or "",
        error_stage=row["error_stage"] or "",
        error=row["error"] or "",
        created_at=row["created_at"] or "",
        updated_at=row["updated_at"] or "",
        extra=extra if isinstance(extra, dict) else {},
    )
