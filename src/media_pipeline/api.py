from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from media_pipeline.asr.registry import list_models
from media_pipeline.config import AppConfig
from media_pipeline.media import UnsupportedURLError, canonicalize_url, parse_video_ref
from media_pipeline.models import ASR_MODELS, Task, TaskStatus, asr_label
from media_pipeline.store import TaskStore
from media_pipeline.worker import TaskWorker


class CreateTaskRequest(BaseModel):
    url: str
    asr_model: str = Field(min_length=1)
    language: str | None = None


class RetryTaskRequest(BaseModel):
    asr_model: str | None = None


def create_app(config: AppConfig, store: TaskStore | None = None, worker: TaskWorker | None = None) -> FastAPI:
    store = store or TaskStore(config.paths.db)
    worker = worker or TaskWorker(config, store)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        config.ensure_directories()
        worker.start()
        yield
        worker.stop()

    app = FastAPI(title="Media Pipeline", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.config = config
    app.state.store = store
    app.state.worker = worker

    @app.get("/")
    def root() -> dict[str, Any]:
        return {"service": "media-pipeline", "health": "/v1/health", "models": "/v1/models"}

    @app.get("/v1/health")
    def health() -> dict[str, Any]:
        notes_dir = config.notes_dir()
        return {
            "ok": True,
            "vault": str(config.paths.vault) if config.paths.vault else None,
            "notes_dir": str(notes_dir) if notes_dir else None,
            "default_asr_model": config.asr.default,
            "default_language": config.asr.language,
        }

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "default": config.asr.default,
            "models": [
                {
                    "id": item.id,
                    "label": item.label,
                    "runtime": item.runtime,
                    "available": item.available,
                    "detail": item.detail,
                    "code_switching": item.code_switching,
                }
                for item in list_models()
            ],
        }

    @app.post("/v1/tasks", status_code=202)
    def create_task(payload: CreateTaskRequest) -> dict[str, Any]:
        _validate_model(payload.asr_model)
        page_url = canonicalize_url(payload.url.strip())
        try:
            platform, video_id = parse_video_ref(page_url)
        except UnsupportedURLError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if config.notes_dir() is None:
            raise HTTPException(
                status_code=500,
                detail="Obsidian vault is not configured. Set paths.vault in ~/.config/media-pipeline/config.yaml",
            )
        task = Task(
            id=str(uuid.uuid4()),
            url=page_url,
            asr_model=payload.asr_model,
            status=TaskStatus.queued,
            platform=platform,
            video_id=video_id,
            extra={"language": payload.language or config.asr.language or "auto"},
        )
        store.insert(task)
        worker.submit(task)
        return {
            "id": task.id,
            "status": task.status.value,
            "asr_model": task.asr_model,
            "asr_label": asr_label(task.asr_model),
            "language": task.extra["language"],
            "message": "Added to queue",
        }

    @app.get("/v1/tasks")
    def list_tasks() -> dict[str, Any]:
        tasks = [item.to_public() for item in store.list_recent()]
        return {"tasks": tasks}

    @app.get("/v1/tasks/{task_id}")
    def get_task(task_id: str) -> dict[str, Any]:
        task = store.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return task.to_public()

    @app.post("/v1/tasks/{task_id}/retry")
    def retry_task(task_id: str, payload: RetryTaskRequest | None = None) -> dict[str, Any]:
        task = store.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        if payload and payload.asr_model:
            _validate_model(payload.asr_model)
            task.asr_model = payload.asr_model
        worker.retry(task)
        return task.to_public()

    return app


def _validate_model(model_id: str) -> None:
    if model_id not in ASR_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown ASR model {model_id!r}. Supported: {', '.join(ASR_MODELS)}",
        )
