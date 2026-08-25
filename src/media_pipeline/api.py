from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from media_pipeline.artifacts import ArtifactStore
from media_pipeline.asr.registry import list_models
from media_pipeline.config import AppConfig, persist_worker_config
from media_pipeline.media import UnsupportedURLError, canonicalize_url, parse_video_ref
from media_pipeline.models import ASR_MODELS, Task, TaskStatus, asr_label
from media_pipeline.pipeline import RERUN_STAGES
from media_pipeline.store import TaskStore
from media_pipeline.worker import TaskWorker

DASHBOARD_FILE = Path(__file__).resolve().parent / "dashboard" / "static" / "index.html"
ALLOWED_FRAME_DIRS = {"candidate_frames", "keyframes"}


class CreateTaskRequest(BaseModel):
    url: str
    asr_model: str = Field(min_length=1)
    language: str | None = None


class RetryTaskRequest(BaseModel):
    asr_model: str | None = None
    stage: str | None = None
    visual: dict[str, Any] | None = None


class FrameDecisionRequest(BaseModel):
    decision: str


class WorkerLimitsRequest(BaseModel):
    youtube: int | None = Field(default=None, ge=0, le=64)
    bilibili: int | None = Field(default=None, ge=0, le=64)
    other: int | None = Field(default=None, ge=0, le=64)
    model_jobs: int | None = Field(default=None, ge=1, le=64)


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
    def root():
        if DASHBOARD_FILE.exists():
            return FileResponse(DASHBOARD_FILE, media_type="text/html")
        return {"service": "media-pipeline", "health": "/v1/health", "dashboard": "/"}

    @app.get("/v1/health")
    def health() -> dict[str, Any]:
        notes_dir = config.notes_dir()
        return {
            "ok": True,
            "vault": str(config.paths.vault) if config.paths.vault else None,
            "notes_dir": str(notes_dir) if notes_dir else None,
            "default_asr_model": config.asr.default,
            "default_language": config.asr.language,
            "dashboard": f"http://{config.server.host}:{config.server.port}/",
            "visual": config.visual.as_dict(),
            "worker": _worker_snapshot(worker, config),
        }

    @app.get("/v1/worker")
    def worker_status() -> dict[str, Any]:
        return _worker_snapshot(worker, config)

    @app.put("/v1/worker")
    def update_worker(payload: WorkerLimitsRequest) -> dict[str, Any]:
        updates = payload.model_dump(exclude_none=True)
        if not updates:
            raise HTTPException(status_code=400, detail="No worker limits provided")
        config.worker.set_limits(**updates)
        setter = getattr(worker, "set_limits", None)
        if callable(setter):
            setter(**updates)
        persist_worker_config(config)
        return {"ok": True, "worker": _worker_snapshot(worker, config)}

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

    @app.get("/v1/visual/config")
    def visual_config() -> dict[str, Any]:
        return {"visual": config.visual.as_dict(), "stages": sorted(RERUN_STAGES)}

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
        tasks = [_enrich_task(item, config) for item in store.list_recent()]
        return {"tasks": tasks}

    @app.get("/v1/tasks/{task_id}")
    def get_task(task_id: str) -> dict[str, Any]:
        task = store.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return _enrich_task(task, config)

    @app.get("/v1/tasks/{task_id}/debug")
    def debug_task(task_id: str) -> dict[str, Any]:
        task = store.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        payload = _enrich_task(task, config)
        if task.video_id:
            artifacts = ArtifactStore(config.paths.artifacts, task.video_id)
            payload["debug"] = artifacts.debug_summary()
            payload["decisions"] = artifacts.load_candidate_decisions()
            payload["overrides"] = artifacts.load_overrides()
            payload["scenes"] = [item.to_dict() for item in (artifacts.load_scenes() or [])]
        return payload

    @app.post("/v1/tasks/{task_id}/retry")
    def retry_task(task_id: str, payload: RetryTaskRequest | None = None) -> dict[str, Any]:
        task = store.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        stage = payload.stage if payload else None
        if stage and stage not in RERUN_STAGES:
            raise HTTPException(status_code=400, detail=f"Unknown stage {stage!r}")
        if payload and payload.asr_model:
            _validate_model(payload.asr_model)
            task.asr_model = payload.asr_model
        worker.retry(task, stage=stage, visual=payload.visual if payload else None)
        return task.to_public()

    @app.get("/v1/videos/{video_id}/frames/{kind}/{filename}")
    def get_frame(video_id: str, kind: str, filename: str):
        if kind not in ALLOWED_FRAME_DIRS:
            raise HTTPException(status_code=404, detail="Unknown frame collection")
        if Path(filename).name != filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        path = config.paths.artifacts / video_id / kind / filename
        if not path.exists():
            raise HTTPException(status_code=404, detail="Frame not found")
        return FileResponse(path)

    @app.post("/v1/videos/{video_id}/frames/{filename}/decision")
    def set_frame_decision(video_id: str, filename: str, payload: FrameDecisionRequest) -> dict[str, Any]:
        decision = payload.decision.strip().lower()
        if decision not in {"keep", "drop"}:
            raise HTTPException(status_code=400, detail="decision must be keep or drop")
        if Path(filename).name != filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        artifacts = ArtifactStore(config.paths.artifacts, video_id)
        overrides = artifacts.set_override(filename, decision)
        return {"ok": True, "filename": filename, "decision": decision, "overrides": overrides}

    return app


def _worker_snapshot(worker: object, config: AppConfig) -> dict[str, Any]:
    snapshot = getattr(worker, "snapshot", None)
    if callable(snapshot):
        payload = snapshot()
        if isinstance(payload, dict) and payload:
            return payload
    limits = config.worker.as_dict()
    return {key: {"limit": value, "running": 0} for key, value in limits.items()}


def _enrich_task(task: Task, config: AppConfig) -> dict[str, Any]:
    payload = task.to_public()
    if not task.video_id:
        return payload
    artifacts = ArtifactStore(config.paths.artifacts, task.video_id)
    if not payload.get("candidate_count"):
        payload["candidate_count"] = len(artifacts.load_candidates() or [])
    if not payload.get("keyframe_count"):
        payload["keyframe_count"] = len(artifacts.load_keyframes() or [])
    if not payload.get("segment_count"):
        payload["segment_count"] = len(artifacts.load_named() or [])
    payload["has_multimodal"] = artifacts.multimodal_path.exists()
    return payload


def _validate_model(model_id: str) -> None:
    if model_id not in ASR_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown ASR model {model_id!r}. Supported: {', '.join(ASR_MODELS)}",
        )
