from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from media_pipeline.artifacts import ArtifactStore
from media_pipeline.asr.registry import list_models
from media_pipeline.config import AppConfig, persist_dashboard_config, persist_worker_config
from media_pipeline.dashboard_view import (
    dashboard_options,
    flatten_groups,
    normalize_dashboard_filter,
    normalize_dashboard_group,
    normalize_dashboard_order,
    organize_task_payloads,
)
from media_pipeline.media import UnsupportedURLError, extract_supported_url, parse_video_ref
from media_pipeline.models import ASR_MODELS, Task, TaskStatus, asr_label, source_key, source_label
from media_pipeline.notes import load_note
from media_pipeline.pipeline import RERUN_STAGES
from media_pipeline.stage_timing import merge_stage_timings
from media_pipeline.store import TaskStore
from media_pipeline.summary import (
    DEFAULT_PROMPT,
    coerce_summary_runs,
    idle_summary,
    normalize_prompt,
    render_summary_runs,
    strip_summary_media,
)
from media_pipeline.worker import TaskWorker

DASHBOARD_FILE = Path(__file__).resolve().parent / "dashboard" / "static" / "index.html"
ALLOWED_FRAME_DIRS = {"candidate_frames", "keyframes"}


class CreateTaskRequest(BaseModel):
    url: str
    asr_model: str | None = None
    language: str | None = None
    extract_keyframes: bool = False


class RetryTaskRequest(BaseModel):
    asr_model: str | None = None
    stage: str | None = None
    visual: dict[str, Any] | None = None


class FrameDecisionRequest(BaseModel):
    decision: str


class SummarizeRequest(BaseModel):
    prompt: str | None = None
    model: str | None = None


class WorkerLimitsRequest(BaseModel):
    youtube: int | None = Field(default=None, ge=0, le=64)
    bilibili: int | None = Field(default=None, ge=0, le=64)
    other: int | None = Field(default=None, ge=0, le=64)
    model_jobs: int | None = Field(default=None, ge=1, le=64)


class DashboardViewRequest(BaseModel):
    filter: str | None = None
    group: str | None = None
    order: str | None = None


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
            "organize": config.dashboard.as_dict(),
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

    @app.get("/v1/dashboard")
    def dashboard_view() -> dict[str, Any]:
        return _dashboard_snapshot(config)

    @app.put("/v1/dashboard")
    def update_dashboard(payload: DashboardViewRequest) -> dict[str, Any]:
        updates = payload.model_dump(exclude_none=True)
        if not updates:
            raise HTTPException(status_code=400, detail="No dashboard view provided")
        config.dashboard.set_view(**updates)
        persist_dashboard_config(config)
        return {"ok": True, **_dashboard_snapshot(config)}

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

    @app.get("/v1/inbox", status_code=202)
    def inbox(
        request: Request,
        url: str,
        asr_model: str | None = None,
        language: str | None = None,
        extract_keyframes: bool = False,
        token: str | None = None,
        x_ingest_token: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Queue a share URL. GET+query is what iOS Shortcuts can send without breaking HTTP."""
        _authorize_ingest(config, request, x_ingest_token or token, authorization)
        return _enqueue_share(
            config,
            store,
            worker,
            url,
            asr_model=asr_model,
            language=language,
            extract_keyframes=extract_keyframes,
        )

    @app.post("/v1/tasks", status_code=202)
    async def create_task(
        request: Request,
        x_ingest_token: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _authorize_ingest(config, request, x_ingest_token, authorization)
        payload = await _read_create_task(request)
        return _enqueue_share(
            config,
            store,
            worker,
            payload.url,
            asr_model=payload.asr_model,
            language=payload.language,
            extract_keyframes=payload.extract_keyframes,
        )

    @app.get("/v1/tasks")
    def list_tasks(
        filter: str | None = None,
        group: str | None = None,
        order: str | None = None,
    ) -> dict[str, Any]:
        organize = _resolve_organize(config, filter=filter, group=group, order=order)
        payloads = [_enrich_task(item, config) for item in store.list_all()]
        groups = organize_task_payloads(
            payloads,
            time_filter=organize["filter"],
            group=organize["group"],
            order=organize["order"],
        )
        return {
            "tasks": flatten_groups(groups),
            "groups": groups,
            "organize": organize,
        }

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
            summary = artifacts.debug_summary()
            payload["debug"] = summary
            payload["decisions"] = summary.get("decisions") or []
            payload["overrides"] = artifacts.load_overrides()
            payload["scenes"] = [item.to_dict() for item in (artifacts.load_scenes() or [])]
            if not payload.get("selected_count"):
                payload["selected_count"] = int(summary.get("selected_count") or 0)
            payload["summary"] = _public_summary(task, artifacts, worker)
            payload["extracted_markdown"] = _extracted_markdown(task, config, artifacts)
        else:
            payload["summary"] = idle_summary()
            payload["extracted_markdown"] = ""
        return payload

    @app.get("/v1/tasks/{task_id}/summary")
    def get_summary(task_id: str) -> dict[str, Any]:
        task = store.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        if not task.video_id:
            return idle_summary()
        artifacts = ArtifactStore(config.paths.artifacts, task.video_id)
        return _public_summary(task, artifacts, worker)

    @app.get("/v1/summary/models")
    def summary_models() -> dict[str, Any]:
        from media_pipeline.summary_llm import catalog_summary_backends

        vision = getattr(worker, "_vision", None)
        return {
            "providers": catalog_summary_backends(config, vision),
            "default": list(config.summary.providers),
        }

    @app.post("/v1/tasks/{task_id}/summary", status_code=202)
    def start_summary(task_id: str, payload: SummarizeRequest | None = None) -> dict[str, Any]:
        task = store.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        if not task.video_id:
            raise HTTPException(status_code=400, detail="Task has no video id yet")
        prompt = normalize_prompt(payload.prompt if payload else None)
        model = str(payload.model if payload and payload.model else "").strip()
        starter = getattr(worker, "summarize", None)
        if not callable(starter):
            raise HTTPException(status_code=501, detail="Worker cannot summarize")
        try:
            return starter(task, prompt, model)
        except TypeError:
            return starter(task, prompt)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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

    @app.get("/v1/videos/{video_id}/attachments/{filename}")
    def get_attachment(video_id: str, filename: str):
        if Path(filename).name != filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        path = _attachment_path(config, video_id, filename)
        if path is None:
            raise HTTPException(status_code=404, detail="Attachment not found")
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


def _resolve_organize(
    config: AppConfig,
    *,
    filter: str | None = None,
    group: str | None = None,
    order: str | None = None,
) -> dict[str, str]:
    return {
        "filter": normalize_dashboard_filter(filter or config.dashboard.filter),
        "group": normalize_dashboard_group(group or config.dashboard.group),
        "order": normalize_dashboard_order(order or config.dashboard.order),
    }


def _dashboard_snapshot(config: AppConfig) -> dict[str, Any]:
    return {**config.dashboard.as_dict(), **dashboard_options()}


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
    if "extract_keyframes" not in (task.extra or {}) and artifacts.keyframes_path.exists():
        payload["extract_keyframes"] = True
    if payload.get("extract_keyframes"):
        if not payload.get("candidate_count"):
            payload["candidate_count"] = len(artifacts.load_candidates() or [])
        if not payload.get("keyframe_count"):
            payload["keyframe_count"] = len(artifacts.load_keyframes() or [])
        if not payload.get("selected_count"):
            analysis = artifacts.load_frame_analysis()
            if analysis is not None:
                payload["selected_count"] = sum(1 for item in analysis if item.kept)
    if not payload.get("segment_count"):
        payload["segment_count"] = len(artifacts.load_named() or [])
    payload["has_multimodal"] = artifacts.multimodal_path.exists()
    payload["stage_timings"] = merge_stage_timings(
        artifacts.load_stage_timings(),
        payload.get("stage_timings"),
    )
    stored = artifacts.load_summary() or {}
    payload["summary_status"] = str(stored.get("status") or "idle")
    return payload


def _public_summary(task: Task, artifacts: ArtifactStore, worker: object) -> dict[str, Any]:
    payload = artifacts.load_summary() or idle_summary()
    running = getattr(worker, "_summarizing", set())
    if payload.get("status") == "running" and task.id not in running:
        payload = {
            **payload,
            "status": "failed",
            "error": payload.get("error") or "Summary was interrupted",
        }
    payload.setdefault("prompt", DEFAULT_PROMPT)
    payload["prompt"] = normalize_prompt(str(payload.get("prompt") or ""))
    payload.setdefault("error", "")
    payload.setdefault("model", "")
    runs = coerce_summary_runs(payload)
    payload["runs"] = runs
    payload["markdown"] = render_summary_runs(runs) if runs else strip_summary_media(str(payload.get("markdown") or ""))
    payload["image_count"] = 0
    payload.setdefault("updated_at", "")
    return payload


def _extracted_markdown(task: Task, config: AppConfig, artifacts: ArtifactStore) -> str:
    if task.note_path:
        path = Path(task.note_path)
        if path.exists():
            document = load_note(path, rewrite_layout=True)
            return document.transcript_markdown.strip()
    named = artifacts.load_named() or []
    if not named:
        return ""
    from media_pipeline.transcript import render_transcript
    from media_pipeline.visual.filtering import caption_for, selected_from_verdicts

    keyframes = artifacts.load_keyframes() or []
    analysis = artifacts.load_frame_analysis() or []
    chosen = selected_from_verdicts(keyframes, analysis) if analysis else keyframes
    frames = [(frame.timestamp, frame.image_path, caption_for(frame, analysis)) for frame in chosen]
    return render_transcript(named, video_id=task.video_id, frames=frames).strip()


def _attachment_path(config: AppConfig, video_id: str, filename: str) -> Path | None:
    name = Path(filename).name
    if not name or name != filename:
        return None
    artifacts = ArtifactStore(config.paths.artifacts, video_id)
    directories = [
        artifacts.keyframe_dir,
        artifacts.candidate_dir,
        config.paths.videos / video_id,
    ]
    notes_dir = config.notes_dir()
    if notes_dir is not None:
        directories.append(notes_dir / "attachments" / video_id)
    if config.paths.vault is not None:
        directories.append(config.paths.vault / "attachments" / video_id)
    for directory in directories:
        path = directory / name
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def _optional_form_str(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


async def _read_create_task(request: Request) -> CreateTaskRequest:
    content_type = (request.headers.get("content-type") or "").lower()
    if "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
        form = await request.form()
        return CreateTaskRequest(
            url=str(form.get("url") or ""),
            asr_model=_optional_form_str(form.get("asr_model")),
            language=_optional_form_str(form.get("language")),
            extract_keyframes=str(form.get("extract_keyframes") or "").strip().lower()
            in {"1", "true", "yes", "on"},
        )
    try:
        data = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Expected JSON or form fields") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="JSON object required")
    return CreateTaskRequest.model_validate(data)


def _enqueue_share(
    config: AppConfig,
    store: TaskStore,
    worker: TaskWorker,
    raw_url: str,
    *,
    asr_model: str | None,
    language: str | None,
    extract_keyframes: bool,
) -> dict[str, Any]:
    model_id = (asr_model or "").strip() or config.asr.default
    _validate_model(model_id)
    try:
        page_url = extract_supported_url(raw_url)
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
        asr_model=model_id,
        status=TaskStatus.queued,
        platform=platform,
        video_id=video_id,
        extra={
            "language": language or config.asr.language or "auto",
            "extract_keyframes": bool(extract_keyframes),
        },
    )
    store.insert(task)
    worker.submit(task)
    return {
        "id": task.id,
        "status": task.status.value,
        "asr_model": task.asr_model,
        "asr_label": asr_label(task.asr_model),
        "language": task.extra["language"],
        "extract_keyframes": bool(task.extra.get("extract_keyframes")),
        "url": page_url,
        "platform": platform,
        "source": source_key(platform, page_url),
        "source_label": source_label(platform, page_url),
        "created_at": task.created_at,
        "requested_at": task.created_at,
        "message": "Added to queue",
    }


def _authorize_ingest(
    config: AppConfig,
    request: Request,
    x_ingest_token: str | None,
    authorization: str | None,
) -> None:
    expected = (config.server.ingest_token or "").strip()
    if not expected:
        return
    host = (request.client.host if request.client else "") or ""
    if host in {"127.0.0.1", "::1", "localhost"}:
        return
    provided = (x_ingest_token or "").strip()
    if authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()
    if provided != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid ingest token")


def _validate_model(model_id: str) -> None:
    if model_id not in ASR_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown ASR model {model_id!r}. Supported: {', '.join(ASR_MODELS)}",
        )
