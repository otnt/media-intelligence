from __future__ import annotations

import ipaddress
import re
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from media_pipeline.artifacts import ArtifactStore
from media_pipeline.asr.registry import list_models
from media_pipeline import bulk as bulk_jobs
from media_pipeline.config import (
    AppConfig,
    persist_dashboard_config,
    persist_summary_prompt,
    persist_visual_config,
    persist_worker_config,
)
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
from media_pipeline.tags import collect_manual_tags, normalize_tag, set_task_tags
from media_pipeline.worker import TaskWorker

logger = logging.getLogger(__name__)
DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard" / "static"
DASHBOARD_FILE = DASHBOARD_DIR / "index.html"
# The previous dashboard: settings, keyframe decisions and prompt editing still live here.
CLASSIC_DASHBOARD_FILE = DASHBOARD_DIR / "classic.html"
# Pages always revalidate (ETag), so a phone or home-screen app never keeps an old UI.
NO_CACHE = {"Cache-Control": "no-cache"}
STATIC_TYPES = {".png": "image/png", ".svg": "image/svg+xml", ".webmanifest": "application/manifest+json"}
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


class VisualSettingsRequest(BaseModel):
    sample_interval_sec: float | None = Field(default=None, ge=0.1, le=600)
    scene_threshold: float | None = Field(default=None, ge=0, le=100)
    similarity_threshold: float | None = Field(default=None, ge=0, le=1)
    min_scene_duration_sec: float | None = Field(default=None, ge=0, le=60)
    context_before_sec: float | None = Field(default=None, ge=0, le=600)
    context_after_sec: float | None = Field(default=None, ge=0, le=600)
    vlm_keep_threshold: float | None = Field(default=None, ge=0, le=1)


class SummaryPromptRequest(BaseModel):
    prompt: str | None = None


class TaskTagsRequest(BaseModel):
    tags: list[str] = Field(default_factory=list)


class ReadRequest(BaseModel):
    read: bool = True


class ReadAllRequest(BaseModel):
    collection: str = ""


def _now_stamp() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M")


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
            return FileResponse(DASHBOARD_FILE, media_type="text/html", headers=NO_CACHE)
        return {"service": "media-pipeline", "health": "/v1/health", "dashboard": "/"}

    @app.get("/classic")
    def classic():
        return FileResponse(CLASSIC_DASHBOARD_FILE, media_type="text/html", headers=NO_CACHE)

    @app.get("/manifest.webmanifest")
    def manifest():
        return FileResponse(DASHBOARD_DIR / "manifest.webmanifest", media_type="application/manifest+json", headers=NO_CACHE)

    @app.get("/static/{filename}")
    def static_file(filename: str):
        path = DASHBOARD_DIR / "icons" / filename
        media_type = STATIC_TYPES.get(Path(filename).suffix)
        if Path(filename).name != filename or media_type is None or not path.is_file():
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(path, media_type=media_type, headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/learn")
    def learn_page():
        return FileResponse(DASHBOARD_DIR / "learn.html", media_type="text/html", headers=NO_CACHE)

    def _learn_dirs() -> dict[str, Path]:
        """{collection: <notes>/<collection>/_学习} for every collection that has study material."""
        from media_pipeline.learning import LEARN_DIR

        notes = config.notes_dir()
        found: dict[str, Path] = {}
        if notes is None:
            return found
        for name in bulk_jobs.list_jobs(config):
            try:
                title = str(bulk_jobs.open_job(config, name).meta("title", "") or name)
            except Exception:  # noqa: BLE001
                continue
            folder = notes / title / LEARN_DIR
            if folder.is_dir():
                found[title] = folder
        return found

    @app.get("/v1/learn")
    def learn_index() -> dict[str, Any]:
        from media_pipeline.learning import PLAN_NAME, TOPICS_DIR

        out = []
        for collection, folder in _learn_dirs().items():
            topics = []
            for path in sorted((folder / TOPICS_DIR).glob("*.md")):
                head = path.read_text(encoding="utf-8")[:400]
                count = re.search(r"(\d+) 场演讲", head)
                topics.append({"name": path.stem, "path": f"{TOPICS_DIR}/{path.name}",
                               "talks": int(count.group(1)) if count else 0})
            topics.sort(key=lambda t: -t["talks"])
            plan = folder / f"{PLAN_NAME}.md"
            out.append({
                "collection": collection,
                "plan": plan.name if plan.is_file() else "",
                "updated": time.strftime("%m-%d %H:%M", time.localtime(plan.stat().st_mtime)) if plan.is_file() else "",
                "topics": topics,
            })
        return {"collections": out}

    @app.get("/v1/learn/doc")
    def learn_doc(collection: str, path: str) -> dict[str, Any]:
        folder = _learn_dirs().get(collection)
        if folder is None:
            raise HTTPException(status_code=404, detail="No study material for this collection")
        target = (folder / path).resolve()
        if folder.resolve() not in target.parents or target.suffix != ".md" or not target.is_file():
            raise HTTPException(status_code=404, detail="No such document")
        links = {}
        for task in store.list_all(limit=100_000):
            if str(task.extra.get("collection") or "") == collection and task.note_path:
                links[Path(task.note_path).stem] = f"/#/n/{task.id}"
        return {"title": target.stem, "markdown": target.read_text(encoding="utf-8"), "links": links}

    @app.get("/bulk")
    def bulk_page():
        return FileResponse(DASHBOARD_DIR / "bulk.html", media_type="text/html", headers=NO_CACHE)

    def _bulk_job(name: str) -> bulk_jobs.BulkStore:
        try:
            return bulk_jobs.open_job(config, name)
        except (ValueError, FileNotFoundError):
            raise HTTPException(status_code=404, detail="No such bulk job")

    @app.get("/v1/bulk")
    def bulk_list() -> dict[str, Any]:
        jobs = []
        for name in bulk_jobs.list_jobs(config):
            try:
                jobs.append(bulk_jobs.open_job(config, name).summary())
            except Exception:  # noqa: BLE001 - one broken job must not hide the others
                logger.exception("Could not read bulk job %s", name)
        return {"jobs": jobs}

    @app.get("/v1/bulk/{name}")
    def bulk_detail(name: str) -> dict[str, Any]:
        job = _bulk_job(name)
        videos = job.rows()
        processed = store.status_by_video([video["id"] for video in videos])
        for video in videos:
            video["task"] = processed.get(video["id"])
        from media_pipeline.watchdog import last_result

        summary = job.summary()
        dog = last_result(config)
        summary["watchdog"] = {key: dog.get(key) for key in ("at", "ok", "problems", "notes", "actions")} if dog else {}
        return {"summary": summary, "videos": videos, "events": job.events()}

    @app.get("/v1/bulk/{name}/transcript/{video_id}")
    def bulk_transcript(name: str, video_id: str) -> Response:
        text = bulk_jobs.transcript_text(config, _bulk_job(name), video_id)
        if text is None:
            raise HTTPException(status_code=404, detail="No transcript yet")
        return Response(text, media_type="text/plain; charset=utf-8")

    @app.get("/v1/bulk/{name}/csv")
    def bulk_csv(name: str) -> Response:
        job = _bulk_job(name)
        return Response(
            "\ufeff" + job.to_csv(),  # BOM so Numbers/Excel read the Chinese titles as UTF-8
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{name}.csv"'},
        )

    @app.post("/v1/bulk/{name}/videos/{video_id}/process", status_code=202)
    def bulk_process(name: str, video_id: str) -> dict[str, Any]:
        """Turn a transcribed bulk video into a normal task (note + summary), reusing its download,
        metadata and transcript; the task belongs to the job's collection."""
        job = _bulk_job(name)
        video = job.video(video_id)
        if video is None:
            raise HTTPException(status_code=404, detail="No such video in this job")
        if video.get("asr_status") != "done":
            raise HTTPException(status_code=409, detail="Not transcribed yet")
        if config.notes_dir() is None:
            raise HTTPException(status_code=500, detail="Obsidian vault is not configured")
        source = str(video.get("transcript_source") or "")
        asr_model = {"youtube": "youtube-captions", "ai-engineer": "ai-engineer"}.get(source) or (
            source or job.meta("asr_model", "") or config.asr.default)
        _validate_model(asr_model)
        task = Task(
            id=str(uuid.uuid4()),
            url=str(video["url"]),
            asr_model=asr_model,
            status=TaskStatus.queued,
            platform="YouTube",
            video_id=video_id,
            title=str(video.get("title") or ""),
            extra={
                "language": config.asr.language or "auto",
                "extract_keyframes": False,
                "summary_model": str(job.meta("summary_model", "") or ""),
                "collection": str(job.meta("title", "") or name),
                "bulk_job": name,
            },
        )
        store.insert(task)
        worker.submit(task)
        return task.to_public()

    @app.get("/v1/collections")
    def collections() -> dict[str, Any]:
        counts: dict[str, int] = {}
        for task in store.list_all(limit=100_000):
            name = str(task.extra.get("collection") or "")
            if name:
                counts[name] = counts.get(name, 0) + 1
        return {"collections": [{"name": name, "count": n} for name, n in sorted(counts.items())]}

    @app.post("/v1/bulk/{name}/pause")
    def bulk_pause(name: str) -> dict[str, Any]:
        job = _bulk_job(name)
        job.set_meta("control", "pause")
        return job.summary()

    @app.post("/v1/bulk/{name}/resume")
    def bulk_resume(name: str) -> dict[str, Any]:
        import time as _time

        job = _bulk_job(name)
        job.set_meta("control", "")
        if float(job.meta("pause_until", 0) or 0) > _time.time():
            job.set_meta("pause_until", _time.time())  # probe YouTube on the runner's next loop
        return job.summary()

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
            "summary_prompt": _configured_prompt(config),
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

    @app.put("/v1/visual/config")
    def update_visual_config(payload: VisualSettingsRequest) -> dict[str, Any]:
        updates = payload.model_dump(exclude_none=True)
        if not updates:
            raise HTTPException(status_code=400, detail="No visual settings provided")
        config.visual.apply_overrides(updates)
        persist_visual_config(config)
        return {"ok": True, "visual": config.visual.as_dict(), "stages": sorted(RERUN_STAGES)}

    @app.get("/v1/summary/config")
    def summary_config() -> dict[str, Any]:
        return {"prompt": _configured_prompt(config)}

    @app.put("/v1/summary/config")
    def update_summary_config(payload: SummaryPromptRequest) -> dict[str, Any]:
        if payload.prompt is None:
            raise HTTPException(status_code=400, detail="No summary prompt provided")
        config.summary.prompt = normalize_prompt(payload.prompt, default=DEFAULT_PROMPT)
        persist_summary_prompt(config)
        return {"ok": True, "prompt": _configured_prompt(config)}

    @app.get("/v1/inbox", status_code=202)
    def inbox(
        request: Request,
        url: str = "",
        asr_model: str | None = None,
        language: str | None = None,
        extract_keyframes: bool = False,
        token: str | None = None,
        x_ingest_token: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Queue a share URL. GET+query is what iOS Shortcuts can send without breaking HTTP.

        Shortcuts' "Get Contents of URL" does not fail on HTTP errors, so every reply, good or bad,
        carries a human-readable ``message`` that the shortcut shows as its notification.
        """
        try:
            _authorize_ingest(config, request, x_ingest_token or token, authorization)
            if not url.strip():
                raise HTTPException(status_code=400, detail="没有收到链接：快捷指令没有把分享的内容传过来")
            queued = _enqueue_share(
                config,
                store,
                worker,
                url,
                asr_model=asr_model,
                language=language,
                extract_keyframes=extract_keyframes,
            )
        except HTTPException as exc:
            logger.warning("Inbox rejected %r: %s", url[:200], exc.detail)
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail, "message": f"❌ 没有加入队列：{exc.detail}"},
            )
        queued["message"] = f"✅ 已加入队列 · {queued['source_label']}"
        return queued

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
        tag: str | None = None,
        collection: str | None = None,
    ) -> dict[str, Any]:
        organize = _resolve_organize(config, filter=filter, group=group, order=order)
        tag_filter = normalize_tag(tag)
        # A collection (a whole channel from a bulk job) has its own view; the library stays yours.
        wanted = (collection or "").strip()
        payloads = [
            _enrich_task(item, config)
            for item in store.list_all(limit=100_000 if wanted else 2000)
            if str(item.extra.get("collection") or "") == wanted
        ]
        groups = organize_task_payloads(
            payloads,
            time_filter=organize["filter"],
            group=organize["group"],
            order=organize["order"],
            tag=tag_filter or None,
        )
        return {
            "tasks": flatten_groups(groups),
            "groups": groups,
            "organize": {**organize, "tag": tag_filter},
        }

    @app.get("/v1/tags")
    def list_tags() -> dict[str, Any]:
        return {"tags": collect_manual_tags(store.list_all())}

    @app.post("/v1/tasks/{task_id}/read")
    def mark_task_read(task_id: str, payload: ReadRequest | None = None) -> dict[str, Any]:
        """Read state lives on the server so the phone and the Mac agree."""
        task = store.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        read = True if payload is None else payload.read
        if read:
            task.extra["read_at"] = task.extra.get("read_at") or _now_stamp()
        else:
            task.extra.pop("read_at", None)
        store.update(task)
        return {"ok": True, "id": task.id, "read_at": task.extra.get("read_at", "")}

    @app.post("/v1/tasks/read-all")
    def mark_all_read(payload: ReadAllRequest | None = None) -> dict[str, Any]:
        """Mark every finished note read, in the library or in one collection."""
        collection = (payload.collection if payload else "") or ""
        stamp = _now_stamp()
        count = 0
        for task in store.list_all(limit=100_000):
            if str(task.extra.get("collection") or "") != collection or task.status != TaskStatus.completed:
                continue
            if not task.extra.get("read_at"):
                task.extra["read_at"] = stamp
                store.update(task)
                count += 1
        return {"ok": True, "marked": count}

    @app.put("/v1/tasks/{task_id}/tags")
    def update_task_tags(task_id: str, payload: TaskTagsRequest) -> dict[str, Any]:
        task = store.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        tags = set_task_tags(task, payload.tags)
        store.update(task)
        return {"ok": True, "id": task.id, "tags": tags}

    @app.get("/v1/tasks/{task_id}")
    def get_task(task_id: str) -> dict[str, Any]:
        task = store.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return _enrich_task(task, config)

    @app.get("/v1/tasks/{task_id}/context")
    def get_context(task_id: str) -> Response:
        """The whole context of a note as plain text (metadata, description, chapters, slides, summary,
        transcript), for pasting into an agent session."""
        from media_pipeline.summary import full_context

        task = store.get(task_id)
        if task is None or not task.video_id:
            raise HTTPException(status_code=404, detail="Task not found")
        artifacts = ArtifactStore(config.paths.artifacts, task.video_id)
        return Response(full_context(artifacts, artifacts.load_metadata(), task.url),
                        media_type="text/plain; charset=utf-8")

    @app.get("/v1/tasks/{task_id}/note")
    def get_note(task_id: str) -> dict[str, Any]:
        """Everything the reader needs for one note, without the debug artifacts."""
        task = store.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return _note_payload(task, config, worker)

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
            payload["summary"] = _public_summary(task, artifacts, worker, config)
            payload["extracted_markdown"] = _extracted_markdown(task, config, artifacts)
        else:
            payload["summary"] = idle_summary(_configured_prompt(config))
            payload["extracted_markdown"] = ""
        return payload

    @app.get("/v1/tasks/{task_id}/summary")
    def get_summary(task_id: str) -> dict[str, Any]:
        task = store.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        if not task.video_id:
            return idle_summary(_configured_prompt(config))
        artifacts = ArtifactStore(config.paths.artifacts, task.video_id)
        return _public_summary(task, artifacts, worker, config)

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
        prompt = normalize_prompt(payload.prompt if payload else None, default=_configured_prompt(config))
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


def _configured_prompt(config: AppConfig) -> str:
    return normalize_prompt(config.summary.prompt, default=DEFAULT_PROMPT)


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
    payload.update(_metadata_fields(artifacts))
    return payload


def _metadata_fields(artifacts: ArtifactStore) -> dict[str, Any]:
    """Small display fields from metadata.json (author, duration, thumbnail) for list rows."""
    try:
        metadata = artifacts.load_metadata()
    except Exception:
        metadata = None
    if metadata is None:
        return {}
    try:
        duration = float(metadata.duration) if metadata.duration not in (None, "") else None
    except (TypeError, ValueError):
        duration = None
    return {
        "author": metadata.author or "",
        "duration": duration,
        "thumbnail_url": metadata.thumbnail_url or "",
        "published": metadata.published or "",
    }


def _note_payload(task: Task, config: AppConfig, worker: object) -> dict[str, Any]:
    payload: dict[str, Any] = {"task": _enrich_task(task, config), "transcript": [], "speakers": [], "frames": []}
    if not task.video_id:
        payload["summary"] = idle_summary(_configured_prompt(config))
        return payload
    from media_pipeline.visual.filtering import caption_for, selected_from_verdicts

    artifacts = ArtifactStore(config.paths.artifacts, task.video_id)
    metadata = artifacts.load_metadata()
    if metadata is not None:
        payload["task"]["description"] = (metadata.description or "")[:20000]
    payload["summary"] = _public_summary(task, artifacts, worker, config)
    named = artifacts.load_named() or []
    stats: dict[str, dict[str, Any]] = {}
    for segment in named:
        payload["transcript"].append(
            {
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "speaker": segment.speaker_id,
                "label": segment.speaker_label,
                "text": segment.text,
            }
        )
        entry = stats.setdefault(
            segment.speaker_id, {"id": segment.speaker_id, "label": segment.speaker_label, "segments": 0, "seconds": 0.0}
        )
        entry["segments"] += 1
        entry["seconds"] += max(0.0, segment.end - segment.start)
    payload["speakers"] = sorted(
        ({**item, "seconds": round(item["seconds"], 1)} for item in stats.values()),
        key=lambda item: -item["seconds"],
    )
    keyframes = artifacts.load_keyframes() or []
    analysis = artifacts.load_frame_analysis() or []
    chosen = selected_from_verdicts(keyframes, analysis) if analysis else keyframes
    for frame in chosen:
        name = Path(frame.image_path).name
        payload["frames"].append(
            {
                "timestamp": round(frame.timestamp, 2),
                "url": f"/v1/videos/{task.video_id}/frames/keyframes/{name}",
                "caption": caption_for(frame, analysis),
            }
        )
    if payload["task"].get("media_kind") == "image":
        payload["post_markdown"] = _extracted_markdown(task, config, artifacts)
    return payload


def _public_summary(task: Task, artifacts: ArtifactStore, worker: object, config: AppConfig) -> dict[str, Any]:
    default_prompt = _configured_prompt(config)
    payload = artifacts.load_summary() or idle_summary(default_prompt)
    running = getattr(worker, "_summarizing", set())
    if payload.get("status") == "running" and task.id not in running:
        payload = {
            **payload,
            "status": "failed",
            "error": payload.get("error") or "Summary was interrupted",
        }
    payload.setdefault("prompt", default_prompt)
    payload["prompt"] = normalize_prompt(str(payload.get("prompt") or ""), default=default_prompt)
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


_INGEST_PROXY_HEADERS = (
    "x-forwarded-for",
    "x-forwarded-proto",
    "x-forwarded-host",
    "forwarded",
    "tailscale-user-login",
    "tailscale-user-name",
    "tailscale-headers-info",
)


def _is_direct_loopback(request: Request) -> bool:
    """True only for a process on this Mac talking to loopback, not Tailscale Serve."""
    host = (request.client.host if request.client else "") or ""
    if not _is_loopback_host(host):
        return False
    for header in _INGEST_PROXY_HEADERS:
        if request.headers.get(header):
            return False
    return True


def _is_loopback_host(host: str) -> bool:
    if host in {"localhost"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _authorize_ingest(
    config: AppConfig,
    request: Request,
    x_ingest_token: str | None,
    authorization: str | None,
) -> None:
    expected = (config.server.ingest_token or "").strip()
    if not expected:
        return
    if _is_direct_loopback(request):
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
