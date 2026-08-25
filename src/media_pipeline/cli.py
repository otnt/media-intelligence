from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from media_pipeline.config import AppConfig, load_config, write_default_config
from media_pipeline.models import ASR_MODELS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="media-pipeline", description="Local video transcription pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the local HTTP API and background worker")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--config", type=Path, default=None)

    doctor = sub.add_parser("doctor", help="Check local dependencies and configuration")
    doctor.add_argument("--config", type=Path, default=None)

    transcribe = sub.add_parser("transcribe", help="Download, transcribe, and write an Obsidian note for one URL")
    transcribe.add_argument("url", help="Bilibili, YouTube, or Xiaohongshu URL")
    transcribe.add_argument(
        "--asr-model",
        default=None,
        help=f"ASR model id. Default comes from config. Choices: {', '.join(ASR_MODELS)}",
    )
    transcribe.add_argument(
        "--language",
        default="auto",
        help="auto (multilingual detect) or a language such as zh, en, Chinese, English",
    )
    transcribe.add_argument("--config", type=Path, default=None)

    retry = sub.add_parser("retry", help="Re-queue a task, reusing downloaded artifacts")
    retry.add_argument("task_id")
    retry.add_argument("--asr-model", default=None)
    retry.add_argument(
        "--stage",
        default=None,
        help="Rerun from this stage: transcribing, detecting_scenes, sampling_frames, deduplicating_frames, ...",
    )
    retry.add_argument("--config", type=Path, default=None)

    status = sub.add_parser("status", help="Show recent tasks")
    status.add_argument("--config", type=Path, default=None)
    status.add_argument("--limit", type=int, default=20)

    init = sub.add_parser("init", help="Write a default config file")
    init.add_argument("--config", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "init":
        path = write_default_config(args.config)
        print(f"Wrote {path}")
        return 0

    config = load_config(getattr(args, "config", None))
    if args.command == "serve":
        return cmd_serve(config, host=args.host, port=args.port)
    if args.command == "doctor":
        return cmd_doctor(config)
    if args.command == "transcribe":
        return cmd_transcribe(config, args.url, args.asr_model, args.language)
    if args.command == "retry":
        return cmd_retry(config, args.task_id, args.asr_model, getattr(args, "stage", None))
    if args.command == "status":
        return cmd_status(config, args.limit)
    parser.error(f"Unknown command {args.command}")
    return 2


def cmd_serve(config: AppConfig, host: str | None, port: int | None) -> int:
    _configure_logging(config)
    config.ensure_directories()
    if config.notes_dir() is None:
        print(
            "Obsidian vault was not found. Set paths.vault in ~/.config/media-pipeline/config.yaml",
            file=sys.stderr,
        )
        return 1
    import uvicorn

    from media_pipeline.api import create_app

    host = host or config.server.host
    port = port or config.server.port
    print(f"API:       http://{host}:{port}/v1/health")
    print(f"Dashboard: http://{host}:{port}/")
    app = create_app(config)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )
    return 0


def cmd_doctor(config: AppConfig) -> int:
    checks: list[tuple[str, bool, str]] = []
    checks.append(("python", sys.version_info < (3, 14), f"{sys.version.split()[0]} (need <3.14 for ML packages)"))
    ffmpeg = shutil.which("ffmpeg")
    checks.append(("ffmpeg", bool(ffmpeg), ffmpeg or "not found"))
    try:
        import yt_dlp

        checks.append(("yt-dlp", True, getattr(yt_dlp, "version", None) and str(yt_dlp.version.__version__) or "ok"))
    except Exception as exc:
        checks.append(("yt-dlp", False, str(exc)))
    from media_pipeline.asr.registry import list_models

    for model in list_models():
        checks.append((f"asr:{model.id}", model.available, model.detail or "ok"))
    try:
        import pyannote.audio  # noqa: F401

        token = bool(config.diarization.hf_token)
        checks.append(("pyannote.audio", True, "installed" + ("" if token else " (HF token missing)")))
    except ImportError:
        checks.append(("pyannote.audio", False, "optional; speaker labels fall back to Speaker 1"))
    try:
        import PIL  # noqa: F401

        checks.append(("pillow", True, "ok"))
    except ImportError:
        checks.append(("pillow", False, "required for keyframe hashing"))
    from media_pipeline.visual.vlm import probe_vlm

    analysis_ok, analysis_detail = probe_vlm(config)
    checks.append(("analysis:qwen3.8", analysis_ok, analysis_detail or "ok"))
    try:
        import curl_cffi

        checks.append(("curl-cffi", True, getattr(curl_cffi, "__version__", "ok")))
    except Exception as exc:
        checks.append(("curl-cffi", False, str(exc)))
    vault = config.paths.vault
    checks.append(("obsidian vault", bool(vault and vault.exists()), str(vault) if vault else "not configured"))
    notes = config.notes_dir()
    checks.append(("notes dir", notes is not None, str(notes) if notes else "unavailable"))

    failed = 0
    for name, ok, detail in checks:
        optional = name.startswith("asr:qwen") or name.startswith("analysis:") or name == "pyannote.audio"
        if ok:
            mark = "ok"
        elif optional:
            mark = "WARN"
        else:
            mark = "FAIL"
            failed += 1
        print(f"{mark:4}  {name:24} {detail}")
    print(f"config: {config.source_path or '(defaults)'}")
    print(f"API:    http://{config.server.host}:{config.server.port}")
    print(f"UI:     http://{config.server.host}:{config.server.port}/")
    return 0 if failed == 0 else 1


def cmd_transcribe(config: AppConfig, url: str, asr_model: str | None, language: str | None = "auto") -> int:
    import uuid

    from media_pipeline.media import UnsupportedURLError, canonicalize_url, parse_video_ref
    from media_pipeline.models import Task, TaskStatus, asr_label
    from media_pipeline.pipeline import Pipeline
    from media_pipeline.store import TaskStore

    model_id = (asr_model or config.asr.default).strip()
    if model_id not in ASR_MODELS:
        print(f"Unknown ASR model {model_id!r}. Supported: {', '.join(ASR_MODELS)}", file=sys.stderr)
        return 2
    url = canonicalize_url(url.strip())
    try:
        platform, video_id = parse_video_ref(url)
    except UnsupportedURLError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if config.notes_dir() is None:
        print(
            "Obsidian vault was not found. Set paths.vault in ~/.config/media-pipeline/config.yaml",
            file=sys.stderr,
        )
        return 1
    _configure_logging(config)
    config.ensure_directories()
    store = TaskStore(config.paths.db)
    task = Task(
        id=str(uuid.uuid4()),
        url=url.strip(),
        asr_model=model_id,
        status=TaskStatus.queued,
        platform=platform,
        video_id=video_id,
        extra={"language": language or config.asr.language or "auto"},
    )
    store.insert(task)
    print(f"Task: {task.id}")
    print(f"ASR:  {asr_label(model_id)}")
    print(f"Language: {task.extra['language']}")
    print(f"URL:  {task.url}")
    result = Pipeline(config, store).run(task)
    print()
    print(f"Status: {result.status.value}")
    if result.note_path:
        print(f"Note:   {result.note_path}")
    if result.video_path:
        print(f"Video:  {result.video_path}")
    if result.audio_path:
        print(f"Audio:  {result.audio_path}")
    if result.error:
        print(f"Stage:  {result.error_stage}")
        print(f"Error:  {result.error}", file=sys.stderr)
    return 0 if result.status == TaskStatus.completed else 1


def cmd_retry(config: AppConfig, task_id: str, asr_model: str | None, stage: str | None = None) -> int:
    from media_pipeline.artifacts import ArtifactStore
    from media_pipeline.models import TaskStatus
    from media_pipeline.pipeline import Pipeline
    from media_pipeline.stage_timing import clear_invalidated_timings
    from media_pipeline.store import TaskStore

    if asr_model and asr_model not in ASR_MODELS:
        print(f"Unknown ASR model {asr_model}", file=sys.stderr)
        return 2
    config.ensure_directories()
    store = TaskStore(config.paths.db)
    task = store.get(task_id)
    if task is None:
        print(f"Task not found: {task_id}", file=sys.stderr)
        return 1
    if asr_model:
        task.asr_model = asr_model
    if stage:
        task.extra["rerun_stage"] = stage
        clear_invalidated_timings(task.extra, stage)
        if task.video_id:
            ArtifactStore(config.paths.artifacts, task.video_id).clear_invalidated_timings(stage)
    task.status = TaskStatus.queued
    task.error = ""
    task.error_stage = ""
    store.update(task)
    result = Pipeline(config, store).run(task)
    print(json.dumps(result.to_public(), ensure_ascii=False, indent=2))
    return 0 if result.status.value == "completed" else 1


def cmd_status(config: AppConfig, limit: int) -> int:
    from media_pipeline.store import TaskStore

    store = TaskStore(config.paths.db)
    tasks = store.list_recent(limit=limit)
    if not tasks:
        print("No tasks yet.")
        return 0
    for task in tasks:
        title = task.title or task.video_id or task.url
        extra = f" error={task.error}" if task.error else ""
        print(f"{task.status.value:18} {task.asr_model:24} {title}{extra}")
    return 0


def _configure_logging(config: AppConfig) -> None:
    config.paths.logs.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.paths.logs / "pipeline.log", encoding="utf-8"),
        ],
    )
