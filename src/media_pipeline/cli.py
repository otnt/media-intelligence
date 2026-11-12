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
    serve.add_argument(
        "--lan",
        action="store_true",
        help="Bind 0.0.0.0 so a phone on the same Wi-Fi can queue tasks from a Shortcut",
    )
    serve.add_argument(
        "--tailscale",
        action="store_true",
        help="Publish the dashboard on the Tailscale tailnet via Serve (HTTPS MagicDNS). Never enables Funnel.",
    )
    serve.add_argument("--config", type=Path, default=None)

    doctor = sub.add_parser("doctor", help="Check local dependencies and configuration")
    doctor.add_argument("--config", type=Path, default=None)

    transcribe = sub.add_parser("transcribe", help="Download, transcribe, and write an Obsidian note for one URL")
    transcribe.add_argument("url", help="Bilibili, YouTube, Xiaohongshu, or RedNote URL")
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
    transcribe.add_argument(
        "--keyframes",
        action="store_true",
        help="Extract stills and run the visual pipeline (slow; off by default)",
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

    clean = sub.add_parser(
        "cleanup",
        help="Free disk: drop downloaded videos/WAVs of finished tasks, keep a small audio copy (storage.*)",
    )
    clean.add_argument("--dry-run", action="store_true", help="Only report what would be removed")
    clean.add_argument("--config", type=Path, default=None)

    bulk = sub.add_parser(
        "bulk",
        help="Download a whole channel or playlist politely, with a per-video progress table at /bulk",
    )
    bulk_sub = bulk.add_subparsers(dest="bulk_command", required=True)
    bulk_add = bulk_sub.add_parser("add", help="Create a job, or add videos that are new since the last add")
    bulk_add.add_argument("name", help="Short job name, e.g. aie")
    bulk_add.add_argument("--source", required=True, help="Channel/playlist URL, or a file of URLs or id|seconds|title lines")
    bulk_add.add_argument("--title", default="", help="Display name, e.g. 'AI Engineer'")
    bulk_add.add_argument("--dest", default=None, help="Where finished files go, e.g. a NAS share under /Volumes")
    bulk_run = bulk_sub.add_parser("run", help="Download until done; pauses itself when YouTube limits it")
    bulk_run.add_argument("name")
    bulk_run.add_argument("--max", type=int, default=None, help="Stop after this many videos (a trial run)")
    bulk_run.add_argument("--workers", type=int, default=1)
    bulk_run.add_argument("--min-free-gb", type=float, default=40.0, help="Pause while the Mac has less free space")
    bulk_run.add_argument("--asr-workers", type=int, default=3, help="Videos transcribed at the same time")
    bulk_asr = bulk_sub.add_parser("asr", help="Transcribe downloaded videos as part of the job (or --off)")
    bulk_asr.add_argument("name")
    bulk_asr.add_argument("--model", default=None, help="ASR model id; default is asr.default from the config")
    bulk_asr.add_argument("--off", action="store_true", help="Stop transcribing (downloads continue)")
    bulk_asr.add_argument(
        "--policy",
        choices=("captions-first", "asr"),
        default=None,
        help="captions-first (default): YouTube captions where they exist, the ASR model for the rest; asr: always ASR",
    )
    bulk_status = bulk_sub.add_parser("status", help="Show a job's progress (all jobs without a name)")
    bulk_status.add_argument("name", nargs="?")
    bulk_dest = bulk_sub.add_parser("dest", help="Set where finished files go ('' = keep them on the Mac)")
    bulk_dest.add_argument("name")
    bulk_dest.add_argument("path")
    bulk_summary = bulk_sub.add_parser(
        "summary", help="Summarize transcribed videos (a note + summary each), inside the half-price window"
    )
    bulk_summary.add_argument("name")
    bulk_summary.add_argument("--model", default="deepseek-v4.1-flash", help="Summary model key")
    bulk_summary.add_argument("--window", default=None, help="HH:MM-HH:MM@+08:00 (default 22:00-08:00 Beijing)")
    bulk_summary.add_argument("--any-time", action="store_true", help="No time window")
    bulk_summary.add_argument("--off", action="store_true")
    bulk_import = bulk_sub.add_parser(
        "import-aie", help="Take transcripts, chapters and slides from AI Engineer's public talk library"
    )
    bulk_import.add_argument("name")
    bulk_import.add_argument("--slides", type=int, default=8, help="Slide images per talk for the summary")
    bulk_import.add_argument("--refresh", action="store_true", help="Fetch again even what is already saved")
    bulk_import.add_argument("--limit", type=int, default=None, help="Only the first N matching talks (a trial)")
    bulk_archive = bulk_sub.add_parser(
        "archive-aie", help="Save everything AI Engineer publishes (index, talk pages, transcripts, all slides)"
    )
    bulk_archive.add_argument("name")
    bulk_archive.add_argument("--dest", default=None, help="Folder (default: <job dest>/official, e.g. on the NAS)")
    bulk_learn = bulk_sub.add_parser("learn", help="Topic syntheses and a learning plan from the collection's summaries")
    bulk_learn.add_argument("name")
    bulk_learn.add_argument("--wait", action="store_true",
                            help="Wait for the half-price window and for the summaries to finish first")
    bulk_learn.add_argument("--min-topic-talks", type=int, default=8)
    bulk_learn.add_argument("--model", default="deepseek-v4.1-flash")
    bulk_youtube = bulk_sub.add_parser("youtube", help="Stop (--off) or restart (--on) downloads and captions from YouTube")
    bulk_youtube.add_argument("name")
    bulk_youtube.add_argument("switch", choices=("on", "off"))
    bulk_pace = bulk_sub.add_parser("pace", help="Seconds between downloads, e.g. 45-90 (applies while it runs)")
    bulk_pace.add_argument("name")
    bulk_pace.add_argument("seconds", help="MIN-MAX, or 'default'")
    bulk_pause = bulk_sub.add_parser("pause", help="Pause a running job after the current video")
    bulk_pause.add_argument("name")
    bulk_resume = bulk_sub.add_parser("resume", help="Undo a pause; a YouTube-limit pause is probed right away")
    bulk_resume.add_argument("name")
    for parser_ in (bulk_add, bulk_run, bulk_status, bulk_dest, bulk_pause, bulk_resume, bulk_asr, bulk_summary,
                    bulk_pace, bulk_import, bulk_youtube, bulk_archive, bulk_learn):
        parser_.add_argument("--config", type=Path, default=None)

    watchdog = sub.add_parser("watchdog", help="One self-check: API, bulk jobs, disk; restarts what is stuck")
    watchdog.add_argument("--config", type=Path, default=None)

    init = sub.add_parser("init", help="Write a default config file")
    init.add_argument("--config", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "init":
        path = write_default_config(args.config)
        print(f"Wrote {path}")
        return 0

    config = load_config(getattr(args, "config", None))
    if args.command == "serve":
        return cmd_serve(
            config,
            host=args.host,
            port=args.port,
            lan=args.lan,
            tailscale=args.tailscale,
        )
    if args.command == "doctor":
        return cmd_doctor(config)
    if args.command == "transcribe":
        return cmd_transcribe(config, args.url, args.asr_model, args.language, args.keyframes)
    if args.command == "retry":
        return cmd_retry(config, args.task_id, args.asr_model, getattr(args, "stage", None))
    if args.command == "status":
        return cmd_status(config, args.limit)
    if args.command == "cleanup":
        return cmd_cleanup(config, dry_run=args.dry_run)
    if args.command == "bulk":
        return cmd_bulk(config, args)
    if args.command == "watchdog":
        from media_pipeline.watchdog import check

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                            handlers=[logging.StreamHandler(sys.stdout)])
        result = check(config)
        print(json.dumps({key: result[key] for key in ("ok", "problems", "notes", "actions")}, ensure_ascii=False))
        return 0
    parser.error(f"Unknown command {args.command}")
    return 2


def cmd_serve(
    config: AppConfig,
    host: str | None,
    port: int | None,
    lan: bool = False,
    tailscale: bool = False,
) -> int:
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
    from media_pipeline.config import ensure_ingest_token

    host = host or ("0.0.0.0" if lan else config.server.host)
    port = port or config.server.port
    created_token = None
    if lan or tailscale:
        created_token = ensure_ingest_token(config)
    published = None
    if tailscale:
        from media_pipeline.tailscale import publish_local_http

        published = publish_local_http(port)
        if not published.ok:
            print(f"Tailscale: {published.detail}", file=sys.stderr)
            return 1
    print(f"API:       http://{host}:{port}/v1/health")
    print(f"Dashboard: http://{host}:{port}/")
    lan_ip = _lan_ipv4() if host in {"0.0.0.0", "::"} else None
    if lan_ip:
        print(f"Phone:     http://{lan_ip}:{port}/")
        print(f"Inbox:     http://{lan_ip}:{port}/v1/inbox?url=")
    if published and published.url:
        print(f"Phone/Tailscale: {published.url}")
        print(f"Inbox/Tailscale: {published.url.rstrip('/')}/v1/inbox?url=")
    if created_token:
        print("Ingest:    created server.ingest_token (save this for Shortcuts; shown once)")
        print(f"           {created_token}")
        print("           Send as X-Ingest-Token, Authorization: Bearer, or ?token= on /v1/inbox")
    elif (lan or tailscale) and config.server.ingest_token:
        print("Ingest:    send X-Ingest-Token (or ?token=) from Shortcuts; localhost dashboard is exempt")
    elif lan and not config.server.ingest_token:
        print("Ingest:    set server.ingest_token before exposing the API on your LAN")
    app = create_app(config)
    http_impl = "h11"
    try:
        import httptools  # noqa: F401

        http_impl = "httptools"
    except ImportError:
        pass
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        http=http_impl,
    )
    return 0


def cmd_bulk(config: AppConfig, args: argparse.Namespace) -> int:
    import time

    from media_pipeline.bulk import BulkRunner, Limits, dest_ready, list_jobs, open_job, read_list

    command = args.bulk_command
    if command == "status":
        names = [args.name] if args.name else list_jobs(config)
        if not names:
            print("No bulk jobs yet: media-pipeline bulk add NAME --source URL")
        for name in names:
            summary = open_job(config, name).summary()
            counts = summary["counts"]
            print(
                f"{name} ({summary['title']}): {summary['state']} {summary['state_detail']}".rstrip() + "\n"
                f"  done {counts.get('done', 0)}/{summary['total']} · pending {counts.get('pending', 0)} · "
                f"failed {counts.get('failed', 0)} · skipped {counts.get('skipped', 0)} · "
                f"{summary['gigabytes']} GB ({summary['on_nas']} on NAS)\n"
                f"  dest: {summary['dest'] or '(Mac only)'} {'ready' if summary['dest_ready'] else 'not mounted'}"
                f" · staging {summary['staging']} · free {summary['free_gb']} GB"
            )
            if summary["summary_model"]:
                done = summary["summary_counts"]
                print(f"  summaries ({summary['summary_model']}, {summary['summary_window'] or 'any time'}): "
                      f"done {done.get('done', 0)} · running {done.get('running', 0) + done.get('queued', 0)} · "
                      f"failed {done.get('failed', 0)} · ≈ ${summary['summary_cost_usd']}")
            if summary["asr_model"]:
                asr = summary["asr_counts"]
                print(
                    f"  asr ({summary['asr_model']}): done {asr.get('done', 0)} · running {asr.get('running', 0)} · "
                    f"waiting {asr.get('pending', 0)} · failed {asr.get('failed', 0)} · "
                    f"{summary['asr_hours']} h ≈ ${summary['asr_cost_usd']}"
                )
        return 0
    store = open_job(config, args.name, create=command == "add")
    if command == "add":
        entries = read_list(args.source)
        added = store.add_videos(entries)
        store.set_meta("name", args.name)
        if args.title or not store.meta("title"):
            store.set_meta("title", args.title or args.name)
        store.set_meta("source", args.source)
        if store.meta("youtube_extras") is None:  # captions, chapters and metadata come with each download
            store.set_meta("youtube_extras", all(len(entry["id"]) == 11 for entry in entries[:20]) and bool(entries))
        if args.dest is not None:
            store.set_meta("dest", str(Path(args.dest).expanduser()) if args.dest else "")
        print(f"{args.name}: {added} new of {len(entries)} listed; {store.summary()['total']} in the job")
        return 0
    if command == "dest":
        dest = str(Path(args.path).expanduser()) if args.path else ""
        store.set_meta("dest", dest)
        ready = dest_ready(dest) is not None
        print(f"dest = {dest or '(Mac only)'}: {'ready' if ready else 'not mounted yet; files move there once it is'}")
        return 0
    if command == "asr":
        if args.off:
            store.set_meta("asr_model", "")
            print(f"{args.name}: transcription off (restart the job to apply)")
            return 0
        from media_pipeline.models import ASR_MODELS

        model = args.model or config.asr.default
        if model not in ASR_MODELS:
            print(f"Unknown ASR model {model!r}. Choices: {', '.join(ASR_MODELS)}", file=sys.stderr)
            return 2
        store.set_meta("asr_model", model)
        if args.policy:
            store.set_meta("transcript_policy", args.policy)
        print(f"{args.name}: transcribing with {model}, {store.meta('transcript_policy', 'captions-first')} "
              f"(restart the job to apply: "
              f"launchctl kickstart -k gui/$(id -u)/com.media-pipeline.bulk.{args.name})")
        return 0
    if command == "summary":
        from media_pipeline.bulk import DEFAULT_SUMMARY_WINDOW
        from media_pipeline.summary_llm import normalize_summary_key

        if args.off:
            store.set_meta("summary_model", "")
            print(f"{args.name}: summaries off (restart the job to apply)")
            return 0
        store.set_meta("summary_model", normalize_summary_key(args.model))
        store.set_meta("summary_window", "" if args.any_time else (args.window or DEFAULT_SUMMARY_WINDOW))
        window = store.meta("summary_window") or "any time"
        print(f"{args.name}: summaries with {store.meta('summary_model')} during {window} (restart the job to apply: "
              f"launchctl kickstart -k gui/$(id -u)/com.media-pipeline.bulk.{args.name})")
        return 0
    if command == "import-aie":
        from media_pipeline.aie_library import import_job

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                            handlers=[logging.StreamHandler(sys.stdout)])
        stats = import_job(config, store, slides_per_talk=max(0, args.slides), refresh=args.refresh, limit=args.limit)
        store.event("import", f"AI Engineer 官方资料：{stats['transcripts']} 个转写、{stats['slides']} 个有幻灯片、"
                              f"{stats['images']} 张幻灯片图片，失败 {stats['failed']}")
        print(json.dumps(stats))
        return 0
    if command == "archive-aie":
        from media_pipeline.aie_library import Library, archive_library

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                            handlers=[logging.StreamHandler(sys.stdout)])
        if args.dest:
            dest = Path(args.dest).expanduser()
        else:
            base = dest_ready(store.meta("dest", ""))
            if base is None:
                print("The job's dest (the NAS) is not mounted; pass --dest", file=sys.stderr)
                return 1
            dest = base / "official"
        stats = archive_library(dest, library=Library(asset_interval=0.15), artifacts_root=config.paths.artifacts)
        store.event("import", f"官方资料已归档到 {dest}：{stats['talks']} 个演讲，{stats['slides'] + stats['slides_copied']} 张幻灯片")
        print(json.dumps(stats))
        return 0
    if command == "learn":
        from media_pipeline.learning import LEARN_DIR, generate_fn, run_learning

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                            handlers=[logging.StreamHandler(sys.stdout)])
        notes = config.notes_dir()
        if notes is None:
            print("Obsidian vault is not configured", file=sys.stderr)
            return 1
        if args.wait:
            _wait_for_summaries(store)
        library = _aie_library_index(store)
        out_dir = notes / str(store.meta("title", "") or args.name) / LEARN_DIR
        stats = run_learning(config, store, library, out_dir, generate=generate_fn(config, args.model),
                             min_topic_talks=args.min_topic_talks)
        store.event("import", f"学习材料已生成：{stats['topics']} 份主题综述 + 学习路线（{out_dir}）")
        print(json.dumps(stats, ensure_ascii=False))
        return 0
    if command == "youtube":
        on = args.switch == "on"
        store.set_meta("downloads", on)
        store.set_meta("captions", on)
        print(f"{args.name}: YouTube downloads and captions {'on' if on else 'off'} "
              f"(restart the job to apply: launchctl kickstart -k gui/$(id -u)/com.media-pipeline.bulk.{args.name})")
        return 0
    if command == "pace":
        if args.seconds == "default":
            store.set_meta("pause_between", None)
            print(f"{args.name}: default pacing")
            return 0
        low, _, high = args.seconds.partition("-")
        pace = sorted([float(low), float(high or low)])
        store.set_meta("pause_between", pace)
        hourly = 3600 / (sum(pace) / 2 + 20)  # ~20 s per download
        print(f"{args.name}: {pace[0]:.0f}-{pace[1]:.0f} s between downloads (about {hourly:.0f} videos an hour)")
        return 0
    if command == "pause":
        store.set_meta("control", "pause")
        print(f"{args.name}: pausing after the current video")
        return 0
    if command == "resume":
        store.set_meta("control", "")
        if float(store.meta("pause_until", 0) or 0) > time.time():
            store.set_meta("pause_until", time.time())  # probe YouTube on the next loop
        print(f"{args.name}: resumed")
        return 0
    if command == "run":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                            handlers=[logging.StreamHandler(sys.stdout)])
        limits = Limits(workers=max(1, args.workers), min_free_gb=args.min_free_gb, asr_workers=max(0, args.asr_workers))
        state = BulkRunner(config, store, limits=limits).run(max_videos=args.max)
        print(f"{args.name}: {state}")
        return 0
    return 2


def _wait_for_summaries(store, poll: float = 300.0) -> None:
    """Until the half-price window is open and the job's summaries have finished (or stopped moving)."""
    import time

    from media_pipeline.bulk import DEFAULT_SUMMARY_WINDOW, in_window

    last_done, still_since = -1, time.time()
    while True:
        summary = store.summary()
        counts = summary["summary_counts"]
        done = counts.get("done", 0)
        waiting = counts.get("pending", 0) + counts.get("queued", 0) + counts.get("running", 0)
        if done != last_done:
            last_done, still_since = done, time.time()
        window = store.meta("summary_window", DEFAULT_SUMMARY_WINDOW)
        settled = waiting == 0 or (done > 0 and time.time() - still_since > 2700)
        if in_window(window, time.time()) and settled:
            return
        logging.info("waiting: %s summaries done, %s to go, window open: %s", done, waiting, in_window(window, time.time()))
        time.sleep(poll)


def _aie_library_index(store) -> dict:
    """talks + topics of the official library: from the NAS archive when it is there, else fetched once."""
    from media_pipeline.aie_library import BASE, Library, _rows
    from media_pipeline.bulk import dest_ready

    base = dest_ready(store.meta("dest", ""))
    index = base / "official" / "index" if base else None
    data = {}
    for name in ("talks", "topics"):
        path = index / f"{name}.json" if index else None
        if path is not None and path.is_file():
            data[name] = _rows(json.loads(path.read_text(encoding="utf-8")))
        else:
            data[name] = _rows(Library().json(f"{BASE}/{name}?format=json"))
    return data


def cmd_cleanup(config: AppConfig, *, dry_run: bool) -> int:
    from media_pipeline.storage import cleanup
    from media_pipeline.store import TaskStore

    report = cleanup(config, TaskStore(config.paths.db), dry_run=dry_run)
    print(report.summary())
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
    from media_pipeline.summary_llm import catalog_summary_backends

    for item in catalog_summary_backends(config):
        if item["key"] == "qwen" or str(item["key"]).startswith("qwen-"):
            continue
        checks.append((f"summary:{item['key']}", item["available"], item["detail"] or item["model"]))
    try:
        import curl_cffi

        checks.append(("curl-cffi", True, getattr(curl_cffi, "__version__", "ok")))
    except Exception as exc:
        checks.append(("curl-cffi", False, str(exc)))
    vault = config.paths.vault
    checks.append(("obsidian vault", bool(vault and vault.exists()), str(vault) if vault else "not configured"))
    notes = config.notes_dir()
    checks.append(("notes dir", notes is not None, str(notes) if notes else "unavailable"))
    from media_pipeline.tailscale import doctor_checks

    checks.extend(doctor_checks())

    failed = 0
    for name, ok, detail in checks:
        optional = (
            name.startswith("asr:qwen")
            or name.startswith("analysis:")
            or name.startswith("summary:")
            or name.startswith("tailscale")
            or name == "pyannote.audio"
        )
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


def cmd_transcribe(
    config: AppConfig,
    url: str,
    asr_model: str | None,
    language: str | None = "auto",
    extract_keyframes: bool = False,
) -> int:
    import uuid

    from media_pipeline.media import UnsupportedURLError, extract_supported_url, parse_video_ref
    from media_pipeline.models import Task, TaskStatus, asr_label
    from media_pipeline.pipeline import Pipeline
    from media_pipeline.store import TaskStore

    model_id = (asr_model or config.asr.default).strip()
    if model_id not in ASR_MODELS:
        print(f"Unknown ASR model {model_id!r}. Supported: {', '.join(ASR_MODELS)}", file=sys.stderr)
        return 2
    try:
        url = extract_supported_url(url)
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
        extra={
            "language": language or config.asr.language or "auto",
            "extract_keyframes": bool(extract_keyframes),
        },
    )
    store.insert(task)
    print(f"Task: {task.id}")
    print(f"ASR:  {asr_label(model_id)}")
    print(f"Language: {task.extra['language']}")
    print(f"Keyframes: {'on' if extract_keyframes else 'off'}")
    print(f"URL:  {task.url}")
    pipeline = Pipeline(config, store)
    try:
        result = pipeline.run(task)
    finally:
        pipeline.close()
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
    from media_pipeline.pipeline import VISUAL_RERUN_STAGES, Pipeline
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
        if stage in VISUAL_RERUN_STAGES:
            task.extra["extract_keyframes"] = True
        clear_invalidated_timings(task.extra, stage)
        if task.video_id:
            ArtifactStore(config.paths.artifacts, task.video_id).clear_invalidated_timings(stage)
    task.status = TaskStatus.queued
    task.error = ""
    task.error_stage = ""
    store.update(task)
    pipeline = Pipeline(config, store)
    try:
        result = pipeline.run(task)
    finally:
        pipeline.close()
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


def _lan_ipv4() -> str | None:
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
    except OSError:
        return None
    finally:
        sock.close()
    if not ip or ip.startswith("127."):
        return None
    return ip


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
