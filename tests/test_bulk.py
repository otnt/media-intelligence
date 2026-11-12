from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from media_pipeline import bulk
from media_pipeline.bulk import BulkRunner, Limits, classify, dest_ready, find_bulk_media, open_job, read_list
from media_pipeline.config import AppConfig, PathsConfig
from media_pipeline.models import Task, TaskStatus, VideoMetadata
from media_pipeline.store import TaskStore


def _config(tmp_path: Path) -> AppConfig:
    config = AppConfig(
        paths=PathsConfig(
            videos=tmp_path / "videos", audio=tmp_path / "audio", artifacts=tmp_path / "artifacts",
            logs=tmp_path / "logs", vault=tmp_path / "vault", db=tmp_path / "tasks.sqlite3", bulk=tmp_path / "bulk",
        )
    )
    config.ensure_directories()
    return config


class Clock:
    def __init__(self) -> None:
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(seconds, 0.001)


class FakeYouTube:
    """Writes a real file per video; `script` maps a video id to errors raised on its first calls."""

    def __init__(self, script=None, size=6_000_000, seconds=2.0):
        self.script = {vid: list(errors) for vid, errors in (script or {}).items()}
        self.size = size
        self.seconds = seconds
        self.calls: list[str] = []

    def __call__(self, url, video_id, target_dir, progress):
        self.calls.append(video_id)
        errors = self.script.get(video_id) or []
        if errors:
            raise RuntimeError(errors.pop(0))
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{video_id}.mp4"
        path.write_bytes(b"v" * 1000)
        progress(500)
        return path, self.size, self.seconds


def _job(config: AppConfig, ids=("a", "b", "c"), **meta) -> bulk.BulkStore:
    store = open_job(config, "aie", create=True)
    store.add_videos([{"id": vid, "title": f"Talk {vid}", "duration": 1800} for vid in ids])
    for key, value in meta.items():
        store.set_meta(key, value)
    return store


def _runner(config, store, clock, youtube, probe=lambda: None, free=lambda path: 500.0, **limits):
    return BulkRunner(config, store, limits=Limits(pause_between=(0, 0), **limits), downloader=youtube,
                      prober=probe, clock=clock, sleep=clock.sleep, free_gb=free)


def _row(store, vid):
    return next(row for row in store.rows() if row["id"] == vid)


def _events(store, kind):
    return [event for event in reversed(store.events(200)) if event["kind"] == kind]


@pytest.mark.parametrize(
    "message, kind",
    [
        ("[youtube] x: Sign in to confirm you’re not a bot. Use --cookies-from-browser", "blocked"),
        ("HTTP Error 429: Too Many Requests", "blocked"),
        ("[youtube] x: This content isn't available, try again later.", "blocked"),
        ("[youtube] x: Private video. Sign in if you've been granted access to this video", "skipped"),
        ("[youtube] x: Sign in to confirm your age. This video may be inappropriate for some users.", "skipped"),
        ("[youtube] x: Join this channel to get access to members-only content", "skipped"),
        ("unable to download video data: HTTP Error 403: Forbidden", "failed"),
    ],
)
def test_classify_tells_limits_from_unavailable_videos(message, kind):
    assert classify(message) == kind


def test_add_reads_list_files_and_keeps_known_videos(tmp_path: Path):
    config = _config(tmp_path)
    listing = tmp_path / "videos.txt"
    listing.write_text("# AI Engineer\nabc123def45|1609|Pick Problems | That Survive\n"
                       "https://www.youtube.com/watch?v=zzz999yyy88\n", encoding="utf-8")
    entries = read_list(str(listing))
    assert entries[0] == {"id": "abc123def45", "duration": 1609.0, "title": "Pick Problems | That Survive"}
    assert entries[1]["id"] == "zzz999yyy88"
    store = open_job(config, "aie", create=True)
    assert store.add_videos(entries) == 2
    store.execute("UPDATE videos SET status='done' WHERE id='abc123def45'")
    assert store.add_videos(entries) == 0
    assert _row(store, "abc123def45")["status"] == "done"
    with pytest.raises(ValueError):
        open_job(config, "../escape", create=True)


def test_run_downloads_all_then_moves_to_the_nas_once_it_is_set(tmp_path: Path):
    config = _config(tmp_path)
    store = _job(config)
    clock, youtube = Clock(), FakeYouTube()
    assert _runner(config, store, clock, youtube).run() == "finished"
    assert youtube.calls == ["a", "b", "c"]  # channel order
    staged = _row(store, "a")
    assert staged["status"] == "done" and staged["location"] == "mac" and staged["bytes"] == 6_000_000
    assert find_bulk_media(config, "a") == store.db.parent / "media" / "a.mp4"

    nas = tmp_path / "nas" / "AI Engineer"
    store.set_meta("dest", str(nas))
    _runner(config, store, clock, FakeYouTube()).run()
    assert {row["location"] for row in store.rows()} == {"nas"}
    assert sorted(p.name for p in nas.iterdir()) == ["a.mp4", "b.mp4", "c.mp4"]
    assert not list((store.db.parent / "media").glob("*.mp4"))
    assert find_bulk_media(config, "b") == nas / "b.mp4"
    assert find_bulk_media(config, "missing") is None


def test_youtube_block_pauses_probes_then_resumes_without_using_an_attempt(tmp_path: Path):
    config = _config(tmp_path)
    store = _job(config, ids=("a", "b"))
    clock = Clock()
    youtube = FakeYouTube({"a": ["[youtube] a: Sign in to confirm you're not a bot"]})
    probes = []
    started = clock.t
    assert _runner(config, store, clock, youtube, probe=lambda: probes.append(clock.t)).run() == "finished"
    assert youtube.calls == ["a", "a", "b"]
    assert len(probes) == 1 and probes[0] - started >= 15 * 60
    row = _row(store, "a")
    assert row["status"] == "done" and row["attempts"] == 1
    assert len(_events(store, "pause")) == 1 and len(_events(store, "resume")) == 1
    summary = store.summary(now=clock.t)
    assert summary["pause_until"] == 0 and summary["last_probe"]["ok"] is True and summary["last_ok"] > probes[0]


def test_failed_probe_keeps_the_job_paused_with_a_longer_wait(tmp_path: Path):
    config = _config(tmp_path)
    store = _job(config, ids=("a",))
    clock = Clock()
    outcomes = [RuntimeError("HTTP Error 429: Too Many Requests"), None]

    def probe():
        outcome = outcomes.pop(0)
        if outcome:
            raise outcome

    youtube = FakeYouTube({"a": ["HTTP Error 429: Too Many Requests"]})
    started = clock.t
    _runner(config, store, clock, youtube, probe=probe).run()
    pauses = _events(store, "pause")
    assert len(pauses) == 2 and "15 分钟" in pauses[0]["detail"] and "30 分钟" in pauses[1]["detail"]
    assert clock.t - started >= 45 * 60
    assert _row(store, "a")["status"] == "done"


def test_unavailable_videos_are_skipped_and_failures_in_a_row_pause(tmp_path: Path):
    config = _config(tmp_path)
    store = _job(config, ids=("p", "f1", "f2", "f3", "f4", "ok"))
    clock = Clock()
    broken = ["unable to download video data: HTTP Error 403: Forbidden"]
    youtube = FakeYouTube({"p": ["Private video. Sign in if you've been granted access"],
                           "f1": broken * 3, "f2": broken * 3, "f3": broken * 3, "f4": broken * 3})
    assert _runner(config, store, clock, youtube, fail_streak=4).run() == "finished"
    assert _row(store, "p")["status"] == "skipped"
    assert all(_row(store, vid)["status"] == "failed" and _row(store, vid)["attempts"] == 3
               for vid in ("f1", "f2", "f3", "f4"))
    assert _row(store, "ok")["status"] == "done"
    assert any("连续 4 个" in event["detail"] for event in _events(store, "pause"))


def test_a_video_that_keeps_saying_unavailable_is_eventually_failed(tmp_path: Path):
    config = _config(tmp_path)
    store = _job(config, ids=("gone", "ok"))
    clock = Clock()
    youtube = FakeYouTube({"gone": ["This content isn't available, try again later."] * 10})
    _runner(config, store, clock, youtube).run()
    gone = _row(store, "gone")
    assert gone["status"] == "failed" and youtube.calls.count("gone") == 3
    assert _row(store, "ok")["status"] == "done"


def test_slow_downloads_are_reported_as_throttling(tmp_path: Path):
    config = _config(tmp_path)
    store = _job(config, ids=tuple(f"v{i}" for i in range(6)))
    clock = Clock()
    youtube = FakeYouTube(size=6_000_000, seconds=100.0)  # 60 KB/s
    _runner(config, store, clock, youtube).run()
    pauses = _events(store, "pause")
    assert len(pauses) == 1 and "疑似被限速" in pauses[0]["detail"]
    assert all(row["status"] == "done" for row in store.rows())


def test_low_disk_waits_and_manual_pause_holds(tmp_path: Path):
    config = _config(tmp_path)
    store = _job(config, ids=("a",))
    clock = Clock()
    free = [10.0, 10.0]
    runner = _runner(config, store, clock, FakeYouTube(), free=lambda path: free.pop(0) if free else 500.0)
    states = []
    original = runner._set_state
    runner._set_state = lambda state, detail: (states.append((state, detail)), original(state, detail))
    runner.run()
    assert states[0][0] == "paused" and "Mac 只剩 10 GB" in states[0][1]
    assert _row(store, "a")["status"] == "done"

    store.add_videos([{"id": "b"}])
    store.set_meta("control", "pause")
    calls = {"n": 0}

    def sleep(seconds):
        calls["n"] += 1
        clock.sleep(seconds)
        if calls["n"] == 3:
            store.set_meta("control", "")  # the phone pressed 继续

    runner = BulkRunner(config, store, limits=Limits(pause_between=(0, 0)), downloader=FakeYouTube(),
                        prober=lambda: None, clock=clock, sleep=sleep, free_gb=lambda path: 500.0)
    runner.run()
    assert calls["n"] >= 3 and _row(store, "b")["status"] == "done"


def test_dest_on_an_unmounted_volume_is_not_ready(tmp_path: Path):
    assert dest_ready("/Volumes/NoSuchShare-mi-test/AI Engineer") is None
    assert not Path("/Volumes/NoSuchShare-mi-test").exists()
    assert dest_ready(str(tmp_path / "share" / "aie")) == tmp_path / "share" / "aie"
    assert dest_ready("") is None


def test_bulk_api_lists_progress_with_processing_state(tmp_path: Path):
    from media_pipeline.api import create_app
    from tests.test_api import DummyWorker

    config = _config(tmp_path)
    store = _job(config, title="AI Engineer")
    clock = Clock()
    _runner(config, store, clock, FakeYouTube(), max_attempts=3).run(max_videos=1)
    tasks = TaskStore(config.paths.db)
    tasks.insert(Task(id="t1", url="https://www.youtube.com/watch?v=a", asr_model="fun-asr", video_id="a",
                      status=TaskStatus.completed))
    client = TestClient(create_app(config, store=tasks, worker=DummyWorker()))

    jobs = client.get("/v1/bulk").json()["jobs"]
    assert [job["name"] for job in jobs] == ["aie"] and jobs[0]["title"] == "AI Engineer"
    detail = client.get("/v1/bulk/aie").json()
    assert detail["summary"]["counts"] == {"done": 1, "pending": 2}
    videos = {video["id"]: video for video in detail["videos"]}
    assert videos["a"]["task"] == {"id": "t1", "status": "completed"} and videos["b"]["task"] is None
    assert [event["kind"] for event in detail["events"]][:1] == ["stop"]

    csv_text = client.get("/v1/bulk/aie/csv").text
    assert "Talk a" in csv_text and csv_text.startswith("﻿#,id,title")
    assert client.post("/v1/bulk/aie/pause").json()["control"] == "pause"
    assert client.post("/v1/bulk/aie/resume").json()["control"] == ""
    assert client.get("/v1/bulk/nope").status_code == 404
    assert client.get("/v1/bulk/..%2Fetc").status_code == 404
    page = client.get("/bulk")
    assert page.status_code == 200 and "批量下载" in page.text


def test_pipeline_uses_a_bulk_download_instead_of_downloading(tmp_path: Path, monkeypatch):
    from media_pipeline.artifacts import ArtifactStore
    from media_pipeline.pipeline import Pipeline
    from tests.test_pipeline import FakeVision, FakeVisual
    from media_pipeline.diarization import NullDiarizationProvider

    config = _config(tmp_path)
    store = _job(config, ids=("vidBulk0001",))
    _runner(config, store, Clock(), FakeYouTube()).run()
    monkeypatch.setattr("media_pipeline.pipeline.download_video",
                        lambda *args, **kwargs: pytest.fail("must not download again"))
    tasks = TaskStore(config.paths.db)
    pipeline = Pipeline(config, tasks, diarization=NullDiarizationProvider(), visual=FakeVisual(), vision=FakeVision())
    task = tasks.insert(Task(id="t", url="https://www.youtube.com/watch?v=vidBulk0001", asr_model="fun-asr",
                             video_id="vidBulk0001"))
    metadata = VideoMetadata(url=task.url, title="t", platform="YouTube", author="a", video_id="vidBulk0001",
                             duration=1.0, published="", description="", thumbnail_url="", asr_model="fun-asr")
    path = pipeline._download(task, metadata, ArtifactStore(config.paths.artifacts, "vidBulk0001"))
    assert path == store.db.parent / "media" / "vidBulk0001.mp4"


def test_yt_dlp_gets_a_js_runtime_when_one_is_installed(monkeypatch):
    import media_pipeline.media as media

    media.js_runtimes.cache_clear()
    monkeypatch.setattr(media.shutil, "which", lambda name: "/x/bin/node" if name == "node" else None)
    try:
        assert media._base_ydl_opts(AppConfig())["js_runtimes"] == {"node": {"path": "/x/bin/node"}}
        media.js_runtimes.cache_clear()
        monkeypatch.setattr(media.shutil, "which", lambda name: None)
        monkeypatch.setattr(media.os, "access", lambda path, mode: False)
        assert "js_runtimes" not in media._base_ydl_opts(AppConfig())
    finally:
        media.js_runtimes.cache_clear()
