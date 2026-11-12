import time
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
        time.sleep(0)  # let the other stage's threads run


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


def _runner(config, store, clock, youtube, probe=lambda: None, free=lambda path: 500.0, transcriber=None,
            submit_summary=None, task_status=None, **limits):
    return BulkRunner(config, store, limits=Limits(pause_between=(0, 0), **limits), downloader=youtube,
                      prober=probe, transcriber=transcriber, submit_summary=submit_summary, task_status=task_status,
                      clock=clock, sleep=clock.sleep, free_gb=free)


class FakeASR:
    """Stands in for Fun-ASR; `fail` maps a video id to errors raised on its first calls ("*" = every video)."""

    def __init__(self, fail=None, on_call=None):
        self.fail = {vid: list(errors) for vid, errors in (fail or {}).items()}
        self.on_call = on_call
        self.calls: list[str] = []

    def __call__(self, path, row, model):
        assert model == "fun-asr" and path.is_file()
        self.calls.append(row["id"])
        if self.on_call:
            self.on_call(path, row)
        errors = self.fail.get(row["id"]) or self.fail.get("*")
        if errors:
            raise RuntimeError(errors[0] if "*" in self.fail else errors.pop(0))
        return {"segments": 12, "speakers": 2, "language": "en", "seconds": 1790.0}


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


def test_find_media_file_ignores_pieces_of_a_split_download(tmp_path: Path):
    from media_pipeline.media import find_media_file

    # The video half arrived, the audio half hit HTTP 567: that piece is not a playable video.
    (tmp_path / "BV1d7W26SEkb_p1.f100023.mp4").write_bytes(b"video only")
    (tmp_path / "BV1d7W26SEkb_p1.f30280.m4a.part").write_bytes(b"partial")
    (tmp_path / "BV1d7W26SEkb_p1.mp4.ytdl").write_bytes(b"state")
    assert find_media_file(tmp_path, "BV1d7W26SEkb_p1") is None
    (tmp_path / "BV1d7W26SEkb_p1.flv").write_bytes(b"merged")
    assert find_media_file(tmp_path, "BV1d7W26SEkb_p1") == tmp_path / "BV1d7W26SEkb_p1.flv"
    (tmp_path / "BV1d7W26SEkb_p1.mp4").write_bytes(b"merged")
    assert find_media_file(tmp_path, "BV1d7W26SEkb_p1") == tmp_path / "BV1d7W26SEkb_p1.mp4"



def test_transcription_runs_alongside_downloads_and_records_each_stage(tmp_path: Path):
    config = _config(tmp_path)
    store = _job(config, asr_model="fun-asr")
    clock, asr = Clock(), FakeASR()
    assert _runner(config, store, clock, FakeYouTube(), transcriber=asr).run() == "finished"
    assert sorted(asr.calls) == ["a", "b", "c"]
    for vid in "abc":
        row = _row(store, vid)
        assert row["status"] == "done" and row["asr_status"] == "done"
        assert row["segments"] == 12 and row["speakers"] == 2 and row["language"] == "en"
        assert row["finished_at"] <= row["asr_started_at"] <= row["asr_finished_at"]  # download, then ASR
    summary = store.summary(now=clock.t)
    assert summary["asr_counts"] == {"done": 3} and summary["asr_hours"] == 1.5
    assert summary["asr_cost_usd"] == round(3 * 1800 * 0.000035, 2)
    assert store.work_left() == (0, 0, 0)


def test_transcription_failures_are_retried_then_paused_after_a_streak(tmp_path: Path):
    config = _config(tmp_path)
    store = _job(config, ids=tuple(f"v{i}" for i in range(5)), asr_model="fun-asr")
    clock, asr = Clock(), FakeASR(fail={"*": ["DashScope task failed: Throttling.RateQuota"]})
    assert _runner(config, store, clock, FakeYouTube(), transcriber=asr, asr_fail_streak=5).run() == "finished"
    assert len(asr.calls) == 15  # every video, three attempts each
    rows = store.rows()
    assert all(row["asr_status"] == "failed" and row["asr_attempts"] == 3 for row in rows)
    assert "Throttling" in rows[0]["asr_error"]
    assert any("连续 5 个视频转写失败" in event["detail"] for event in _events(store, "asr-pause"))
    assert store.work_left() == (0, 0, 0)


def test_a_video_moves_to_the_nas_only_after_it_is_transcribed(tmp_path: Path):
    config = _config(tmp_path)
    nas = tmp_path / "nas"
    store = _job(config, ids=("a", "b"), asr_model="fun-asr", dest=str(nas))
    reads = []
    asr = FakeASR(on_call=lambda path, row: reads.append((row["id"], path.parent.name, (nas / path.name).exists())))
    _runner(config, store, Clock(), FakeYouTube(), transcriber=asr).run()
    assert sorted(reads) == [("a", "media", False), ("b", "media", False)]  # read locally, not yet copied
    assert {row["location"] for row in store.rows()} == {"nas"}
    assert sorted(p.name for p in nas.iterdir()) == ["a.mp4", "b.mp4"]


def test_bulk_transcriber_saves_where_the_pipeline_looks(tmp_path: Path, monkeypatch):
    from media_pipeline.artifacts import ArtifactStore
    from media_pipeline.models import Transcript, TranscriptSegment

    config = _config(tmp_path)
    store = _job(config, ids=("talk1",), asr_model="fun-asr", title="AI Engineer")
    calls = []

    class Provider:
        def bind_config(self, cfg):
            calls.append("bind")

        def transcribe(self, path, options):
            calls.append(options.context)
            return Transcript(language="en", provider="dashscope", model="fun-asr", segments=[
                TranscriptSegment(0.0, 3.0, "Welcome to the talk.", speaker="SPEAKER_00"),
                TranscriptSegment(3.0, 6.0, "Thanks for having me.", speaker="SPEAKER_01"),
                TranscriptSegment(6.0, 9.0, "Let us start with evals.", speaker="SPEAKER_01"),
            ])

    monkeypatch.setattr("media_pipeline.asr.registry.require_provider", lambda model: Provider())
    runner = BulkRunner(config, store, limits=Limits(pause_between=(0, 0)), downloader=FakeYouTube(),
                        prober=lambda: None, clock=Clock(), sleep=Clock().sleep, free_gb=lambda path: 500.0)
    assert runner.run() == "finished"
    row = _row(store, "talk1")
    assert row["asr_status"] == "done" and row["segments"] == 3 and row["speakers"] == 2
    assert calls == ["bind", "Talk talk1\nAI Engineer"]
    saved = ArtifactStore(config.paths.artifacts, "talk1").load_transcript("fun-asr")
    assert saved is not None and len(saved.segments) == 3

    # The pipeline reuses it: no audio, no provider call.
    from media_pipeline.diarization import NullDiarizationProvider
    from media_pipeline.pipeline import Pipeline
    from tests.test_pipeline import FakeVision, FakeVisual

    tasks = TaskStore(config.paths.db)
    pipeline = Pipeline(config, tasks, diarization=NullDiarizationProvider(), visual=FakeVisual(), vision=FakeVision())
    task = tasks.insert(Task(id="t", url="https://www.youtube.com/watch?v=talk1", asr_model="fun-asr", video_id="talk1"))
    metadata = VideoMetadata(url=task.url, title="t", platform="YouTube", author="a", video_id="talk1",
                             duration=9.0, published="", description="", thumbnail_url="", asr_model="fun-asr")
    reused = pipeline._transcribe(task, metadata, ArtifactStore(config.paths.artifacts, "talk1"), None)
    assert len(reused.segments) == 3 and len(calls) == 2

    from media_pipeline.api import create_app
    from tests.test_api import DummyWorker

    client = TestClient(create_app(config, store=tasks, worker=DummyWorker()))
    text = client.get("/v1/bulk/aie/transcript/talk1").text
    assert "[00:00:00] SPEAKER_00\nWelcome to the talk." in text
    assert "[00:00:03] SPEAKER_01\nThanks for having me. Let us start with evals." in text
    assert client.get("/v1/bulk/aie/transcript/unknown").status_code == 404
    summary = client.get("/v1/bulk/aie").json()["summary"]
    assert summary["asr_counts"] == {"done": 1} and summary["asr_model"] == "fun-asr"


JSON3 = {"events": [
    {"tStartMs": 0, "dDurationMs": 4000, "segs": [{"utf8": "Hey everyone, welcome"}]},
    {"tStartMs": 3500, "dDurationMs": 40, "aAppend": 1, "segs": [{"utf8": "\n"}]},
    {"tStartMs": 4000, "dDurationMs": 5000, "segs": [{"utf8": "to the talk about evals."}]},
    {"tStartMs": 9500, "dDurationMs": 1000, "segs": [{"utf8": "[Music]"}]},
    {"tStartMs": 11000, "dDurationMs": 6000, "segs": [{"utf8": "First, "}, {"utf8": "what is an eval?"}]},
]}


class FakeYdl:
    def __init__(self, payload):
        self.payload = payload
        self.opened = []

    def urlopen(self, url):
        import io
        import json as _json

        self.opened.append(url)
        return io.BytesIO(_json.dumps(self.payload).encode())


def _info(vid, captions=True, chapters=True):
    return {
        "id": vid, "title": f"Talk {vid}", "uploader": "AI Engineer", "duration": 1800, "upload_date": "20261101",
        "description": "desc", "webpage_url": f"https://www.youtube.com/watch?v={vid}", "language": "en-US",
        "extractor_key": "Youtube",
        "automatic_captions": {"en-orig": [{"ext": "vtt", "url": "u.vtt"}, {"ext": "json3", "url": f"https://cap/{vid}"}]}
        if captions else {},
        "chapters": [{"start_time": 0, "end_time": 60, "title": "Intro"},
                     {"start_time": 60, "end_time": 1800, "title": "What is an eval"}] if chapters else None,
    }


def test_captions_become_a_one_speaker_transcript_and_tracks_prefer_uploaded_ones():
    from media_pipeline.youtube_captions import parse_json3, pick_track

    transcript = parse_json3(JSON3, "en")
    assert [s.text for s in transcript.segments] == ["Hey everyone, welcome to the talk about evals.",
                                                     "First, what is an eval?"]
    assert {s.speaker for s in transcript.segments} == {"SPEAKER_00"} and transcript.model == "youtube-captions"
    assert transcript.segments[0].start == 0.0 and transcript.segments[1].start == 11.0
    info = _info("x")
    assert pick_track(info) == ("en", "auto", "https://cap/x")
    info["subtitles"] = {"en": [{"ext": "json3", "url": "https://manual/x"}], "fr": [{"ext": "json3", "url": "f"}]}
    assert pick_track(info) == ("en", "manual", "https://manual/x")
    assert pick_track({"language": "en", "automatic_captions": {}}) is None


def test_captions_first_skips_paid_asr_and_backfills_extras(tmp_path: Path):
    from media_pipeline.artifacts import ArtifactStore

    config = _config(tmp_path)
    store = _job(config, ids=("cap1", "nocap"), asr_model="fun-asr", youtube_extras=True)
    store.execute("UPDATE videos SET status='done', path=? WHERE id='cap1'",
                  (str(store.db.parent / "media" / "cap1.mp4"),))  # downloaded before extras existed
    (store.db.parent / "media").mkdir(parents=True, exist_ok=True)
    (store.db.parent / "media" / "cap1.mp4").write_bytes(b"v")
    fetched = []
    holder = {}

    def extras(url, vid, captions=True):  # what a metadata-only extraction would save
        fetched.append(vid)
        info = _info(vid, captions=vid != "nocap", chapters=vid == "cap1")
        holder["runner"]._save_extras(FakeYdl(JSON3), info, vid, captions=captions)

    class YouTubeWithExtras(FakeYouTube):
        def __call__(self, url, video_id, target_dir, progress):
            result = super().__call__(url, video_id, target_dir, progress)
            extras(url, video_id)  # the real downloader saves extras from the same extraction
            return result

    asr = FakeASR()
    runner = _runner(config, store, Clock(), YouTubeWithExtras(), transcriber=asr)
    runner.fetch_extras = extras
    holder["runner"] = runner
    assert runner.run() == "finished"
    cap1, nocap = _row(store, "cap1"), _row(store, "nocap")
    assert sorted(fetched) == ["cap1", "nocap"]  # the backfill, and the new download's own extraction
    assert cap1["captions_status"] == "done" and cap1["transcript_source"] == "youtube" and cap1["chapters"] == 2
    assert cap1["asr_status"] == "done" and cap1["segments"] == 2 and cap1["speakers"] == 1
    assert nocap["captions_status"] == "none" and nocap["transcript_source"] == "fun-asr"
    assert asr.calls == ["nocap"]  # only the video without captions cost ASR money
    artifacts = ArtifactStore(config.paths.artifacts, "cap1")
    assert artifacts.load_transcript("youtube-captions").segments[0].text.startswith("Hey everyone")
    assert artifacts.load_metadata().title == "Talk cap1" and artifacts.load_metadata().published == "2026-11-01"
    assert artifacts.read_json(artifacts.root / "youtube.json")["chapters"][1]["title"] == "What is an eval"
    summary = store.summary()
    assert summary["transcript_sources"] == {"youtube": 1, "fun-asr": 1}
    assert summary["asr_cost_usd"] == round(1800 * 0.000035, 2)  # only the Fun-ASR one is paid
    assert summary["captions_counts"] == {"done": 1, "none": 1} and summary["with_chapters"] == 1


def test_a_captions_rate_limit_cools_captions_down_but_never_stops_downloads(tmp_path: Path):
    config = _config(tmp_path)
    store = _job(config, ids=("v1", "v2", "v3"), youtube_extras=True)
    store.execute("UPDATE videos SET status='done' WHERE id='v1'")
    clock = Clock()
    calls = []

    def extras(url, vid, captions=True):
        calls.append((vid, clock.t))
        if len(calls) == 1:
            raise RuntimeError("HTTP Error 429: Too Many Requests")
        holder["runner"]._save_extras(FakeYdl(JSON3), _info(vid), vid, captions=captions)

    holder = {}
    youtube = FakeYouTube()
    runner = _runner(config, store, clock, youtube)
    runner.fetch_extras = extras
    holder["runner"] = runner
    assert runner.run() == "finished"
    assert youtube.calls == ["v2", "v3"]  # downloads went on during the cooldown
    v1 = [at for vid, at in calls if vid == "v1"]
    assert calls[0][0] == "v1" and len(v1) == 2 and v1[1] - v1[0] >= 30 * 60  # retried after the cooldown
    assert {vid for vid, _at in calls} == {"v1", "v2", "v3"}  # the new downloads got their captions too
    assert _row(store, "v1")["captions_status"] == "done"
    assert _events(store, "pause") == [] and len(_events(store, "captions-pause")) == 1


def test_captions_that_never_come_fall_back_to_asr_after_six_hours(tmp_path: Path):
    config = _config(tmp_path)
    store = _job(config, ids=("v1",), asr_model="fun-asr", youtube_extras=True)
    clock = Clock()
    downloaded_at = clock.t
    store.execute("UPDATE videos SET status='done', finished_at=?, path=? WHERE id='v1'",
                  (downloaded_at, str(store.db.parent / "media" / "v1.mp4")))
    (store.db.parent / "media").mkdir(parents=True, exist_ok=True)
    (store.db.parent / "media" / "v1.mp4").write_bytes(b"v")
    asked = []

    def extras(url, vid, captions=True):
        asked.append(captions)
        if captions:
            raise RuntimeError("HTTP Error 429: Too Many Requests")
        holder["runner"]._save_extras(FakeYdl(JSON3), _info(vid), vid, captions=False)

    holder = {}
    asr = FakeASR(on_call=lambda path, row: asked.append(("asr", clock.t - downloaded_at)))
    runner = _runner(config, store, clock, FakeYouTube(), transcriber=asr)
    runner.fetch_extras = extras
    holder["runner"] = runner
    assert runner.run() == "finished"
    row = _row(store, "v1")
    assert asr.calls == ["v1"] and row["transcript_source"] == "fun-asr"
    assert next(item for item in asked if isinstance(item, tuple))[1] >= 6 * 3600  # waited for captions first
    assert row["captions_status"] == "skipped" and row["chapters"] == 2  # later: metadata and chapters only
    assert asked[-1] is False


def test_a_bot_check_seen_while_backfilling_pauses_downloads_and_captions_wait_for_it(tmp_path: Path):
    config = _config(tmp_path)
    store = _job(config, ids=("old", "new1"), youtube_extras=True)
    store.execute("UPDATE videos SET status='done', finished_at=? WHERE id='old'", (1.0,))
    clock = Clock()
    seen = []

    def extras(url, vid, captions=True):
        seen.append((vid, clock.t))
        if len(seen) == 1:
            raise RuntimeError("[youtube] old: Sign in to confirm you’re not a bot.")
        holder["runner"]._save_extras(FakeYdl(JSON3), _info(vid), vid, captions=captions)

    holder = {}
    youtube = FakeYouTube()
    started = clock.t
    runner = _runner(config, store, clock, youtube)
    runner.fetch_extras = extras
    holder["runner"] = runner
    runner.run()
    pauses = _events(store, "pause")
    assert len(pauses) == 1 and "补抓元数据" in pauses[0]["detail"]
    assert _row(store, "old")["captions_status"] == "done" and seen[1][1] - seen[0][1] >= 15 * 60
    assert all(at - started >= 0 for _vid, at in seen)



def test_finished_files_move_to_the_nas_while_youtube_has_the_job_paused(tmp_path: Path):
    config = _config(tmp_path)
    nas = tmp_path / "nas"
    store = _job(config, ids=("a", "b"), dest=str(nas))
    media = store.db.parent / "media"
    media.mkdir(parents=True, exist_ok=True)
    (media / "a.mp4").write_bytes(b"x" * 100)
    store.execute("UPDATE videos SET status='done', path=?, location='mac', bytes=100 WHERE id='a'", (str(media / "a.mp4"),))
    clock = Clock()
    store.set_meta("pause_until", clock.t + 3600)  # YouTube said "not a bot" an hour ago
    at_probe = []
    runner = _runner(config, store, clock, FakeYouTube(), probe=lambda: at_probe.append((nas / "a.mp4").exists()))
    runner.run()
    assert at_probe == [True]  # copied during the pause, before downloading resumed
    assert _row(store, "a")["location"] == "nas" and not (media / "a.mp4").exists()



def test_pacing_can_be_changed_while_a_job_runs(tmp_path: Path):
    config = _config(tmp_path)
    store = _job(config, ids=("a", "b"))
    clock = Clock()
    naps = []
    runner = _runner(config, store, clock, FakeYouTube())
    runner.sleep = lambda seconds: (naps.append(seconds), clock.sleep(seconds))
    store.set_meta("pause_between", [90, 45])
    runner.run()
    assert any(45 <= nap <= 90 for nap in naps)
