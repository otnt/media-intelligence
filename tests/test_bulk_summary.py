from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from media_pipeline.bulk import in_window, next_window_start
from media_pipeline.models import Task, TaskStatus, VideoMetadata
from media_pipeline.store import TaskStore
from tests.test_bulk import Clock, FakeYouTube, _config, _job, _row, _runner

BEIJING = timezone(timedelta(hours=8))
WINDOW = "22:00-08:00@+08:00"


def _at(hour: int, minute: int = 0, day: int = 12) -> float:
    return datetime(2026, 11, day, hour, minute, tzinfo=BEIJING).timestamp()


def test_the_half_price_window_is_beijing_night_whatever_the_mac_clock_says():
    assert in_window(WINDOW, _at(23)) and in_window(WINDOW, _at(7, 59)) and in_window(WINDOW, _at(22))
    assert not in_window(WINDOW, _at(8)) and not in_window(WINDOW, _at(16, 45))
    assert next_window_start(WINDOW, _at(16, 45)) == _at(22)
    assert next_window_start(WINDOW, _at(23)) == _at(23)
    assert in_window("", _at(12)) and next_window_start("", _at(12)) == _at(12)


def test_summaries_start_in_the_window_a_few_at_a_time_and_retry_a_failure(tmp_path: Path):
    config = _config(tmp_path)
    ids = tuple(f"v{i}" for i in range(6))
    store = _job(config, ids=ids, summary_model="deepseek-v4.1-flash", summary_window=WINDOW)
    store.execute("UPDATE videos SET status='done', asr_status='done', transcript_source='youtube'")
    clock = Clock()
    clock.t = _at(21, 50)
    tasks: dict[str, list] = {}
    submitted: list[tuple[str, float, int]] = []

    def submit(job, vid):
        assert job == "aie"
        running = sum(1 for state in tasks.values() if state[0] in ("queued", "running"))
        task_id = f"t-{vid}-{sum(1 for item in submitted if item[0] == vid)}"
        submitted.append((vid, clock.t, running))
        tasks[task_id] = ["queued", 0]
        return task_id

    def status(task_id):
        state = tasks[task_id]
        state[1] += 1
        if state[1] == 1:
            state[0] = "running"
        elif state[1] >= 3:
            state[0] = "failed" if task_id == "t-v3-0" else "done"
        return state[0], "summarizing: Flash timed out" if state[0] == "failed" else ""

    runner = _runner(config, store, clock, FakeYouTube(), submit_summary=submit, task_status=status)
    assert runner.run() == "finished"
    assert min(at for _vid, at, _n in submitted) >= _at(22)  # nothing at full price
    assert max(n for _vid, _at_, n in submitted) < 4  # never more than 4 in flight
    assert [vid for vid, _at_, _n in submitted].count("v3") == 2  # the failure was retried once
    assert all(_row(store, vid)["summary_status"] == "done" for vid in ids)
    summary = store.summary(now=clock.t)
    assert summary["summary_counts"] == {"done": 6} and summary["summary_cost_usd"] == 0.04
    assert store.work_left() == (0, 0, 0)


def test_bulk_videos_become_collection_tasks_kept_out_of_the_main_library(tmp_path: Path):
    from media_pipeline.api import create_app
    from tests.test_api import DummyWorker

    config = _config(tmp_path)
    store = _job(config, ids=("cap1", "fun1", "raw1"), title="AI Engineer", asr_model="fun-asr",
                 summary_model="deepseek-v4.1-flash")
    store.execute("UPDATE videos SET status='done', asr_status='done', transcript_source='youtube' WHERE id='cap1'")
    store.execute("UPDATE videos SET status='done', asr_status='done', transcript_source='fun-asr' WHERE id='fun1'")
    tasks = TaskStore(config.paths.db)
    tasks.insert(Task(id="mine", url="https://www.youtube.com/watch?v=abcdefghijk", asr_model="fun-asr",
                      video_id="abcdefghijk", status=TaskStatus.completed))
    worker = DummyWorker()
    client = TestClient(create_app(config, store=tasks, worker=worker))

    first = client.post("/v1/bulk/aie/videos/cap1/process")
    assert first.status_code == 202
    created = tasks.get(first.json()["id"])
    assert created.asr_model == "youtube-captions" and created.video_id == "cap1"
    assert created.extra["collection"] == "AI Engineer" and created.extra["summary_model"] == "deepseek-v4.1-flash"
    assert created.extra["extract_keyframes"] is False and created.extra["bulk_job"] == "aie"
    assert tasks.get(client.post("/v1/bulk/aie/videos/fun1/process").json()["id"]).asr_model == "fun-asr"
    assert client.post("/v1/bulk/aie/videos/raw1/process").status_code == 409
    assert client.post("/v1/bulk/aie/videos/nope/process").status_code == 404
    assert len(worker.submitted) == 2

    assert [task["id"] for task in client.get("/v1/tasks").json()["tasks"]] == ["mine"]
    collection = client.get("/v1/tasks", params={"collection": "AI Engineer"}).json()["tasks"]
    assert sorted(task["video_id"] for task in collection) == ["cap1", "fun1"]
    assert client.get("/v1/collections").json() == {"collections": [{"name": "AI Engineer", "count": 2}]}


def test_collection_notes_get_their_own_folder(tmp_path: Path):
    from media_pipeline.notes import NoteWriter

    writer = NoteWriter(tmp_path / "Transcripts")
    metadata = VideoMetadata(url="https://www.youtube.com/watch?v=a", title="Talk A", platform="YouTube", author="x",
                             video_id="a", duration=60.0, published="", description="", thumbnail_url="",
                             asr_model="youtube-captions")
    grouped = writer.create_or_open(metadata, Task(id="t", url=metadata.url, asr_model="youtube-captions",
                                                   video_id="a", extra={"collection": "AI Engineer"}))
    assert grouped == tmp_path / "Transcripts" / "AI Engineer" / "Talk A.md" and grouped.exists()
    plain = writer.create_or_open(metadata, Task(id="u", url=metadata.url, asr_model="fun-asr", video_id="a"))
    assert plain == tmp_path / "Transcripts" / "Talk A.md"


def test_chapters_reach_the_summary_prompt_and_tasks_pick_their_summary_model(tmp_path: Path, monkeypatch):
    from media_pipeline.artifacts import ArtifactStore
    from media_pipeline.summary import _load_chapters, build_model_prompt

    artifacts = ArtifactStore(tmp_path / "artifacts", "v")
    artifacts.write_json(artifacts.root / "youtube.json", {"chapters": [
        {"start": 0, "end": 60, "title": "Intro"}, {"start": 75, "end": 300, "title": "Why evals"}]})
    prompt = build_model_prompt("Summarize.", None, "the transcript", chapters=_load_chapters(artifacts))
    assert "- [00:00:00] Intro" in prompt and "- [00:01:15] Why evals" in prompt
    assert prompt.index("Chapters") < prompt.index("Transcript:")
    assert "Chapters" not in build_model_prompt("Summarize.", None, "the transcript")

    from media_pipeline.diarization import NullDiarizationProvider
    from media_pipeline.pipeline import Pipeline
    from tests.test_pipeline import FakeVision, FakeVisual
    from tests.test_pipeline import _config as pipeline_config

    config = pipeline_config(tmp_path / "p")
    seen = []
    monkeypatch.setattr("media_pipeline.summary_llm.resolve_summary_backends",
                        lambda cfg, vision, selection=None, vision_8bit=None: seen.append(selection) or [])
    pipeline = Pipeline(config, TaskStore(config.paths.db), diarization=NullDiarizationProvider(),
                        visual=FakeVisual(), vision=FakeVision())
    task = Task(id="t", url="https://www.youtube.com/watch?v=v", asr_model="youtube-captions", video_id="v",
                extra={"summary_model": "deepseek-v4.1-flash"})
    metadata = VideoMetadata(url=task.url, title="t", platform="YouTube", author="a", video_id="v", duration=1.0,
                             published="", description="", thumbnail_url="", asr_model="youtube-captions")
    pipeline._summarize(task, metadata, ArtifactStore(config.paths.artifacts, "v"))
    task.extra.pop("summary_model")
    pipeline._summarize(task, metadata, ArtifactStore(config.paths.artifacts, "v"))
    assert seen == ["deepseek-v4.1-flash", None]
