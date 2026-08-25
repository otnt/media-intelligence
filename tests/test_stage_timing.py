from datetime import datetime, timezone

from media_pipeline.artifacts import ArtifactStore
from media_pipeline.stage_timing import (
    begin_stage,
    clear_invalidated_timings,
    finish_stage,
    merge_stage_timings,
    public_timings,
    stage_keys_invalidated_by,
)


def test_begin_and_finish_record_start_and_duration():
    extra: dict = {}
    start = datetime(2026, 8, 24, 21, 27, 25, tzinfo=timezone.utc)
    begin_stage(extra, "transcribing", now=start, monotonic=10.0)
    entry = finish_stage(extra, "transcribing", succeeded=True, now=start, monotonic=13.4)
    assert entry["started_at"].startswith("2026-08-24T21:27:25")
    assert entry["status"] == "succeeded"
    assert entry["duration_sec"] == 3.4
    assert "_mono" not in public_timings(extra["stage_timings"])["transcribing"]


def test_failed_stage_has_start_but_no_duration():
    extra: dict = {}
    begin_stage(extra, "fetching_metadata", monotonic=1.0)
    entry = finish_stage(extra, "fetching_metadata", succeeded=False, monotonic=4.0)
    assert entry["status"] == "failed"
    assert "started_at" in entry
    assert "duration_sec" not in entry


def test_invalidation_matches_artifact_scope():
    assert "transcribing" in stage_keys_invalidated_by("transcribing")
    assert "detecting_scenes" not in stage_keys_invalidated_by("transcribing")
    assert "detecting_scenes" in stage_keys_invalidated_by("detecting_scenes")
    assert "writing_outputs" in stage_keys_invalidated_by("deduplicating_frames")
    assert "downloading" not in stage_keys_invalidated_by("all")
    assert "filtering_frames" in stage_keys_invalidated_by("filtering_frames")
    assert "filtering_frames" in stage_keys_invalidated_by("deduplicating_frames")
    assert "detecting_scenes" not in stage_keys_invalidated_by("filtering_frames")
    extra = {
        "stage_timings": {
            "transcribing": {"status": "succeeded", "duration_sec": 9},
            "detecting_scenes": {"status": "succeeded", "duration_sec": 2},
        }
    }
    clear_invalidated_timings(extra, "detecting_scenes")
    assert "transcribing" in extra["stage_timings"]
    assert "detecting_scenes" not in extra["stage_timings"]


def test_artifact_store_persists_and_clears_timings(tmp_path):
    store = ArtifactStore(tmp_path, "BVtest")
    store.save_stage_timing(
        "transcribing",
        {"started_at": "2026-08-24T21:00:00-07:00", "status": "succeeded", "duration_sec": 12.5, "_mono": 99},
    )
    loaded = store.load_stage_timings()
    assert loaded["transcribing"]["duration_sec"] == 12.5
    assert "_mono" not in loaded["transcribing"]
    store.clear_invalidated_timings("transcribing")
    assert "transcribing" not in store.load_stage_timings()


def test_merge_prefers_current_task_overlay():
    merged = merge_stage_timings(
        {"transcribing": {"status": "succeeded", "duration_sec": 10, "started_at": "old"}},
        {"transcribing": {"status": "running", "started_at": "new"}},
    )
    assert merged["transcribing"]["status"] == "running"
    assert merged["transcribing"]["started_at"] == "new"
