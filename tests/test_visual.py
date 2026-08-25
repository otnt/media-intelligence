from pathlib import Path

from PIL import Image

from media_pipeline.models import NamedSegment, frame_filename
from media_pipeline.visual.align import align_keyframes, render_visual_timeline
from media_pipeline.visual.dedup import apply_dedup
from media_pipeline.visual.models import CandidateFrame, Keyframe
from media_pipeline.visual.ocr import text_change_ratio
from media_pipeline.visual.timestamps import (
    SOURCE_CHANGE,
    SOURCE_PERIODIC,
    SOURCE_SCENE,
    candidates_from_groups,
    merge_candidate_times,
    periodic_timestamps,
)


def test_frame_filename_is_inspectable():
    assert frame_filename(745.3) == "00-12-25.300.jpg"
    assert frame_filename(0) == "00-00-00.000.jpg"


def test_periodic_sampling_covers_the_timeline():
    stamps = periodic_timestamps(60.0, 12.0)
    assert stamps[0] == 0.0
    assert 12.0 in stamps
    assert stamps[-1] >= 48.0
    assert all(stamp < 60.0 for stamp in stamps)


def test_candidate_sources_merge_nearby_timestamps():
    merged = merge_candidate_times(
        {
            SOURCE_SCENE: [12.0],
            SOURCE_PERIODIC: [12.2],
            SOURCE_CHANGE: [40.0],
        }
    )
    by_time = {stamp: sources for stamp, sources in merged}
    assert SOURCE_SCENE in by_time[12.0]
    assert SOURCE_PERIODIC in by_time[12.0]
    assert SOURCE_CHANGE in by_time[40.0]
    frames = candidates_from_groups({SOURCE_PERIODIC: [5.0], SOURCE_SCENE: [5.1]})
    assert len(frames) == 1
    assert frames[0].path.endswith("00-00-05.000.jpg")


def test_dedup_drops_near_duplicates_but_honors_keep_override(tmp_path: Path):
    first = tmp_path / "candidate_frames" / "00-00-01.000.jpg"
    second = tmp_path / "candidate_frames" / "00-00-02.000.jpg"
    first.parent.mkdir(parents=True)
    _solid(first, (20, 20, 20))
    _solid(second, (20, 20, 20))
    candidates = [
        CandidateFrame(1.0, "candidate_frames/00-00-01.000.jpg", [SOURCE_PERIODIC]),
        CandidateFrame(2.0, "candidate_frames/00-00-02.000.jpg", [SOURCE_PERIODIC]),
    ]
    auto = apply_dedup(candidates, tmp_path, similarity_threshold=0.9, overrides={})
    assert auto[0][1].kept is True
    assert auto[1][1].kept is False
    forced = apply_dedup(
        candidates,
        tmp_path,
        similarity_threshold=0.9,
        overrides={"00-00-02.000.jpg": "keep"},
    )
    assert forced[1][1].kept is True
    assert forced[1][1].decision == "manual"


def test_keyframe_gets_transcript_window_not_nearest_only():
    frame = Keyframe(timestamp=24.0, image_path="keyframes/00-00-24.000.jpg", sources=[SOURCE_SCENE])
    segments = [
        NamedSegment(10.0, 12.0, "s0", "Speaker 1", "Intro before the slide."),
        NamedSegment(20.0, 22.0, "s0", "Speaker 1", "Now look at this table."),
        NamedSegment(30.0, 33.0, "s0", "Speaker 1", "As you can see here."),
        NamedSegment(80.0, 85.0, "s0", "Speaker 1", "Unrelated later talk."),
    ]
    timeline = align_keyframes([frame], segments, before_sec=10, after_sec=20)
    texts = [item.text for item in timeline[0].transcript_context.segments]
    assert "Now look at this table." in texts
    assert "As you can see here." in texts
    assert "Intro before the slide." not in texts
    assert "Unrelated later talk." not in texts
    markdown = render_visual_timeline(timeline, "BVtest")
    assert "![[attachments/BVtest/00-00-24.000.jpg]]" in markdown
    assert "Nearby transcript:" in markdown


def test_ocr_change_ratio_detects_slide_text_swap():
    assert text_change_ratio("Training Pipeline", "Training Pipeline") == 0.0
    assert text_change_ratio("Training Pipeline", "Evaluation Results") > 0.5


def _solid(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (64, 36), color).save(path, format="JPEG", quality=90)
