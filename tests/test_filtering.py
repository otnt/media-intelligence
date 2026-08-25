from pathlib import Path

from media_pipeline.artifacts import ArtifactStore
from media_pipeline.models import NamedSegment, VideoMetadata
from media_pipeline.visual.filtering import (
    PROMPT_VERSION,
    apply_threshold_and_overrides,
    filter_keyframes,
    parse_verdict,
)
from media_pipeline.visual.models import FrameVerdict, Keyframe
from media_pipeline.visual.vlm import NullVisionProvider, VisionProvider, probe_vlm


class FakeVision(VisionProvider):
    name = "fake"
    model_id = "fake-vlm"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def judge(self, image_path: Path, prompt: str) -> str:
        self.calls.append(Path(image_path).name)
        if "talking" in Path(image_path).name:
            return '{"informative": false, "score": 0.1, "category": "talking_head", "reason": "face only", "caption": ""}'
        return '{"informative": true, "score": 0.9, "category": "slide", "reason": "on-screen text", "caption": "Book covers on a desk"}'


def _metadata() -> VideoMetadata:
    return VideoMetadata(
        url="https://www.bilibili.com/video/BVtest",
        title="Three books",
        platform="Bilibili",
        author="Host",
        video_id="BVtest",
        duration=60.0,
        published="",
        description="",
        thumbnail_url="",
        asr_model="qwen3-asr-1.7b",
    )


def test_parse_verdict_extracts_json_and_strips_thinking():
    text = '<think>ignore</think> {"informative": true, "score": 0.8, "category": "chart", "reason": "graph", "caption": "A bar chart"}'
    verdict = parse_verdict(text, filename="00-00-01.000.jpg", timestamp=1.0, provider=FakeVision())
    assert verdict.informative is True
    assert verdict.score == 0.8
    assert verdict.category == "chart"
    assert verdict.caption == "A bar chart"
    assert verdict.prompt_version == PROMPT_VERSION


def test_parse_failure_keeps_the_frame():
    verdict = parse_verdict("no json here", filename="00-00-01.000.jpg", timestamp=1.0, provider=FakeVision())
    assert verdict.kept is True
    assert verdict.decision == "parse_error"
    assert verdict.score == 1.0


def test_threshold_and_manual_overrides():
    raw = FrameVerdict(
        filename="00-00-01.000.jpg",
        timestamp=1.0,
        informative=True,
        score=0.3,
        category="talking_head",
        reason="face",
        caption="",
        kept=True,
        decision="auto",
        model="fake-vlm",
        prompt_version=PROMPT_VERSION,
    )
    dropped = apply_threshold_and_overrides(raw, 0.45, {})
    assert dropped.kept is False
    kept = apply_threshold_and_overrides(raw, 0.45, {"00-00-01.000.jpg": "keep"})
    assert kept.kept is True
    assert kept.decision == "manual"
    assert raw.kept is True


def test_filter_keyframes_caches_and_honors_overrides(tmp_path: Path):
    artifacts = ArtifactStore(tmp_path, "BVtest")
    (artifacts.root / "keyframes").mkdir(parents=True, exist_ok=True)
    talking = artifacts.root / "keyframes" / "talking-00-00-01.000.jpg"
    slide = artifacts.root / "keyframes" / "00-00-12.000.jpg"
    talking.write_bytes(b"jpg")
    slide.write_bytes(b"jpg")
    frames = [
        Keyframe(1.0, "keyframes/talking-00-00-01.000.jpg", sources=["periodic"]),
        Keyframe(12.0, "keyframes/00-00-12.000.jpg", sources=["scene_boundary"]),
    ]
    segments = [NamedSegment(10.0, 20.0, "s0", "Host", "Look at these three books.")]
    vision = FakeVision()
    settings = {"vlm_keep_threshold": 0.45, "context_before_sec": 10, "context_after_sec": 20}
    verdicts, selected = filter_keyframes(frames, artifacts, _metadata(), segments, settings, vision)
    assert [Path(item.image_path).name for item in selected] == ["00-00-12.000.jpg"]
    assert vision.calls == ["talking-00-00-01.000.jpg", "00-00-12.000.jpg"]
    assert verdicts[1].caption == "Book covers on a desk"
    artifacts.set_override("talking-00-00-01.000.jpg", "keep")
    again, selected_again = filter_keyframes(frames, artifacts, _metadata(), segments, settings, vision)
    assert vision.calls == ["talking-00-00-01.000.jpg", "00-00-12.000.jpg"]
    assert {Path(item.image_path).name for item in selected_again} == {
        "talking-00-00-01.000.jpg",
        "00-00-12.000.jpg",
    }
    assert again[0].decision == "manual"
    stricter, selected_strict = filter_keyframes(
        frames,
        artifacts,
        _metadata(),
        segments,
        {**settings, "vlm_keep_threshold": 0.95},
        vision,
    )
    assert vision.calls == ["talking-00-00-01.000.jpg", "00-00-12.000.jpg"]
    assert {Path(item.image_path).name for item in selected_strict} == {"talking-00-00-01.000.jpg"}
    assert stricter[1].kept is False
    assert stricter[1].score == 0.9


def test_null_provider_keeps_every_frame(tmp_path: Path):
    artifacts = ArtifactStore(tmp_path, "BVtest")
    frames = [Keyframe(1.0, "keyframes/00-00-01.000.jpg", sources=["periodic"])]
    verdicts, selected = filter_keyframes(
        frames,
        artifacts,
        _metadata(),
        [],
        {"vlm_keep_threshold": 0.99},
        NullVisionProvider(),
    )
    assert len(selected) == 1
    assert verdicts[0].decision == "unavailable"


def test_probe_vlm_handles_disabled_and_missing():
    from media_pipeline.config import AnalysisConfig
    from media_pipeline.visual.vlm import build_vision_provider

    ok, detail = probe_vlm(AnalysisConfig(enabled=False))
    assert ok is True
    assert detail == "disabled"
    ok, detail = probe_vlm(AnalysisConfig(enabled=True, model="missing-org/missing-vlm"))
    assert ok is False
    assert "mlx_vlm" in detail or "weights missing" in detail
    provider = build_vision_provider(AnalysisConfig(enabled=True, model="missing-org/missing-vlm"))
    assert isinstance(provider, NullVisionProvider)


def test_resolved_model_path_uses_lmstudio_weights(tmp_path: Path, monkeypatch):
    from media_pipeline.visual import vlm as vlm_mod

    monkeypatch.setattr(vlm_mod.Path, "home", classmethod(lambda cls: tmp_path))
    model = "mlx-community/Qwen3.8-27B-4bit"
    weights = tmp_path / ".lmstudio" / "models" / "mlx-community" / "Qwen3.8-27B-4bit"
    weights.mkdir(parents=True)
    (weights / "config.json").write_text("{}", encoding="utf-8")
    (weights / "model.safetensors").write_bytes(b"weights")
    assert vlm_mod._local_model_available(model)
    assert vlm_mod._resolved_model_path(model) == str(weights)


def test_invalidate_from_preserves_analysis_on_filter_rerun(tmp_path: Path):
    artifacts = ArtifactStore(tmp_path, "BVtest")
    artifacts.save_frame_analysis(
        [
            FrameVerdict(
                filename="00-00-01.000.jpg",
                timestamp=1.0,
                informative=True,
                score=0.9,
                category="slide",
                reason="text",
                caption="A slide",
                kept=True,
                model="fake-vlm",
                prompt_version=PROMPT_VERSION,
            )
        ]
    )
    artifacts.save_multimodal({"keyframes": []})
    artifacts.invalidate_from("filtering_frames")
    assert artifacts.frame_analysis_path.exists()
    assert not artifacts.multimodal_path.exists()
    artifacts.save_multimodal({"keyframes": []})
    artifacts.invalidate_from("deduplicating_frames")
    assert not artifacts.frame_analysis_path.exists()
    assert not artifacts.multimodal_path.exists()


def test_debug_summary_attaches_vlm_fields_to_kept_candidates(tmp_path: Path):
    from media_pipeline.visual.models import CandidateFrame, DedupInfo

    artifacts = ArtifactStore(tmp_path, "BVtest")
    candidate = CandidateFrame(1.0, "candidate_frames/00-00-01.000.jpg", ["periodic"])
    artifacts.save_candidate_decisions([(candidate, DedupInfo(kept=True, decision="auto"))])
    artifacts.save_frame_analysis(
        [
            FrameVerdict(
                filename="00-00-01.000.jpg",
                timestamp=1.0,
                informative=False,
                score=0.1,
                category="talking_head",
                reason="face only",
                caption="",
                kept=False,
                model="fake-vlm",
                prompt_version=PROMPT_VERSION,
            )
        ]
    )
    summary = artifacts.debug_summary()
    assert summary["selected_count"] == 0
    assert summary["decisions"][0]["analysis"]["category"] == "talking_head"
    assert summary["decisions"][0]["analysis"]["score"] == 0.1


def test_visual_extractor_filters_then_skips_cached_inference(tmp_path: Path):
    from media_pipeline.visual.extract import VisualExtractor
    from media_pipeline.visual.models import SceneSpan

    artifacts = ArtifactStore(tmp_path, "BVtest")
    (artifacts.root / "keyframes").mkdir(parents=True, exist_ok=True)
    talking = artifacts.root / "keyframes" / "talking-00-00-01.000.jpg"
    slide = artifacts.root / "keyframes" / "00-00-12.000.jpg"
    talking.write_bytes(b"jpg")
    slide.write_bytes(b"jpg")
    artifacts.save_scenes([SceneSpan(0.0, 60.0)])
    artifacts.save_candidates([])
    frames = [
        Keyframe(1.0, "keyframes/talking-00-00-01.000.jpg", sources=["periodic"]),
        Keyframe(12.0, "keyframes/00-00-12.000.jpg", sources=["scene_boundary"]),
    ]
    artifacts.save_keyframes(frames)
    vision = FakeVision()
    extractor = VisualExtractor(vision=vision)
    settings = {"vlm_keep_threshold": 0.45, "context_before_sec": 10, "context_after_sec": 20}
    first = extractor.run(
        tmp_path / "missing.mp4",
        artifacts,
        _metadata(),
        [],
        settings,
        from_stage="filtering_frames",
    )
    assert [Path(item.image_path).name for item in first["selected"]] == ["00-00-12.000.jpg"]
    assert first["document"]["frame_analysis"][1]["caption"] == "Book covers on a desk"
    assert vision.calls == ["talking-00-00-01.000.jpg", "00-00-12.000.jpg"]
    second = extractor.run(
        tmp_path / "missing.mp4",
        artifacts,
        _metadata(),
        [],
        settings,
        from_stage="aligning_multimodal",
    )
    assert vision.calls == ["talking-00-00-01.000.jpg", "00-00-12.000.jpg"]
    assert len(second["selected"]) == 1

