from __future__ import annotations

import logging
import shutil
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Callable

from media_pipeline.media import probe_duration
from media_pipeline.models import NamedSegment, VideoMetadata, frame_filename
from media_pipeline.visual.align import align_keyframes, build_multimodal_document
from media_pipeline.visual.dedup import apply_dedup, keyframes_from_dedup
from media_pipeline.visual.filtering import apply_threshold_and_overrides, filter_keyframes, selected_from_verdicts
from media_pipeline.visual.frames import extract_frame, visual_change_timestamps
from media_pipeline.visual.models import CandidateFrame, Keyframe, SceneSpan
from media_pipeline.visual.ocr import build_ocr_engine, ocr_keep_names
from media_pipeline.visual.scenes import build_scene_detector
from media_pipeline.visual.timestamps import (
    SOURCE_CHANGE,
    SOURCE_PERIODIC,
    SOURCE_SCENE,
    candidates_from_groups,
    periodic_timestamps,
    scene_boundary_timestamps,
)
from media_pipeline.visual.vlm import VisionProvider

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str, dict], None]


class VisualExtractor:
    def __init__(
        self,
        vision: VisionProvider | None = None,
        model_lock: AbstractContextManager[object] | None = None,
    ) -> None:
        self.vision = vision
        self._model_lock = model_lock

    def detect_scenes(self, video_path: Path, duration: float, settings: dict) -> list[SceneSpan]:
        detector = build_scene_detector(str(settings.get("scene_detector") or "auto"))
        scenes = detector.detect(
            video_path,
            threshold=float(settings.get("scene_threshold") or 27),
            min_scene_duration=float(settings.get("min_scene_duration_sec") or 0.8),
            duration=duration,
        )
        logger.info("Scene detector %s found %s scenes", detector.name, len(scenes))
        return scenes

    def collect_candidates(
        self,
        video_path: Path,
        duration: float,
        scenes: list[SceneSpan],
        settings: dict,
        artifacts,
        *,
        force: bool = False,
    ) -> list[CandidateFrame]:
        existing = artifacts.load_candidates()
        if existing and not force:
            return existing
        interval = float(settings.get("sample_interval_sec") or 12)
        groups = {
            SOURCE_SCENE: scene_boundary_timestamps(scenes),
            SOURCE_PERIODIC: periodic_timestamps(duration, interval),
            SOURCE_CHANGE: visual_change_timestamps(
                video_path,
                duration,
                threshold=float(settings.get("visual_change_threshold") or 0.18),
            ),
        }
        candidates = candidates_from_groups(groups)
        candidate_dir = artifacts.candidate_dir
        candidate_dir.mkdir(parents=True, exist_ok=True)
        extracted: list[CandidateFrame] = []
        for candidate in candidates:
            dest = artifacts.root / candidate.path
            stamp = min(candidate.timestamp, max(0.0, duration - 0.08)) if duration else candidate.timestamp
            try:
                extract_frame(video_path, stamp, dest)
            except Exception as exc:
                logger.warning("Skipping frame at %.3fs: %s", candidate.timestamp, exc)
                continue
            extracted.append(candidate)
        artifacts.save_candidates(extracted)
        return extracted

    def deduplicate(
        self,
        candidates: list[CandidateFrame],
        artifacts,
        settings: dict,
        *,
        force: bool = False,
    ) -> list[Keyframe]:
        existing = artifacts.load_keyframes()
        if existing and not force:
            return existing
        ocr = build_ocr_engine()
        ocr_keep = ocr_keep_names(
            candidates,
            artifacts.root,
            ocr,
            threshold=float(settings.get("ocr_change_threshold") or 0.45),
        )
        results = apply_dedup(
            candidates,
            artifacts.root,
            similarity_threshold=float(settings.get("similarity_threshold") or 0.92),
            overrides=artifacts.load_overrides(),
            ocr_keep=ocr_keep,
        )
        artifacts.save_candidate_decisions(results)
        keyframes = keyframes_from_dedup(results, artifacts.root)
        artifacts.save_keyframes(keyframes)
        return keyframes

    def run(
        self,
        video_path: Path,
        artifacts,
        metadata: VideoMetadata,
        segments: list[NamedSegment],
        settings: dict,
        *,
        from_stage: str = "detecting_scenes",
        progress: ProgressFn | None = None,
    ) -> dict:
        duration = float(metadata.duration or probe_duration(video_path) or 0.0)
        force_scenes = from_stage in {"detecting_scenes", "all"}
        force_sample = from_stage in {"detecting_scenes", "sampling_frames", "all"}
        force_dedup = from_stage in {
            "detecting_scenes",
            "sampling_frames",
            "deduplicating_frames",
            "all",
        }
        force_filter = from_stage in {
            "detecting_scenes",
            "sampling_frames",
            "deduplicating_frames",
            "filtering_frames",
            "all",
        }

        scenes = artifacts.load_scenes()
        if scenes is None or force_scenes:
            _notify(progress, "detecting_scenes")
            scenes = self.detect_scenes(video_path, duration, settings)
            artifacts.save_scenes(scenes)
            _notify(progress, "detecting_scenes", {"scene_count": len(scenes)}, event="done")
        else:
            _notify(progress, "detecting_scenes", {"scene_count": len(scenes)}, event="skip")

        existing_candidates = artifacts.load_candidates()
        if existing_candidates and not force_sample:
            candidates = existing_candidates
            _notify(
                progress,
                "sampling_frames",
                {"scene_count": len(scenes), "candidate_count": len(candidates)},
                event="skip",
            )
        else:
            _notify(progress, "sampling_frames", {"scene_count": len(scenes)})
            candidates = self.collect_candidates(
                video_path,
                duration,
                scenes,
                settings,
                artifacts,
                force=force_sample,
            )
            _notify(progress, "sampling_frames", {"candidate_count": len(candidates)}, event="done")

        existing_keys = artifacts.load_keyframes()
        if existing_keys and not force_dedup:
            keyframes = existing_keys
            _notify(
                progress,
                "deduplicating_frames",
                {"candidate_count": len(candidates), "keyframe_count": len(keyframes)},
                event="skip",
            )
        else:
            _notify(progress, "deduplicating_frames", {"candidate_count": len(candidates)})
            keyframes = self.deduplicate(
                candidates,
                artifacts,
                settings,
                force=force_dedup,
            )
            _notify(progress, "deduplicating_frames", {"keyframe_count": len(keyframes)}, event="done")

        existing_analysis = artifacts.load_frame_analysis()
        if existing_analysis is not None and not force_filter:
            _notify(progress, "filtering_frames", {"keyframe_count": len(keyframes)}, event="skip")
            threshold = float(settings.get("vlm_keep_threshold") or 0.45)
            analysis = [
                apply_threshold_and_overrides(item, threshold, artifacts.load_overrides())
                for item in existing_analysis
            ]
            artifacts.save_frame_analysis(analysis)
            selected = selected_from_verdicts(keyframes, analysis)
        else:
            _notify(progress, "filtering_frames", {"keyframe_count": len(keyframes)})
            analysis, selected = filter_keyframes(
                keyframes,
                artifacts,
                metadata,
                segments,
                settings,
                self.vision,
                model_lock=self._model_lock,
            )
            _notify(
                progress,
                "filtering_frames",
                {"keyframe_count": len(keyframes), "selected_count": len(selected)},
                event="done",
            )

        _notify(progress, "aligning_multimodal", {"keyframe_count": len(selected)})
        timeline = align_keyframes(
            selected,
            segments,
            before_sec=float(settings.get("context_before_sec") or 10),
            after_sec=float(settings.get("context_after_sec") or 20),
        )
        document = build_multimodal_document(
            metadata,
            segments,
            selected,
            timeline,
            analysis=[item.to_dict() for item in analysis],
        )
        artifacts.save_multimodal(document)
        _notify(progress, "aligning_multimodal", {"keyframe_count": len(selected)}, event="done")
        return {
            "scenes": scenes,
            "candidates": candidates,
            "keyframes": keyframes,
            "selected": selected,
            "analysis": analysis,
            "timeline": timeline,
            "document": document,
        }


def _notify(progress: ProgressFn | None, status: str, extra: dict | None = None, *, event: str = "start") -> None:
    if progress is None:
        return
    payload = dict(extra or {})
    payload["_event"] = event
    progress(status, payload)


def copy_keyframes_to_vault(keyframes: list[Keyframe], artifact_root: Path, vault_dir: Path) -> None:
    vault_dir.mkdir(parents=True, exist_ok=True)
    keep_names = {frame_filename(frame.timestamp) for frame in keyframes}
    for frame in keyframes:
        source = artifact_root / frame.image_path
        if not source.exists():
            continue
        dest = vault_dir / frame_filename(frame.timestamp)
        shutil.copy2(source, dest)
    for leftover in vault_dir.glob("*.jpg"):
        if leftover.name not in keep_names:
            leftover.unlink(missing_ok=True)
