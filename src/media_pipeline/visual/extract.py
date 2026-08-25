from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Callable

from media_pipeline.media import probe_duration
from media_pipeline.models import NamedSegment, VideoMetadata, frame_filename
from media_pipeline.visual.align import align_keyframes, build_multimodal_document
from media_pipeline.visual.dedup import apply_dedup, keyframes_from_dedup
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

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str, dict], None]


class VisualExtractor:
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

        if progress:
            progress("detecting_scenes", {})
        scenes = artifacts.load_scenes()
        if scenes is None or force_scenes:
            scenes = self.detect_scenes(video_path, duration, settings)
            artifacts.save_scenes(scenes)

        if progress:
            progress("sampling_frames", {"scene_count": len(scenes)})
        candidates = self.collect_candidates(
            video_path,
            duration,
            scenes,
            settings,
            artifacts,
            force=force_sample,
        )

        if progress:
            progress("deduplicating_frames", {"candidate_count": len(candidates)})
        keyframes = self.deduplicate(
            candidates,
            artifacts,
            settings,
            force=force_dedup,
        )

        if progress:
            progress("aligning_multimodal", {"keyframe_count": len(keyframes)})
        timeline = align_keyframes(
            keyframes,
            segments,
            before_sec=float(settings.get("context_before_sec") or 10),
            after_sec=float(settings.get("context_after_sec") or 20),
        )
        document = build_multimodal_document(metadata, segments, keyframes, timeline)
        artifacts.save_multimodal(document)
        return {
            "scenes": scenes,
            "candidates": candidates,
            "keyframes": keyframes,
            "timeline": timeline,
            "document": document,
        }


def copy_keyframes_to_vault(keyframes: list[Keyframe], artifact_root: Path, vault_dir: Path) -> None:
    vault_dir.mkdir(parents=True, exist_ok=True)
    for frame in keyframes:
        source = artifact_root / frame.image_path
        if not source.exists():
            continue
        dest = vault_dir / frame_filename(frame.timestamp)
        shutil.copy2(source, dest)
