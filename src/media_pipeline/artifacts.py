from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from media_pipeline.models import (
    AlignedSegment,
    DiarizationResult,
    NamedSegment,
    Transcript,
    VideoMetadata,
)
from media_pipeline.stage_timing import public_entry, public_timings, stage_keys_invalidated_by
from media_pipeline.visual.models import CandidateFrame, DedupInfo, Keyframe, SceneSpan


class ArtifactStore:
    def __init__(self, root: Path, video_id: str) -> None:
        self.video_id = video_id
        self.root = root / video_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.asr_dir = self.root / "asr"
        self.asr_dir.mkdir(parents=True, exist_ok=True)
        self.candidate_dir = self.root / "candidate_frames"
        self.keyframe_dir = self.root / "keyframes"
        self.metadata_path = self.root / "metadata.json"
        self.diarization_path = self.root / "diarization.json"
        self.aligned_path = self.root / "aligned.json"
        self.named_path = self.root / "named.json"
        self.note_pointer_path = self.root / "note_path.txt"
        self.scenes_path = self.root / "scenes.json"
        self.candidates_path = self.root / "candidates.json"
        self.candidate_decisions_path = self.root / "candidate_decisions.json"
        self.keyframes_path = self.root / "keyframes.json"
        self.overrides_path = self.root / "overrides.json"
        self.multimodal_path = self.root / "multimodal.json"
        self.timings_path = self.root / "stage_timings.json"

    def asr_path(self, model_id: str) -> Path:
        return self.asr_dir / f"{model_id}.json"

    def write_json(self, path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def read_json(self, path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    def save_metadata(self, metadata: VideoMetadata) -> None:
        self.write_json(self.metadata_path, metadata.to_dict())

    def load_metadata(self) -> VideoMetadata | None:
        if not self.metadata_path.exists():
            return None
        return VideoMetadata.from_dict(self.read_json(self.metadata_path))

    def save_transcript(self, model_id: str, transcript: Transcript) -> None:
        self.write_json(self.asr_path(model_id), transcript.to_dict())

    def load_transcript(self, model_id: str) -> Transcript | None:
        path = self.asr_path(model_id)
        if not path.exists():
            return None
        return Transcript.from_dict(self.read_json(path))

    def save_diarization(self, result: DiarizationResult) -> None:
        self.write_json(self.diarization_path, result.to_dict())

    def load_diarization(self) -> DiarizationResult | None:
        if not self.diarization_path.exists():
            return None
        return DiarizationResult.from_dict(self.read_json(self.diarization_path))

    def save_aligned(self, segments: list[AlignedSegment]) -> None:
        self.write_json(self.aligned_path, [segment.to_dict() for segment in segments])

    def save_named(self, segments: list[NamedSegment]) -> None:
        self.write_json(self.named_path, [segment.to_dict() for segment in segments])

    def load_named(self) -> list[NamedSegment] | None:
        if not self.named_path.exists():
            return None
        return [NamedSegment.from_dict(item) for item in self.read_json(self.named_path)]

    def remember_note(self, note_path: Path) -> None:
        self.note_pointer_path.write_text(str(note_path), encoding="utf-8")

    def save_scenes(self, scenes: list[SceneSpan]) -> None:
        self.write_json(self.scenes_path, [item.to_dict() for item in scenes])

    def load_scenes(self) -> list[SceneSpan] | None:
        if not self.scenes_path.exists():
            return None
        return [SceneSpan.from_dict(item) for item in self.read_json(self.scenes_path)]

    def save_candidates(self, candidates: list[CandidateFrame]) -> None:
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        self.write_json(self.candidates_path, [item.to_dict() for item in candidates])

    def load_candidates(self) -> list[CandidateFrame] | None:
        if not self.candidates_path.exists():
            return None
        return [CandidateFrame.from_dict(item) for item in self.read_json(self.candidates_path)]

    def save_candidate_decisions(self, results: list[tuple[CandidateFrame, DedupInfo]]) -> None:
        payload = []
        for candidate, info in results:
            row = candidate.to_dict()
            row["dedup"] = info.to_dict()
            payload.append(row)
        self.write_json(self.candidate_decisions_path, payload)

    def load_candidate_decisions(self) -> list[dict[str, Any]]:
        if not self.candidate_decisions_path.exists():
            return []
        data = self.read_json(self.candidate_decisions_path)
        return data if isinstance(data, list) else []

    def save_keyframes(self, frames: list[Keyframe]) -> None:
        self.keyframe_dir.mkdir(parents=True, exist_ok=True)
        self.write_json(self.keyframes_path, [item.to_dict() for item in frames])

    def load_keyframes(self) -> list[Keyframe] | None:
        if not self.keyframes_path.exists():
            return None
        return [Keyframe.from_dict(item) for item in self.read_json(self.keyframes_path)]

    def save_multimodal(self, document: dict[str, Any]) -> None:
        self.write_json(self.multimodal_path, document)

    def load_multimodal(self) -> dict[str, Any] | None:
        if not self.multimodal_path.exists():
            return None
        data = self.read_json(self.multimodal_path)
        return data if isinstance(data, dict) else None

    def load_overrides(self) -> dict[str, str]:
        if not self.overrides_path.exists():
            return {}
        data = self.read_json(self.overrides_path)
        if not isinstance(data, dict):
            return {}
        return {str(key): str(value) for key, value in data.items()}

    def set_override(self, filename: str, decision: str) -> dict[str, str]:
        overrides = self.load_overrides()
        overrides[filename] = decision
        self.write_json(self.overrides_path, overrides)
        return overrides

    def invalidate_from(self, stage: str) -> None:
        """Drop downstream visual/audio artifacts so a stage can be rerun."""
        visual_from_scenes = {"detecting_scenes", "all"}
        visual_from_sample = visual_from_scenes | {"sampling_frames"}
        visual_from_dedup = visual_from_sample | {"deduplicating_frames"}
        visual_from_align = visual_from_dedup | {"aligning_multimodal", "writing_outputs"}
        if stage in {"transcribing", "all"}:
            for path in self.asr_dir.glob("*.json"):
                path.unlink(missing_ok=True)
        if stage in {"diarizing", "transcribing", "all"}:
            self.diarization_path.unlink(missing_ok=True)
        if stage in {"aligning_transcript", "diarizing", "transcribing", "all"}:
            self.aligned_path.unlink(missing_ok=True)
            self.named_path.unlink(missing_ok=True)
        if stage in visual_from_scenes:
            self.scenes_path.unlink(missing_ok=True)
        if stage in visual_from_sample:
            self.candidates_path.unlink(missing_ok=True)
            if self.candidate_dir.exists():
                shutil.rmtree(self.candidate_dir)
            self.candidate_dir.mkdir(parents=True, exist_ok=True)
        if stage in visual_from_dedup:
            self.candidate_decisions_path.unlink(missing_ok=True)
            self.keyframes_path.unlink(missing_ok=True)
            if self.keyframe_dir.exists():
                shutil.rmtree(self.keyframe_dir)
            self.keyframe_dir.mkdir(parents=True, exist_ok=True)
        if stage in visual_from_align:
            self.multimodal_path.unlink(missing_ok=True)
        self.clear_invalidated_timings(stage)

    def load_stage_timings(self) -> dict[str, Any]:
        if not self.timings_path.exists():
            return {}
        data = self.read_json(self.timings_path)
        return public_timings(data if isinstance(data, dict) else {})

    def save_stage_timing(self, key: str, entry: dict[str, Any]) -> None:
        payload = public_entry(entry)
        if not payload:
            return
        data = self.load_stage_timings()
        data[key] = payload
        self.write_json(self.timings_path, data)

    def clear_invalidated_timings(self, stage: str) -> None:
        drop = stage_keys_invalidated_by(stage)
        if not drop:
            return
        data = {key: value for key, value in self.load_stage_timings().items() if key not in drop}
        if data:
            self.write_json(self.timings_path, data)
        elif self.timings_path.exists():
            self.timings_path.unlink(missing_ok=True)

    def debug_summary(self) -> dict[str, Any]:
        candidates = self.load_candidates() or []
        keyframes = self.load_keyframes() or []
        named = self.load_named() or []
        decisions = self.load_candidate_decisions()
        return {
            "video_id": self.video_id,
            "metadata": self.metadata_path.exists(),
            "transcript": any(self.asr_dir.glob("*.json")),
            "diarization": self.diarization_path.exists(),
            "aligned": self.aligned_path.exists(),
            "named": self.named_path.exists(),
            "scenes": self.scenes_path.exists(),
            "candidates": self.candidates_path.exists(),
            "candidate_count": len(candidates),
            "keyframes": self.keyframes_path.exists(),
            "keyframe_count": len(keyframes),
            "multimodal": self.multimodal_path.exists(),
            "overrides": self.load_overrides(),
            "segment_count": len(named),
            "decisions": decisions,
            "stage_timings": self.load_stage_timings(),
        }
