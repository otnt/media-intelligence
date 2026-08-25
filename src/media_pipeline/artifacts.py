from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from media_pipeline.models import (
    AlignedSegment,
    DiarizationResult,
    NamedSegment,
    Transcript,
    VideoMetadata,
)


class ArtifactStore:
    def __init__(self, root: Path, video_id: str) -> None:
        self.root = root / video_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.asr_dir = self.root / "asr"
        self.asr_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.root / "metadata.json"
        self.diarization_path = self.root / "diarization.json"
        self.aligned_path = self.root / "aligned.json"
        self.named_path = self.root / "named.json"
        self.note_pointer_path = self.root / "note_path.txt"

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

    def remember_note(self, note_path: Path) -> None:
        self.note_pointer_path.write_text(str(note_path), encoding="utf-8")
