from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class TaskStatus(str, Enum):
    queued = "queued"
    fetching_metadata = "fetching_metadata"
    downloading = "downloading"
    extracting_audio = "extracting_audio"
    transcribing = "transcribing"
    diarizing = "diarizing"
    aligning = "aligning"
    aligning_transcript = "aligning_transcript"
    detecting_scenes = "detecting_scenes"
    sampling_frames = "sampling_frames"
    deduplicating_frames = "deduplicating_frames"
    filtering_frames = "filtering_frames"
    aligning_multimodal = "aligning_multimodal"
    writing_outputs = "writing_outputs"
    completed = "completed"
    failed = "failed"


ASR_MODELS: dict[str, dict[str, str]] = {
    "whisper-large-v3": {
        "label": "Whisper large-v3",
        "runtime": "MLX Whisper",
        "provider": "mlx_whisper",
        "backend_model": "large-v3",
    },
    "whisper-large-v3-turbo": {
        "label": "Whisper large-v3-turbo",
        "runtime": "MLX Whisper",
        "provider": "mlx_whisper",
        "backend_model": "large-v3-turbo",
    },
    "qwen3-asr-1.7b": {
        "label": "Qwen3-ASR-1.7B",
        "runtime": "MLX Qwen3-ASR",
        "provider": "qwen3",
        "backend_model": "qwen3-asr-1.7b",
    },
}

MLX_WHISPER_REPOS = {
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}

QWEN3_MODEL_REPOS = {
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
}


def asr_label(model_id: str) -> str:
    info = ASR_MODELS.get(model_id)
    return info["label"] if info else model_id


@dataclass
class WordSpan:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WordSpan:
        return cls(start=float(data["start"]), end=float(data["end"]), text=str(data["text"]))


@dataclass
class TranscriptSegment:
    """Coarse source-timeline speech span. Word-level times are optional and unused."""

    start: float
    end: float
    text: str
    words: list[WordSpan] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"start": self.start, "end": self.end, "text": self.text}
        if self.words is not None:
            payload["words"] = [word.to_dict() for word in self.words]
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranscriptSegment:
        words = data.get("words")
        return cls(
            start=float(data["start"]),
            end=float(data["end"]),
            text=str(data.get("text") or ""),
            words=[WordSpan.from_dict(item) for item in words] if words else None,
        )


@dataclass
class Transcript:
    language: str | None
    segments: list[TranscriptSegment]
    provider: str
    model: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "segments": [segment.to_dict() for segment in self.segments],
            "provider": self.provider,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transcript:
        return cls(
            language=data.get("language"),
            segments=[TranscriptSegment.from_dict(item) for item in data.get("segments") or []],
            provider=str(data.get("provider") or ""),
            model=str(data.get("model") or ""),
        )


@dataclass
class ASROptions:
    language: str | None = None
    context: str | None = None


@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiarizationSegment:
        return cls(start=float(data["start"]), end=float(data["end"]), speaker=str(data["speaker"]))


@dataclass
class DiarizationResult:
    segments: list[DiarizationSegment]
    provider: str

    def to_dict(self) -> dict[str, Any]:
        return {"segments": [segment.to_dict() for segment in self.segments], "provider": self.provider}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiarizationResult:
        return cls(
            segments=[DiarizationSegment.from_dict(item) for item in data.get("segments") or []],
            provider=str(data.get("provider") or ""),
        )


@dataclass
class AlignedSegment:
    start: float
    end: float
    speaker_id: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlignedSegment:
        return cls(
            start=float(data["start"]),
            end=float(data["end"]),
            speaker_id=str(data["speaker_id"]),
            text=str(data.get("text") or ""),
        )


@dataclass
class NamedSegment:
    start: float
    end: float
    speaker_id: str
    speaker_label: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NamedSegment:
        return cls(
            start=float(data["start"]),
            end=float(data["end"]),
            speaker_id=str(data["speaker_id"]),
            speaker_label=str(data.get("speaker_label") or data.get("speaker_id") or "Speaker 1"),
            text=str(data.get("text") or ""),
        )


@dataclass
class VideoMetadata:
    url: str
    title: str
    platform: str
    author: str
    video_id: str
    duration: float | None
    published: str
    description: str
    thumbnail_url: str
    asr_model: str
    media_kind: str = "video"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoMetadata:
        return cls(
            url=str(data.get("url") or ""),
            title=str(data.get("title") or ""),
            platform=str(data.get("platform") or ""),
            author=str(data.get("author") or ""),
            video_id=str(data.get("video_id") or ""),
            duration=data.get("duration"),
            published=str(data.get("published") or ""),
            description=str(data.get("description") or ""),
            thumbnail_url=str(data.get("thumbnail_url") or ""),
            asr_model=str(data.get("asr_model") or ""),
            media_kind=str(data.get("media_kind") or "video"),
        )


@dataclass
class Task:
    id: str
    url: str
    asr_model: str
    status: TaskStatus = TaskStatus.queued
    video_id: str = ""
    platform: str = ""
    title: str = ""
    note_path: str = ""
    video_path: str = ""
    audio_path: str = ""
    error_stage: str = ""
    error: str = ""
    created_at: str = ""
    updated_at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["extra"] = self.extra
        return payload

    def to_public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "url": self.url,
            "asr_model": self.asr_model,
            "asr_label": asr_label(self.asr_model),
            "status": self.status.value,
            "video_id": self.video_id,
            "platform": self.platform,
            "title": self.title,
            "note_path": self.note_path,
            "video_path": self.video_path,
            "audio_path": self.audio_path,
            "error_stage": self.error_stage,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "language": str(self.extra.get("language") or "auto"),
            "detected_languages": str(self.extra.get("detected_languages") or ""),
            "segment_count": int(self.extra.get("segment_count") or 0),
            "candidate_count": int(self.extra.get("candidate_count") or 0),
            "keyframe_count": int(self.extra.get("keyframe_count") or 0),
            "selected_count": int(self.extra.get("selected_count") or 0),
            "image_count": int(self.extra.get("image_count") or 0),
            "media_kind": str(self.extra.get("media_kind") or ""),
            "visual": dict(self.extra.get("visual") or {}),
            "rerun_stage": str(self.extra.get("rerun_stage") or ""),
            "stage_timings": {
                key: {inner: value for inner, value in entry.items() if not str(inner).startswith("_")}
                for key, entry in dict(self.extra.get("stage_timings") or {}).items()
                if isinstance(entry, dict)
            },
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Task:
        known = {item.name for item in fields(cls)}
        payload = {key: value for key, value in row.items() if key in known}
        status = payload.get("status") or TaskStatus.queued
        if not isinstance(status, TaskStatus):
            payload["status"] = TaskStatus(str(status))
        payload.setdefault("extra", {})
        return cls(**payload)


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def frame_filename(seconds: float) -> str:
    """Stable, inspectable JPEG name on the source timeline."""
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}-{minutes:02d}-{secs:02d}.{millis:03d}.jpg"


def expand_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()
