from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from media_pipeline.models import DiarizationResult, DiarizationSegment


class DiarizationProvider(ABC):
    """Answers 'who spoke when?', independently of ASR."""

    name: str

    @abstractmethod
    def diarize(self, audio_path: Path) -> DiarizationResult:
        raise NotImplementedError


class NullDiarizationProvider(DiarizationProvider):
    name = "none"

    def diarize(self, audio_path: Path) -> DiarizationResult:
        from media_pipeline.media import probe_duration

        duration = probe_duration(audio_path) or 0.0
        segments = []
        if duration > 0:
            segments = [DiarizationSegment(start=0.0, end=duration, speaker="SPEAKER_00")]
        return DiarizationResult(segments=segments, provider=self.name)


class PyannoteDiarizationProvider(DiarizationProvider):
    name = "pyannote"

    def __init__(self, model: str, hf_token: str = "") -> None:
        self.model = model
        self.hf_token = hf_token
        self._pipeline = None

    def diarize(self, audio_path: Path) -> DiarizationResult:
        pipeline = self._load()
        output = pipeline(str(audio_path))
        annotation = (
            getattr(output, "exclusive_speaker_diarization", None)
            or getattr(output, "speaker_diarization", None)
            or output
        )
        segments: list[DiarizationSegment] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append(
                DiarizationSegment(start=float(turn.start), end=float(turn.end), speaker=str(speaker))
            )
        segments.sort(key=lambda item: (item.start, item.end))
        return DiarizationResult(segments=_normalize_speaker_ids(segments), provider=self.name)

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise RuntimeError("pyannote.audio is not installed. Run: uv pip install -e '.[diarize]'") from exc

        token = self.hf_token or None
        self._pipeline = Pipeline.from_pretrained(self.model, token=token)
        if torch.backends.mps.is_available():
            self._pipeline.to(torch.device("mps"))
        elif torch.cuda.is_available():
            self._pipeline.to(torch.device("cuda"))
        return self._pipeline


def build_diarization_provider(provider: str, model: str, hf_token: str = "") -> DiarizationProvider:
    name = (provider or "pyannote").strip().lower()
    if name in {"none", "off", "disabled"}:
        return NullDiarizationProvider()
    if name != "pyannote":
        raise ValueError(f"Unknown diarization provider {provider!r}")
    try:
        import pyannote.audio  # noqa: F401
    except ImportError:
        return NullDiarizationProvider()
    return PyannoteDiarizationProvider(model=model, hf_token=hf_token)


def _normalize_speaker_ids(segments: list[DiarizationSegment]) -> list[DiarizationSegment]:
    mapping: dict[str, str] = {}
    normalized: list[DiarizationSegment] = []
    for segment in segments:
        if segment.speaker not in mapping:
            mapping[segment.speaker] = f"SPEAKER_{len(mapping):02d}"
        normalized.append(
            DiarizationSegment(start=segment.start, end=segment.end, speaker=mapping[segment.speaker])
        )
    return normalized
