from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from media_pipeline.models import ASROptions, Transcript


class ASRProvider(ABC):
    """Generic speech-to-text backend.

    Implementations answer "what was said?", not "who spoke?".
    """

    id: str
    display_name: str
    runtime: str

    @abstractmethod
    def transcribe(self, audio_path: Path, options: ASROptions | None = None) -> Transcript:
        raise NotImplementedError


class ASRNotAvailableError(RuntimeError):
    pass
