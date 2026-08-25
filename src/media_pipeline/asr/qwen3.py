from __future__ import annotations

from pathlib import Path

from media_pipeline.asr.base import ASRNotAvailableError, ASRProvider
from media_pipeline.asr.language import format_detected_languages
from media_pipeline.models import ASROptions, QWEN3_MODEL_REPOS, Transcript, TranscriptSegment
from media_pipeline.vad import AsrChunk, build_asr_chunks, read_wav_slice


class Qwen3ASRProvider(ASRProvider):
    runtime = "MLX Qwen3-ASR"

    def __init__(self, model: str = "qwen3-asr-1.7b") -> None:
        if model not in QWEN3_MODEL_REPOS:
            supported = ", ".join(sorted(QWEN3_MODEL_REPOS))
            raise ValueError(f"Unsupported Qwen3-ASR model {model!r}. Expected one of: {supported}")
        self.backend_model = model
        self.id = model
        self.display_name = "Qwen3-ASR-1.7B" if model == "qwen3-asr-1.7b" else model
        self._repo = QWEN3_MODEL_REPOS[model]
        self._model = None

    def transcribe(self, audio_path: Path, options: ASROptions | None = None) -> Transcript:
        options = options or ASROptions()
        session = self._load()
        chunks = build_asr_chunks(audio_path)
        segments: list[TranscriptSegment] = []
        languages: list[str] = []
        for chunk in chunks:
            text, language = self._transcribe_chunk(session, audio_path, chunk, options)
            if not text:
                continue
            if language:
                languages.append(language)
            segments.append(TranscriptSegment(start=chunk.start, end=chunk.end, text=text))
        return Transcript(
            language=format_detected_languages(languages) or None,
            segments=segments,
            provider=self.runtime,
            model=self.id,
        )

    def _transcribe_chunk(
        self,
        session,
        audio_path: Path,
        chunk: AsrChunk,
        options: ASROptions,
    ) -> tuple[str, str | None]:
        samples, sample_rate = read_wav_slice(audio_path, chunk.start, chunk.end)
        if len(samples) < int(sample_rate * 0.08):
            return "", None
        audio = _to_float32_array(samples)
        kwargs: dict = {
            "audio": (audio, sample_rate),
            "language": options.language,
            "return_timestamps": False,
        }
        if options.context:
            kwargs["context"] = options.context
        result = session.transcribe(**kwargs)
        text = str(getattr(result, "text", "") or "").strip()
        language = format_detected_languages(getattr(result, "language", None)) or None
        return text, language

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from mlx_qwen3_asr import Session
        except ImportError as exc:
            raise ASRNotAvailableError(
                "mlx-qwen3-asr is not installed. Run: uv pip install -e '.[qwen]'"
            ) from exc
        self._model = Session(model=self._repo)
        return self._model


def _to_float32_array(samples: list[float]):
    import numpy as np

    return np.asarray(samples, dtype=np.float32)
