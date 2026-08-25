from __future__ import annotations

from pathlib import Path

from media_pipeline.asr.base import ASRNotAvailableError, ASRProvider
from media_pipeline.models import ASROptions, MLX_WHISPER_REPOS, Transcript, TranscriptSegment, WordSpan


class MLXWhisperProvider(ASRProvider):
    """Apple Silicon Whisper runtime. Model weights are selected independently."""

    runtime = "MLX Whisper"

    def __init__(self, model: str) -> None:
        if model not in MLX_WHISPER_REPOS:
            supported = ", ".join(sorted(MLX_WHISPER_REPOS))
            raise ValueError(f"Unsupported Whisper model {model!r}. Expected one of: {supported}")
        self.backend_model = model
        self.id = f"whisper-{model}" if not model.startswith("whisper-") else model
        if model == "large-v3":
            self.id = "whisper-large-v3"
            self.display_name = "Whisper large-v3"
        elif model == "large-v3-turbo":
            self.id = "whisper-large-v3-turbo"
            self.display_name = "Whisper large-v3-turbo"
        else:
            self.display_name = f"Whisper {model}"
        self._repo = MLX_WHISPER_REPOS[model]

    def transcribe(self, audio_path: Path, options: ASROptions | None = None) -> Transcript:
        try:
            import mlx_whisper
        except ImportError as exc:
            raise ASRNotAvailableError(
                "mlx-whisper is not installed. Run: uv pip install -e '.[whisper]'"
            ) from exc

        options = options or ASROptions()
        # verbose=None hides Whisper's global "Detected language:" line.
        # language=None still auto-detects, but Whisper then locks the whole file to one language.
        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=self._repo,
            word_timestamps=True,
            condition_on_previous_text=False,
            language=options.language,
            verbose=None,
        )
        segments: list[TranscriptSegment] = []
        for raw in result.get("segments") or []:
            words = None
            raw_words = raw.get("words") or []
            if raw_words:
                words = [
                    WordSpan(
                        start=float(word.get("start") or 0.0),
                        end=float(word.get("end") or 0.0),
                        text=str(word.get("word") or word.get("text") or ""),
                    )
                    for word in raw_words
                ]
            segments.append(
                TranscriptSegment(
                    start=float(raw.get("start") or 0.0),
                    end=float(raw.get("end") or 0.0),
                    text=str(raw.get("text") or "").strip(),
                    words=words,
                )
            )
        if not segments:
            text = str(result.get("text") or "").strip()
            if text:
                segments.append(TranscriptSegment(start=0.0, end=0.0, text=text))
        return Transcript(
            language=result.get("language"),
            segments=segments,
            provider=self.runtime,
            model=self.id,
        )
