from __future__ import annotations

from pathlib import Path

from media_pipeline.asr.base import ASRNotAvailableError, ASRProvider
from media_pipeline.models import (
    ASROptions,
    QWEN3_ALIGNER_REPO,
    QWEN3_MODEL_REPOS,
    Transcript,
    TranscriptSegment,
    WordSpan,
)
from media_pipeline.transcript import group_word_spans


class Qwen3ASRProvider(ASRProvider):
    runtime = "Qwen3-ASR"

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
        model = self._load()
        kwargs: dict = {
            "audio": str(audio_path),
            "language": options.language,
            "return_time_stamps": True,
        }
        if options.context:
            kwargs["context"] = options.context
        results = model.transcribe(**kwargs)
        result = results[0] if isinstance(results, list) else results
        language = getattr(result, "language", None) or None
        text = str(getattr(result, "text", "") or "").strip()
        stamps = getattr(result, "time_stamps", None)
        words = _stamps_to_words(stamps)
        if words:
            segments = group_word_spans(words)
        elif text:
            segments = [TranscriptSegment(start=0.0, end=_audio_duration(audio_path), text=text)]
        else:
            segments = []
        return Transcript(language=language, segments=segments, provider=self.runtime, model=self.id)

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise ASRNotAvailableError(
                "qwen-asr is not installed. Run: uv pip install -e '.[qwen]'"
            ) from exc

        device_map, dtype = _torch_device(torch)
        self._model = Qwen3ASRModel.from_pretrained(
            self._repo,
            dtype=dtype,
            device_map=device_map,
            max_inference_batch_size=1,
            max_new_tokens=2048,
            forced_aligner=QWEN3_ALIGNER_REPO,
            forced_aligner_kwargs={"dtype": dtype, "device_map": device_map},
        )
        return self._model


def _torch_device(torch_mod):
    if getattr(torch_mod.backends, "mps", None) and torch_mod.backends.mps.is_available():
        dtype = torch_mod.float16
        if hasattr(torch_mod.backends.mps, "is_macos_or_newer"):
            dtype = torch_mod.bfloat16
        return "mps", dtype
    if torch_mod.cuda.is_available():
        return "cuda:0", torch_mod.bfloat16
    return "cpu", torch_mod.float32


def _stamps_to_words(stamps) -> list[WordSpan]:
    if stamps is None:
        return []
    items = getattr(stamps, "items", None)
    if items is None:
        try:
            items = list(stamps)
        except TypeError:
            return []
    words: list[WordSpan] = []
    for item in items:
        text = str(getattr(item, "text", None) or (item.get("text") if isinstance(item, dict) else "") or "")
        start = getattr(item, "start_time", None)
        end = getattr(item, "end_time", None)
        if isinstance(item, dict):
            start = item.get("start_time", item.get("start"))
            end = item.get("end_time", item.get("end"))
        if start is None or end is None:
            continue
        words.append(WordSpan(start=float(start), end=float(end), text=text))
    return words


def _audio_duration(audio_path: Path) -> float:
    from media_pipeline.media import probe_duration

    return probe_duration(audio_path) or 0.0
