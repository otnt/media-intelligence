from __future__ import annotations

import os
from pathlib import Path

from media_pipeline.asr.base import ASRNotAvailableError, ASRProvider
from media_pipeline.asr.language import format_detected_languages
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
        self._backend = ""

    def transcribe(self, audio_path: Path, options: ASROptions | None = None) -> Transcript:
        options = options or ASROptions()
        self._load()
        if self._backend == "mlx":
            return self._transcribe_mlx(audio_path, options)
        return self._transcribe_torch(audio_path, options)

    def _transcribe_mlx(self, audio_path: Path, options: ASROptions) -> Transcript:
        kwargs: dict = {
            "audio": str(audio_path),
            "language": options.language,
            "return_timestamps": True,
        }
        if options.context:
            kwargs["context"] = options.context
        result = self._model.transcribe(**kwargs)
        language = format_detected_languages(getattr(result, "language", None)) or None
        words = _stamps_to_words(getattr(result, "segments", None))
        text = str(getattr(result, "text", "") or "").strip()
        if words:
            segments = group_word_spans(words)
        elif text:
            segments = [TranscriptSegment(start=0.0, end=_audio_duration(audio_path), text=text)]
        else:
            segments = []
        return Transcript(language=language, segments=segments, provider=self.runtime, model=self.id)

    def _transcribe_torch(self, audio_path: Path, options: ASROptions) -> Transcript:
        kwargs: dict = {
            "audio": str(audio_path),
            "language": options.language,
            "return_time_stamps": True,
        }
        if options.context:
            kwargs["context"] = options.context
        results = self._model.transcribe(**kwargs)
        result = results[0] if isinstance(results, list) else results
        language = format_detected_languages(getattr(result, "language", None)) or None
        text = str(getattr(result, "text", "") or "").strip()
        words = _stamps_to_words(getattr(result, "time_stamps", None))
        if words:
            segments = group_word_spans(words)
        elif text:
            segments = [TranscriptSegment(start=0.0, end=_audio_duration(audio_path), text=text)]
        else:
            segments = []
        _release_torch_cache()
        return Transcript(language=language, segments=segments, provider=self.runtime, model=self.id)

    def _load(self):
        if self._model is not None:
            return self._model
        mlx_error = None
        try:
            from mlx_qwen3_asr import Session
        except ImportError as exc:
            mlx_error = exc
        else:
            self._model = Session(model=self._repo)
            self._backend = "mlx"
            self.runtime = "MLX Qwen3-ASR"
            return self._model

        try:
            self._load_torch()
        except ASRNotAvailableError:
            raise ASRNotAvailableError(
                "Qwen3-ASR is not installed. Run: uv pip install -e '.[qwen]'"
            ) from mlx_error
        return self._model

    def _load_torch(self):
        _limit_mps_cache()
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise ASRNotAvailableError(
                "qwen-asr is not installed. Run: uv pip install -e '.[qwen]'"
            ) from exc

        device_map, dtype = _torch_device(torch)
        load_kwargs: dict = {
            "dtype": dtype,
            "device_map": device_map,
            "low_cpu_mem_usage": True,
            "max_inference_batch_size": 1,
            "max_new_tokens": 2048,
            "forced_aligner": QWEN3_ALIGNER_REPO,
            "forced_aligner_kwargs": {
                "dtype": dtype,
                "device_map": device_map,
                "low_cpu_mem_usage": True,
            },
        }
        try:
            load_kwargs["attn_implementation"] = "sdpa"
            self._model = Qwen3ASRModel.from_pretrained(self._repo, **load_kwargs)
        except Exception:
            load_kwargs.pop("attn_implementation", None)
            self._model = Qwen3ASRModel.from_pretrained(self._repo, **load_kwargs)
        self._backend = "torch"
        self.runtime = "Qwen3-ASR (PyTorch)"
        return self._model


def _limit_mps_cache() -> None:
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.40")


def _release_torch_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _torch_device(torch_mod):
    if getattr(torch_mod.backends, "mps", None) and torch_mod.backends.mps.is_available():
        return "mps", torch_mod.float16
    if torch_mod.cuda.is_available():
        return "cuda:0", torch_mod.bfloat16
    return "cpu", torch_mod.float32


def _stamps_to_words(stamps) -> list[WordSpan]:
    if stamps is None:
        return []
    if isinstance(stamps, dict):
        stamps = stamps.get("items") or stamps.get("segments") or stamps.get("words") or []
    items = getattr(stamps, "items", None)
    if items is None:
        try:
            items = list(stamps)
        except TypeError:
            return []
    words: list[WordSpan] = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("word") or "")
            start = item.get("start_time", item.get("start"))
            end = item.get("end_time", item.get("end"))
        else:
            text = str(getattr(item, "text", None) or getattr(item, "word", None) or "")
            start = getattr(item, "start_time", None)
            if start is None:
                start = getattr(item, "start", None)
            end = getattr(item, "end_time", None)
            if end is None:
                end = getattr(item, "end", None)
        if start is None or end is None:
            continue
        words.append(WordSpan(start=float(start), end=float(end), text=text))
    return words


def _audio_duration(audio_path: Path) -> float:
    from media_pipeline.media import probe_duration

    return probe_duration(audio_path) or 0.0
