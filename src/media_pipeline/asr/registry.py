from __future__ import annotations

from dataclasses import dataclass

from media_pipeline.asr.agy import AgyASRProvider
from media_pipeline.asr.base import ASRNotAvailableError, ASRProvider
from media_pipeline.asr.dashscope import DashScopeASRProvider
from media_pipeline.asr.language import supports_code_switching
from media_pipeline.asr.mlx_whisper import MLXWhisperProvider
from media_pipeline.asr.qwen3 import Qwen3ASRProvider
from media_pipeline.models import ASR_MODELS


@dataclass(frozen=True)
class ModelInfo:
    id: str
    label: str
    runtime: str
    available: bool
    detail: str = ""
    code_switching: bool = False


def get_provider(model_id: str) -> ASRProvider:
    info = ASR_MODELS.get(model_id)
    if info is None:
        supported = ", ".join(ASR_MODELS)
        raise KeyError(f"Unknown ASR model {model_id!r}. Supported: {supported}")
    provider_name = info["provider"]
    backend_model = info["backend_model"]
    if provider_name == "mlx_whisper":
        return MLXWhisperProvider(model=backend_model)
    if provider_name == "qwen3":
        return Qwen3ASRProvider(model=backend_model)
    if provider_name == "dashscope":
        return DashScopeASRProvider(model=backend_model)
    if provider_name == "agy":
        return AgyASRProvider(model=backend_model)
    raise KeyError(f"Unknown ASR provider {provider_name!r}")


def list_models() -> list[ModelInfo]:
    models: list[ModelInfo] = []
    for model_id, info in ASR_MODELS.items():
        available, detail = probe_model(model_id)
        models.append(
            ModelInfo(
                id=model_id,
                label=info["label"],
                runtime=info["runtime"],
                available=available,
                detail=detail,
                code_switching=supports_code_switching(model_id),
            )
        )
    return models


def probe_model(model_id: str) -> tuple[bool, str]:
    info = ASR_MODELS.get(model_id)
    if info is None:
        return False, "unknown model"
    if info["provider"] == "dashscope":
        return _probe_dashscope()
    if info["provider"] == "agy":
        import shutil

        return (True, "agy") if shutil.which("agy") else (False, "agy is not installed or not on PATH")
    try:
        if info["provider"] == "mlx_whisper":
            import mlx_whisper  # noqa: F401

            return True, ""
        if info["provider"] == "qwen3":
            import mlx_qwen3_asr  # noqa: F401

            return True, "MLX"
    except ImportError as exc:
        extra = ".[whisper]" if info["provider"] == "mlx_whisper" else ".[qwen]"
        return False, f"not installed ({exc.name}). uv pip install -e '{extra}'"
    return False, "unsupported provider"


def _probe_dashscope() -> tuple[bool, str]:
    from media_pipeline.dashscope import resolve_api_key

    try:
        from media_pipeline.config import load_config

        config = load_config()
    except Exception:
        config = None
    if resolve_api_key(config):
        return True, "cloud"
    return False, "set dashscope.api_key or DASHSCOPE_API_KEY"


def require_provider(model_id: str) -> ASRProvider:
    available, detail = probe_model(model_id)
    if not available:
        raise ASRNotAvailableError(detail or f"ASR model {model_id} is unavailable")
    return get_provider(model_id)
