from __future__ import annotations

import gc
import logging
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from queue import SimpleQueue

from media_pipeline.config import AnalysisConfig, AppConfig

logger = logging.getLogger(__name__)

DEFAULT_VLM_MODEL = "mlx-community/Qwen3.8-27B-4bit"


class _MlxSerialLoop:
    """Run MLX work on one daemon thread that is never joined or stopped.

    Python 3.13 aborts with `PyThreadState_Get` / exit 133 when a thread that
    touched MLX Metal exits. `ThreadPoolExecutor` joins its workers in atexit,
    which is exactly that abort. This loop parks forever instead.
    """

    def __init__(self) -> None:
        self._jobs: SimpleQueue[tuple[Future, object, tuple]] = SimpleQueue()
        self._thread = threading.Thread(target=self._run, name="mlx-vlm", daemon=True)
        self._thread.start()

    def call(self, fn, *args):
        if threading.current_thread() is self._thread:
            return fn(*args)
        future: Future = Future()
        self._jobs.put((future, fn, args))
        return future.result()

    def _run(self) -> None:
        while True:
            future, fn, args = self._jobs.get()
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(fn(*args))
                except BaseException as exc:
                    future.set_exception(exc)


_LOOP: _MlxSerialLoop | None = None
_LOOP_LOCK = threading.Lock()


def _mlx_loop() -> _MlxSerialLoop:
    global _LOOP
    with _LOOP_LOCK:
        if _LOOP is None:
            _LOOP = _MlxSerialLoop()
        return _LOOP


class VisionProvider:
    """Local vision-language model used to judge extracted stills."""

    name = "none"
    model_id = ""
    last_used: float = 0.0

    @property
    def loaded(self) -> bool:
        return False

    def judge(self, image_path: Path, prompt: str) -> str:
        raise NotImplementedError

    def unload(self) -> None:
        return None


class NullVisionProvider(VisionProvider):
    """Keep every frame when analysis is disabled or the VLM is unavailable."""

    name = "none"

    def judge(self, image_path: Path, prompt: str) -> str:
        return (
            '{"informative": true, "score": 1.0, "category": "other",'
            ' "reason": "analysis_unavailable", "caption": ""}'
        )


class MlxVlmProvider(VisionProvider):
    name = "mlx_vlm"

    def __init__(self, model_id: str, max_tokens: int = 256) -> None:
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.last_used = 0.0
        self._model = None
        self._processor = None
        self._config = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        self._on_thread(self._load_sync)

    def unload(self) -> None:
        if self._model is None:
            return
        try:
            self._on_thread(self._unload_sync)
        except Exception:
            logger.exception("Failed to unload vision model %s", self.model_id)

    def close(self) -> None:
        self.unload()

    def judge(self, image_path: Path, prompt: str) -> str:
        return str(self._on_thread(self._judge_sync, image_path, prompt))

    def _on_thread(self, fn, *args):
        return _mlx_loop().call(fn, *args)

    def _load_sync(self) -> None:
        if self._model is not None:
            return
        from mlx_vlm import load as vlm_load
        from mlx_vlm.utils import load_config

        path = _resolved_model_path(self.model_id)
        logger.info("Loading vision model %s from %s", self.model_id, path)
        self._model, self._processor = vlm_load(path)
        self._config = load_config(path)
        self.last_used = time.monotonic()

    def _unload_sync(self) -> None:
        if self._model is None:
            return
        logger.info("Unloading vision model %s", self.model_id)
        self._model = None
        self._processor = None
        self._config = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass

    def _judge_sync(self, image_path: Path, prompt: str) -> str:
        self._load_sync()
        self.last_used = time.monotonic()
        from mlx_vlm import generate

        formatted = _format_prompt(self._processor, self._config, prompt)
        output = _generate(generate, self._model, self._processor, formatted, image_path, self.max_tokens)
        return _as_text(output)


def probe_vlm(config: AppConfig | AnalysisConfig | None = None) -> tuple[bool, str]:
    analysis = _analysis_config(config)
    if not analysis.enabled:
        return True, "disabled"
    try:
        import mlx_vlm  # noqa: F401
    except ImportError as exc:
        return False, f"not installed ({exc.name}). uv pip install -e '.[analysis]'"
    except Exception as exc:
        return False, f"mlx_vlm import failed: {exc}"
    if not _local_model_available(analysis.model):
        return False, f"weights missing. hf download {analysis.model}"
    return True, analysis.model


def build_vision_provider(config: AppConfig | AnalysisConfig | None = None) -> VisionProvider:
    analysis = _analysis_config(config)
    if not analysis.enabled:
        return NullVisionProvider()
    available, detail = probe_vlm(analysis)
    if not available:
        logger.warning("Vision analysis unavailable: %s", detail)
        return NullVisionProvider()
    return MlxVlmProvider(analysis.model, max_tokens=int(analysis.max_tokens))


def _analysis_config(config: AppConfig | AnalysisConfig | None) -> AnalysisConfig:
    if isinstance(config, AnalysisConfig):
        return config
    if config is not None:
        return config.analysis
    return AnalysisConfig()


def _resolved_model_path(model_id: str) -> str:
    """Prefer a local snapshot so mlx-vlm does not download 16GB at first load."""
    cache = Path.home() / ".cache" / "huggingface" / "hub" / _hub_dirname(model_id)
    snapshots = cache / "snapshots"
    if snapshots.is_dir():
        for snapshot in snapshots.iterdir():
            if _is_weight_dir(snapshot):
                return str(snapshot)
    lmstudio = Path.home() / ".lmstudio" / "models" / model_id
    if _is_weight_dir(lmstudio):
        return str(lmstudio)
    return model_id


def _local_model_available(model_id: str) -> bool:
    path = Path(_resolved_model_path(model_id))
    return path.is_dir() and _is_weight_dir(path)


def _is_weight_dir(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").exists():
        return False
    return any(path.glob("*.safetensors")) or any(path.glob("*.npz"))


def _hub_dirname(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def _format_prompt(processor, config, prompt: str) -> str:
    from mlx_vlm.prompt_utils import apply_chat_template

    try:
        return apply_chat_template(processor, config, prompt, num_images=1, enable_thinking=False)
    except TypeError:
        return apply_chat_template(processor, config, prompt, num_images=1)


def _generate(generate, model, processor, prompt: str, image_path: Path, max_tokens: int):
    image = str(image_path)
    attempts = (
        {"prompt": prompt, "image": image, "max_tokens": max_tokens, "verbose": False, "temperature": 0.0},
        {"prompt": prompt, "images": [image], "max_tokens": max_tokens, "verbose": False},
    )
    for kwargs in attempts:
        try:
            return generate(model, processor, **kwargs)
        except TypeError:
            continue
    return generate(model, processor, prompt, image, max_tokens=max_tokens, verbose=False)


def _as_text(output: object) -> str:
    text = getattr(output, "text", None)
    if text:
        return str(text)
    if isinstance(output, dict):
        for key in ("text", "output", "response"):
            value = output.get(key)
            if value:
                return str(value)
    return str(output or "")
