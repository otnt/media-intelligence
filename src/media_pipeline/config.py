from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from media_pipeline.models import expand_path


DEFAULT_CONFIG_PATH = Path.home() / ".config" / "media-pipeline" / "config.yaml"
OBSIDIAN_JSON = Path.home() / "Library" / "Application Support" / "obsidian" / "obsidian.json"
MAX_CONCURRENCY = 64
DEFAULT_DOMAIN_CONCURRENCY = 10
DEFAULT_MODEL_JOBS = 1
_WORKER_BLOCK_RE = re.compile(r"(?m)^worker:\n(?:(?:[ \t]+.*|[ \t]*)\n)*")


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    ingest_token: str = ""


@dataclass
class PathsConfig:
    videos: Path = field(default_factory=lambda: Path.home() / "AIContent" / "videos")
    audio: Path = field(default_factory=lambda: Path.home() / "AIContent" / "audio")
    artifacts: Path = field(default_factory=lambda: Path.home() / "AIContent" / "artifacts")
    logs: Path = field(default_factory=lambda: Path.home() / "AIContent" / "logs")
    vault: Path | None = None
    notes_folder: str = "Transcripts"
    db: Path = field(default_factory=lambda: Path.home() / "AIContent" / "artifacts" / "tasks.sqlite3")


@dataclass
class ASRConfig:
    default: str = "qwen3-asr-1.7b"
    language: str = "auto"


@dataclass
class DownloadConfig:
    format: str = "bv*[height<=480]+ba/b[height<=480]/bv*[height<=720]+ba/b[height<=720]/b"
    cookies_from_browser: str = "chrome"


@dataclass
class VisualConfig:
    sample_interval_sec: float = 12.0
    scene_detector: str = "auto"
    scene_threshold: float = 27.0
    min_scene_duration_sec: float = 0.8
    similarity_threshold: float = 0.92
    visual_change_threshold: float = 0.18
    ocr_change_threshold: float = 0.45
    context_before_sec: float = 10.0
    context_after_sec: float = 20.0
    vlm_keep_threshold: float = 0.45

    def as_dict(self) -> dict[str, float | str]:
        return {
            "sample_interval_sec": self.sample_interval_sec,
            "scene_detector": self.scene_detector,
            "scene_threshold": self.scene_threshold,
            "min_scene_duration_sec": self.min_scene_duration_sec,
            "similarity_threshold": self.similarity_threshold,
            "visual_change_threshold": self.visual_change_threshold,
            "ocr_change_threshold": self.ocr_change_threshold,
            "context_before_sec": self.context_before_sec,
            "context_after_sec": self.context_after_sec,
            "vlm_keep_threshold": self.vlm_keep_threshold,
        }

    def merged(self, overrides: dict | None) -> dict:
        payload = self.as_dict()
        for key, value in (overrides or {}).items():
            if key in payload and value is not None and value != "":
                if key == "scene_detector":
                    payload[key] = str(value)
                else:
                    payload[key] = float(value)
        return payload


@dataclass
class DiarizationConfig:
    provider: str = "pyannote"
    model: str = "pyannote/speaker-diarization-community-1"
    hf_token: str = ""


@dataclass
class WorkerConfig:
    youtube: int = DEFAULT_DOMAIN_CONCURRENCY
    bilibili: int = DEFAULT_DOMAIN_CONCURRENCY
    other: int = DEFAULT_DOMAIN_CONCURRENCY
    model_jobs: int = DEFAULT_MODEL_JOBS

    def as_dict(self) -> dict[str, int]:
        return {
            "youtube": self.youtube,
            "bilibili": self.bilibili,
            "other": self.other,
            "model_jobs": self.model_jobs,
        }

    def set_limits(
        self,
        youtube: int | None = None,
        bilibili: int | None = None,
        other: int | None = None,
        model_jobs: int | None = None,
    ) -> None:
        if youtube is not None:
            self.youtube = clamp_concurrency(youtube, DEFAULT_DOMAIN_CONCURRENCY)
        if bilibili is not None:
            self.bilibili = clamp_concurrency(bilibili, DEFAULT_DOMAIN_CONCURRENCY)
        if other is not None:
            self.other = clamp_concurrency(other, DEFAULT_DOMAIN_CONCURRENCY)
        if model_jobs is not None:
            self.model_jobs = clamp_concurrency(model_jobs, DEFAULT_MODEL_JOBS, minimum=1)


@dataclass
class AnalysisConfig:
    enabled: bool = True
    model: str = "mlx-community/Qwen3.8-27B-4bit"
    idle_unload_sec: float = 600.0
    max_tokens: int = 256


@dataclass
class SummaryConfig:
    providers: list[str] = field(default_factory=lambda: ["qwen"])
    gemini_model: str = "gemini-2.5-flash"
    openai_model: str = "gpt-4.1-mini"
    gemini_api_key: str = ""
    openai_api_key: str = ""


@dataclass
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    visual: VisualConfig = field(default_factory=VisualConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    source_path: Path | None = None

    def ensure_directories(self) -> None:
        for path in (self.paths.videos, self.paths.audio, self.paths.artifacts, self.paths.logs):
            path.mkdir(parents=True, exist_ok=True)
        self.paths.db.parent.mkdir(parents=True, exist_ok=True)
        notes_dir = self.notes_dir()
        if notes_dir is not None:
            notes_dir.mkdir(parents=True, exist_ok=True)

    def notes_dir(self) -> Path | None:
        if self.paths.vault is None:
            return None
        folder = self.paths.notes_folder.strip()
        return self.paths.vault / folder if folder else self.paths.vault


def detect_obsidian_vault() -> Path | None:
    if not OBSIDIAN_JSON.exists():
        return None
    try:
        data = json.loads(OBSIDIAN_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    vaults = list((data.get("vaults") or {}).values())
    if not vaults:
        return None
    open_vaults = [item for item in vaults if item.get("open")]
    chosen = open_vaults[0] if open_vaults else vaults[0]
    path = chosen.get("path")
    return Path(path) if path else None


def _as_path(value: Any, default: Path) -> Path:
    if not value:
        return default
    return expand_path(str(value))


def load_config(path: Path | None = None) -> AppConfig:
    config_path = path or _config_path_from_env() or DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file must be a mapping: {config_path}")
        raw = loaded

    server_raw = raw.get("server") or {}
    paths_raw = raw.get("paths") or {}
    asr_raw = raw.get("asr") or {}
    download_raw = raw.get("download") or {}
    diar_raw = raw.get("diarization") or {}
    visual_raw = raw.get("visual") or {}
    analysis_raw = raw.get("analysis") or {}
    summary_raw = raw.get("summary") or {}
    worker_raw = raw.get("worker") or {}

    vault_value = paths_raw.get("vault") or ""
    vault = expand_path(vault_value) if str(vault_value).strip() else detect_obsidian_vault()

    hf_token = str(diar_raw.get("hf_token") or "").strip()
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("MEDIA_PIPELINE_HF_TOKEN") or ""

    artifacts = _as_path(paths_raw.get("artifacts"), Path.home() / "AIContent" / "artifacts")
    config = AppConfig(
        server=ServerConfig(
            host=str(server_raw.get("host") or "127.0.0.1"),
            port=int(server_raw.get("port") or 8765),
            ingest_token=str(
                server_raw.get("ingest_token")
                or os.environ.get("MEDIA_PIPELINE_INGEST_TOKEN")
                or ""
            ).strip(),
        ),
        paths=PathsConfig(
            videos=_as_path(paths_raw.get("videos"), Path.home() / "AIContent" / "videos"),
            audio=_as_path(paths_raw.get("audio"), Path.home() / "AIContent" / "audio"),
            artifacts=artifacts,
            logs=_as_path(paths_raw.get("logs"), Path.home() / "AIContent" / "logs"),
            vault=vault,
            notes_folder=str(paths_raw.get("notes_folder") or "Transcripts"),
            db=_as_path(paths_raw.get("db"), artifacts / "tasks.sqlite3"),
        ),
        asr=ASRConfig(
            default=str(asr_raw.get("default") or "qwen3-asr-1.7b"),
            language=str(asr_raw.get("language") or "auto"),
        ),
        download=DownloadConfig(
            format=str(
                download_raw.get("format")
                or "bv*[height<=480]+ba/b[height<=480]/bv*[height<=720]+ba/b[height<=720]/b"
            ),
            cookies_from_browser=str(download_raw.get("cookies_from_browser") or ""),
        ),
        diarization=DiarizationConfig(
            provider=str(diar_raw.get("provider") or "pyannote"),
            model=str(diar_raw.get("model") or "pyannote/speaker-diarization-community-1"),
            hf_token=hf_token,
        ),
        visual=VisualConfig(
            sample_interval_sec=float(visual_raw.get("sample_interval_sec") or 12),
            scene_detector=str(visual_raw.get("scene_detector") or "auto"),
            scene_threshold=float(visual_raw.get("scene_threshold") or 27),
            min_scene_duration_sec=float(visual_raw.get("min_scene_duration_sec") or 0.8),
            similarity_threshold=float(visual_raw.get("similarity_threshold") or 0.92),
            visual_change_threshold=float(visual_raw.get("visual_change_threshold") or 0.18),
            ocr_change_threshold=float(visual_raw.get("ocr_change_threshold") or 0.45),
            context_before_sec=float(visual_raw.get("context_before_sec") or 10),
            context_after_sec=float(visual_raw.get("context_after_sec") or 20),
            vlm_keep_threshold=float(visual_raw.get("vlm_keep_threshold") or 0.45),
        ),
        analysis=AnalysisConfig(
            enabled=_as_bool(analysis_raw.get("enabled"), True),
            model=str(analysis_raw.get("model") or "mlx-community/Qwen3.8-27B-4bit"),
            idle_unload_sec=float(analysis_raw.get("idle_unload_sec") or 600),
            max_tokens=int(analysis_raw.get("max_tokens") or 256),
        ),
        summary=SummaryConfig(
            providers=_summary_providers(summary_raw.get("providers")),
            gemini_model=str(summary_raw.get("gemini_model") or "gemini-2.5-flash"),
            openai_model=str(summary_raw.get("openai_model") or "gpt-4.1-mini"),
            gemini_api_key=str(
                summary_raw.get("gemini_api_key")
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
                or ""
            ).strip(),
            openai_api_key=str(
                summary_raw.get("openai_api_key") or os.environ.get("OPENAI_API_KEY") or ""
            ).strip(),
        ),
        worker=_worker_from_raw(worker_raw),
        source_path=config_path if config_path.exists() else None,
    )
    return config


def _as_bool(value: Any, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _summary_providers(value: Any) -> list[str]:
    if value is None or value == "":
        names = ["qwen"]
    elif isinstance(value, str):
        names = [part.strip().lower() for part in value.split(",") if part.strip()]
    elif isinstance(value, list):
        names = [str(part).strip().lower() for part in value if str(part).strip()]
    else:
        names = ["qwen"]
    known = {
        "qwen",
        "qwen-low",
        "qwen-medium",
        "qwen-xhigh",
        "gemini",
        "openai",
        "all",
        "qwen3.8",
        "local",
    }
    cleaned: list[str] = []
    for name in names:
        key = "qwen" if name in {"qwen3.8", "local"} else name
        if key in known and key not in cleaned:
            cleaned.append(key)
    return cleaned or ["qwen"]


def clamp_concurrency(value: Any, default: int, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(MAX_CONCURRENCY, parsed))


def _worker_from_raw(raw: dict[str, Any]) -> WorkerConfig:
    other_default = raw.get("other", raw.get("default"))
    return WorkerConfig(
        youtube=clamp_concurrency(raw.get("youtube"), DEFAULT_DOMAIN_CONCURRENCY),
        bilibili=clamp_concurrency(raw.get("bilibili"), DEFAULT_DOMAIN_CONCURRENCY),
        other=clamp_concurrency(other_default, DEFAULT_DOMAIN_CONCURRENCY),
        model_jobs=clamp_concurrency(raw.get("model_jobs"), DEFAULT_MODEL_JOBS, minimum=1),
    )


def persist_worker_config(config: AppConfig) -> None:
    path = config.source_path
    if path is None or not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    block = _format_worker_yaml(config.worker)
    updated, count = _WORKER_BLOCK_RE.subn(block, text, count=1)
    if count == 0:
        updated = text.rstrip() + "\n\n" + block
    path.write_text(updated, encoding="utf-8")


def _format_worker_yaml(worker: WorkerConfig) -> str:
    limits = worker.as_dict()
    lines = ["worker:\n"]
    for key, value in limits.items():
        lines.append(f"  {key}: {value}\n")
    return "".join(lines)


def write_default_config(path: Path | None = None) -> Path:
    target = path or DEFAULT_CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    example = Path(__file__).resolve().parents[2] / "config.example.yaml"
    vault = detect_obsidian_vault()
    text = example.read_text(encoding="utf-8") if example.exists() else ""
    if vault and 'vault: ""' in text:
        text = text.replace('vault: ""', f"vault: {yaml_quote(str(vault))}")
    if not text:
        text = "server:\n  host: 127.0.0.1\n  port: 8765\n"
    target.write_text(text, encoding="utf-8")
    return target


def yaml_quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _config_path_from_env() -> Path | None:
    raw = os.environ.get("MEDIA_PIPELINE_CONFIG")
    return expand_path(raw) if raw else None
