from __future__ import annotations

import json
import os
import re
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from media_pipeline.models import expand_path
from media_pipeline.dashboard_view import (
    normalize_dashboard_filter,
    normalize_dashboard_group,
    normalize_dashboard_order,
)


DEFAULT_CONFIG_PATH = Path.home() / ".config" / "media-pipeline" / "config.yaml"
OBSIDIAN_JSON = Path.home() / "Library" / "Application Support" / "obsidian" / "obsidian.json"
MAX_CONCURRENCY = 64
DEFAULT_DOMAIN_CONCURRENCY = 10
DEFAULT_MODEL_JOBS = 1
DEFAULT_SUMMARY_PROVIDERS = ["qwen-xhigh"]
DEFAULT_QWEN_8BIT_MODEL = "mlx-community/Qwen3.8-27B-8bit"
_WORKER_BLOCK_RE = re.compile(r"(?m)^worker:\n(?:(?:[ \t]+.*|[ \t]*)\n)*")
_DASHBOARD_BLOCK_RE = re.compile(r"(?m)^dashboard:\n(?:(?:[ \t]+.*|[ \t]*)\n)*")
_VISUAL_BLOCK_RE = re.compile(r"(?m)^visual:\n(?:(?:[ \t]+.*|[ \t]*)\n)*")
_SUMMARY_BLOCK_RE = re.compile(r"(?m)^summary:\n(?:(?:[ \t]+.*|[ \t]*)\n)*")
_SERVER_BLOCK_RE = re.compile(r"(?m)^server:\n(?:(?:[ \t]+.*|[ \t]*)\n)*")


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8875
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
    # Bulk download jobs (``media-pipeline bulk``): progress tables and the local staging folders.
    bulk: Path = field(default_factory=lambda: BULK_ROOT)


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

    def apply_overrides(self, overrides: dict | None) -> None:
        payload = self.merged(overrides)
        self.sample_interval_sec = float(payload["sample_interval_sec"])
        self.scene_detector = str(payload["scene_detector"])
        self.scene_threshold = float(payload["scene_threshold"])
        self.min_scene_duration_sec = float(payload["min_scene_duration_sec"])
        self.similarity_threshold = float(payload["similarity_threshold"])
        self.visual_change_threshold = float(payload["visual_change_threshold"])
        self.ocr_change_threshold = float(payload["ocr_change_threshold"])
        self.context_before_sec = float(payload["context_before_sec"])
        self.context_after_sec = float(payload["context_after_sec"])
        self.vlm_keep_threshold = float(payload["vlm_keep_threshold"])


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
class DashboardConfig:
    filter: str = "today"
    group: str = "source"
    order: str = "requested_desc"

    def as_dict(self) -> dict[str, str]:
        return {
            "filter": self.filter,
            "group": self.group,
            "order": self.order,
        }

    def set_view(
        self,
        filter: str | None = None,
        group: str | None = None,
        order: str | None = None,
    ) -> None:
        if filter is not None:
            self.filter = normalize_dashboard_filter(filter)
        if group is not None:
            self.group = normalize_dashboard_group(group)
        if order is not None:
            self.order = normalize_dashboard_order(order)


@dataclass
class DashScopeConfig:
    """Shared Alibaba Cloud Model Studio settings for cloud ASR, keyframe judging and Qwen-Max summaries."""

    api_key: str = ""
    # Region root. Empty reuses summary.qwen_max_base_url's host (mainland China by default).
    base_url: str = ""
    # Fun-ASR does ASR + speaker tags in one call. paraformer-v2 also works.
    asr_model: str = "fun-asr"
    # 0 lets DashScope pick the speaker count; 2-100 is a hint.
    speaker_count: int = 0
    # Cloud model used when analysis.provider is dashscope.
    vision_model: str = "qwen3.8-flash"
    # Parallel keyframe judgments against the cloud model.
    vision_concurrency: int = 4
    # Fun-ASR diarization is only reliable up to ~2 h per file; longer audio goes in windows. 0 = never split.
    chunk_sec: float = 5400.0
    chunk_overlap_sec: float = 120.0


ICLOUD_DRIVE = Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs"
BULK_ROOT = Path.home() / "AIContent" / "bulk"


def _default_storage_mode() -> str:
    return "archive" if ICLOUD_DRIVE.is_dir() else "keep"


@dataclass
class StorageConfig:
    """What happens to media once a video is processed (see storage.py)."""

    # Replace the 16 kHz WAV (derived, ~115 MB/hour) with a ~22 MB/hour audio copy for re-transcribing.
    compact_after_processing: bool = True
    # The original download: "archive" moves it into iCloud Drive (kept, and evicted from this Mac once
    # uploaded), "keep" leaves it in paths.videos, "delete" removes it (re-downloaded if ever needed).
    mode: str = field(default_factory=_default_storage_mode)
    archive_dir: Path = field(default_factory=lambda: ICLOUD_DRIVE / "Media Intelligence" / "videos")
    # In archive mode, free the local copy as soon as iCloud has it instead of waiting for disk pressure.
    evict_archived: bool = True
    # Delete the small audio copy after this many days (0 = right after processing, -1 = keep forever).
    keep_audio_days: int = 30


@dataclass
class AgyConfig:
    """Antigravity CLI (agy): Gemini through the user's Google login, no API key."""

    binary: str = "agy"
    asr_model: str = "gemini-4-argon-medium"
    # Used by the agy-gemini summary backend.
    summary_model: str = "gemini-4-argon-medium"
    # Audio windows sent per agy call; neighbours overlap so speakers can be matched.
    window_sec: float = 180.0
    overlap_sec: float = 45.0
    concurrency: int = 3
    timeout_sec: int = 900
    rescale: bool = True


@dataclass
class AnalysisConfig:
    enabled: bool = True
    # local = MLX Qwen on this Mac; dashscope = dashscope.vision_model in the cloud.
    provider: str = "local"
    model: str = "mlx-community/Qwen3.8-27B-4bit"
    idle_unload_sec: float = 600.0
    max_tokens: int = 256


@dataclass
class SummaryConfig:
    providers: list[str] = field(default_factory=lambda: list(DEFAULT_SUMMARY_PROVIDERS))
    qwen_8bit_model: str = DEFAULT_QWEN_8BIT_MODEL
    gemini_model: str = "gemini-2.5-flash"
    openai_model: str = "gpt-4.1-mini"
    deepseek_model: str = "deepseek-v4-pro"
    deepseek_base_url: str = "https://api.deepseek.com"
    qwen_max_model: str = "qwen3.8-max"
    # Mainland China DashScope default. Match key region:
    # Singapore intl: dashscope-intl.aliyuncs.com; US (Virginia): dashscope-us.aliyuncs.com
    qwen_max_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # DashScope reasoning_effort for qwen3.8-max: low (4k thinking tokens), medium (16k), xhigh (256k,
    # the model's default), or none to answer without thinking.
    qwen_max_reasoning: str = "xhigh"
    gemini_api_key: str = ""
    openai_api_key: str = ""
    deepseek_api_key: str = ""
    qwen_api_key: str = ""
    prompt: str = ""


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
    dashscope: DashScopeConfig = field(default_factory=DashScopeConfig)
    agy: AgyConfig = field(default_factory=AgyConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
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
    dashboard_raw = raw.get("dashboard") or {}
    dashscope_raw = raw.get("dashscope") or {}
    agy_raw = raw.get("agy") or {}
    storage_raw = raw.get("storage") or {}

    vault_value = paths_raw.get("vault") or ""
    vault = expand_path(vault_value) if str(vault_value).strip() else detect_obsidian_vault()

    hf_token = str(diar_raw.get("hf_token") or "").strip()
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("MEDIA_PIPELINE_HF_TOKEN") or ""

    artifacts = _as_path(paths_raw.get("artifacts"), Path.home() / "AIContent" / "artifacts")
    config = AppConfig(
        server=ServerConfig(
            host=str(server_raw.get("host") or "127.0.0.1"),
            port=int(server_raw.get("port") or 8875),
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
            bulk=_as_path(paths_raw.get("bulk"), BULK_ROOT),
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
            provider=str(analysis_raw.get("provider") or "local").strip().lower(),
            model=str(analysis_raw.get("model") or "mlx-community/Qwen3.8-27B-4bit"),
            idle_unload_sec=float(analysis_raw.get("idle_unload_sec") or 600),
            max_tokens=int(analysis_raw.get("max_tokens") or 256),
        ),
        summary=SummaryConfig(
            providers=_summary_providers(summary_raw.get("providers")),
            qwen_8bit_model=str(summary_raw.get("qwen_8bit_model") or DEFAULT_QWEN_8BIT_MODEL),
            gemini_model=str(summary_raw.get("gemini_model") or "gemini-2.5-flash"),
            openai_model=str(summary_raw.get("openai_model") or "gpt-4.1-mini"),
            deepseek_model=str(summary_raw.get("deepseek_model") or "deepseek-v4-pro"),
            deepseek_base_url=str(
                summary_raw.get("deepseek_base_url") or "https://api.deepseek.com"
            ).rstrip("/"),
            qwen_max_model=str(summary_raw.get("qwen_max_model") or "qwen3.8-max"),
            qwen_max_reasoning=normalize_qwen_reasoning(summary_raw.get("qwen_max_reasoning")),
            qwen_max_base_url=str(
                summary_raw.get("qwen_max_base_url")
                or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ).rstrip("/"),
            gemini_api_key=str(
                summary_raw.get("gemini_api_key")
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
                or ""
            ).strip(),
            openai_api_key=str(
                summary_raw.get("openai_api_key") or os.environ.get("OPENAI_API_KEY") or ""
            ).strip(),
            deepseek_api_key=str(
                summary_raw.get("deepseek_api_key") or os.environ.get("DEEPSEEK_API_KEY") or ""
            ).strip(),
            qwen_api_key=str(
                summary_raw.get("qwen_api_key")
                or summary_raw.get("qwen_max_api_key")
                or os.environ.get("DASHSCOPE_API_KEY")
                or os.environ.get("QWEN_API_KEY")
                or ""
            ).strip(),
            prompt=str(summary_raw.get("prompt") or "").strip(),
        ),
        dashscope=DashScopeConfig(
            api_key=str(
                dashscope_raw.get("api_key")
                or summary_raw.get("qwen_api_key")
                or summary_raw.get("qwen_max_api_key")
                or os.environ.get("DASHSCOPE_API_KEY")
                or os.environ.get("QWEN_API_KEY")
                or ""
            ).strip(),
            base_url=str(dashscope_raw.get("base_url") or "").strip().rstrip("/"),
            asr_model=str(dashscope_raw.get("asr_model") or "fun-asr").strip(),
            speaker_count=int(dashscope_raw.get("speaker_count") or 0),
            vision_model=str(dashscope_raw.get("vision_model") or "qwen3.8-flash").strip(),
            vision_concurrency=max(1, int(dashscope_raw.get("vision_concurrency") or 4)),
            chunk_sec=float(dashscope_raw.get("chunk_sec", 5400) or 0),
            chunk_overlap_sec=float(dashscope_raw.get("chunk_overlap_sec", 120) or 0),
        ),
        agy=AgyConfig(
            binary=str(agy_raw.get("binary") or "agy"),
            asr_model=str(agy_raw.get("asr_model") or "gemini-4-argon-medium"),
            summary_model=str(agy_raw.get("summary_model") or "gemini-4-argon-medium"),
            window_sec=float(agy_raw.get("window_sec") or 180),
            overlap_sec=float(agy_raw.get("overlap_sec") or 45),
            concurrency=max(1, int(agy_raw.get("concurrency") or 3)),
            timeout_sec=int(agy_raw.get("timeout_sec") or 900),
            rescale=bool(agy_raw.get("rescale", True)),
        ),
        storage=StorageConfig(
            compact_after_processing=_as_bool(storage_raw.get("compact_after_processing"), True),
            mode=_storage_mode(storage_raw.get("mode")),
            archive_dir=_as_path(storage_raw.get("archive_dir"), ICLOUD_DRIVE / "Media Intelligence" / "videos"),
            evict_archived=_as_bool(storage_raw.get("evict_archived"), True),
            keep_audio_days=int(storage_raw.get("keep_audio_days", 30)),
        ),
        worker=_worker_from_raw(worker_raw),
        dashboard=_dashboard_from_raw(dashboard_raw),
        source_path=config_path if config_path.exists() else None,
    )
    return config


def normalize_qwen_reasoning(value: Any) -> str:
    """Map onto the three levels qwen3.8-max accepts; aliases follow Alibaba's documented mapping."""
    effort = str(value or "").strip().lower()
    aliases = {"high": "xhigh", "max": "xhigh", "minimal": "low", "off": "none", "false": "none"}
    effort = aliases.get(effort, effort)
    return effort if effort in {"low", "medium", "xhigh", "none"} else "xhigh"


def _storage_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    return mode if mode in {"archive", "keep", "delete"} else _default_storage_mode()


def _as_bool(value: Any, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _summary_providers(value: Any) -> list[str]:
    if value is None or value == "":
        names = list(DEFAULT_SUMMARY_PROVIDERS)
    elif isinstance(value, str):
        names = [part.strip().lower() for part in value.split(",") if part.strip()]
    elif isinstance(value, list):
        names = [str(part).strip().lower() for part in value if str(part).strip()]
    else:
        names = list(DEFAULT_SUMMARY_PROVIDERS)
    aliases = {
        "qwen3.8": "qwen",
        "local": "qwen",
        "deepseek": "deepseek-v4-pro",
        "deepseek-pro": "deepseek-v4-pro",
        "qwen-max": "qwen3.8-max",
        "agy": "agy-gemini",
        "flash": "deepseek-v4.1-flash",
        "deepseek-flash": "deepseek-v4.1-flash",
    }
    known = {
        "qwen",
        "qwen-low",
        "qwen-medium",
        "qwen-xhigh",
        "qwen-8bit",
        "qwen-8bit-low",
        "qwen-8bit-medium",
        "qwen-8bit-xhigh",
        "gemini",
        "openai",
        "deepseek-v4-pro",
        "deepseek-v4.1-flash",
        "qwen3.8-max",
        "agy-gemini",
        "all",
    }
    cleaned: list[str] = []
    for name in names:
        key = aliases.get(name, name)
        if key in known and key not in cleaned:
            cleaned.append(key)
    return cleaned or list(DEFAULT_SUMMARY_PROVIDERS)


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
    _persist_named_block(config, _WORKER_BLOCK_RE, _format_worker_yaml(config.worker))


def persist_dashboard_config(config: AppConfig) -> None:
    _persist_named_block(config, _DASHBOARD_BLOCK_RE, _format_dashboard_yaml(config.dashboard))


def persist_visual_config(config: AppConfig) -> None:
    _persist_named_block(config, _VISUAL_BLOCK_RE, _format_visual_yaml(config.visual))


def persist_summary_prompt(config: AppConfig) -> None:
    path = config.source_path
    if path is None or not path.exists():
        return
    prompt = str(config.summary.prompt or "").strip()
    line = f"  prompt: {yaml_quote(prompt)}\n"
    text = path.read_text(encoding="utf-8")
    match = _SUMMARY_BLOCK_RE.search(text)
    if match:
        updated = text[: match.start()] + _upsert_indented_key(match.group(0), "prompt", line) + text[match.end() :]
    else:
        updated = text.rstrip() + "\n\nsummary:\n" + line
    path.write_text(updated, encoding="utf-8")


def persist_ingest_token(config: AppConfig) -> None:
    path = config.source_path
    if path is None or not path.exists():
        return
    token = str(config.server.ingest_token or "").strip()
    line = f"  ingest_token: {yaml_quote(token)}\n"
    text = path.read_text(encoding="utf-8")
    match = _SERVER_BLOCK_RE.search(text)
    if match:
        updated = (
            text[: match.start()]
            + _upsert_indented_key(match.group(0), "ingest_token", line)
            + text[match.end() :]
        )
    else:
        updated = text.rstrip() + "\n\nserver:\n" + line
    path.write_text(updated, encoding="utf-8")


def ensure_ingest_token(config: AppConfig) -> str | None:
    """Create and persist ingest_token if missing. Returns the new token, else None."""
    if str(config.server.ingest_token or "").strip():
        return None
    token = secrets.token_urlsafe(24)
    config.server.ingest_token = token
    persist_ingest_token(config)
    return token


def _persist_named_block(config: AppConfig, pattern: re.Pattern[str], block: str) -> None:
    path = config.source_path
    if path is None or not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    updated, count = pattern.subn(block, text, count=1)
    if count == 0:
        updated = text.rstrip() + "\n\n" + block
    path.write_text(updated, encoding="utf-8")


def _format_worker_yaml(worker: WorkerConfig) -> str:
    limits = worker.as_dict()
    lines = ["worker:\n"]
    for key, value in limits.items():
        lines.append(f"  {key}: {value}\n")
    return "".join(lines)


def _format_dashboard_yaml(dashboard: DashboardConfig) -> str:
    view = dashboard.as_dict()
    lines = ["dashboard:\n"]
    for key, value in view.items():
        lines.append(f"  {key}: {value}\n")
    return "".join(lines)


def _format_visual_yaml(visual: VisualConfig) -> str:
    payload = visual.as_dict()
    lines = ["visual:\n"]
    for key, value in payload.items():
        if key == "scene_detector":
            lines.append(f"  {key}: {value}\n")
        else:
            lines.append(f"  {key}: {_yaml_number(float(value))}\n")
    return "".join(lines)


def _yaml_number(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.8f}".rstrip("0").rstrip(".")


def _upsert_indented_key(block: str, key: str, formatted_line: str) -> str:
    pattern = re.compile(
        rf"(?m)^(?P<indent>[ \t]+){re.escape(key)}:[ \t]*(?:[|>][+-]?\d*)?[ \t]*\n"
        rf"(?:(?:(?P=indent)[ \t]+.*|[ \t]*)\n)*"
        rf"|^(?P<indent2>[ \t]+){re.escape(key)}:[ \t].*\n"
    )
    updated, count = pattern.subn(formatted_line, block, count=1)
    if count:
        return updated
    if not block.endswith("\n"):
        block += "\n"
    newline_at = block.find("\n")
    if newline_at == -1:
        return block + formatted_line
    return block[: newline_at + 1] + formatted_line + block[newline_at + 1 :]


def _dashboard_from_raw(raw: dict[str, Any]) -> DashboardConfig:
    return DashboardConfig(
        filter=normalize_dashboard_filter(str(raw.get("filter") or "today")),
        group=normalize_dashboard_group(str(raw.get("group") or "source")),
        order=normalize_dashboard_order(str(raw.get("order") or "requested_desc")),
    )


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
        text = "server:\n  host: 127.0.0.1\n  port: 8875\n"
    target.write_text(text, encoding="utf-8")
    return target


def yaml_quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _config_path_from_env() -> Path | None:
    raw = os.environ.get("MEDIA_PIPELINE_CONFIG")
    return expand_path(raw) if raw else None
