from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from media_pipeline.models import expand_path


DEFAULT_CONFIG_PATH = Path.home() / ".config" / "media-pipeline" / "config.yaml"
OBSIDIAN_JSON = Path.home() / "Library" / "Application Support" / "obsidian" / "obsidian.json"


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765


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
    default: str = "whisper-large-v3-turbo"


@dataclass
class DownloadConfig:
    format: str = "bv*[height<=480]+ba/b[height<=480]/bv*[height<=720]+ba/b[height<=720]/b"
    cookies_from_browser: str = "chrome"


@dataclass
class DiarizationConfig:
    provider: str = "pyannote"
    model: str = "pyannote/speaker-diarization-community-1"
    hf_token: str = ""


@dataclass
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
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
        asr=ASRConfig(default=str(asr_raw.get("default") or "whisper-large-v3-turbo")),
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
        source_path=config_path if config_path.exists() else None,
    )
    return config


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
