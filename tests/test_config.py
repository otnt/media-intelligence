from pathlib import Path

from media_pipeline.config import load_config, persist_dashboard_config, persist_worker_config


def test_load_config_reads_analysis_and_vlm_threshold(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "analysis:\n  enabled: false\n  model: mlx-community/Qwen3.8-27B-4bit\n  idle_unload_sec: 30\n"
        "visual:\n  vlm_keep_threshold: 0.7\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.analysis.enabled is False
    assert config.analysis.idle_unload_sec == 30
    assert config.visual.vlm_keep_threshold == 0.7


def test_load_config_defaults_worker_concurrency(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("server:\n  host: 127.0.0.1\n", encoding="utf-8")
    config = load_config(path)
    assert config.worker.youtube == 10
    assert config.worker.bilibili == 10
    assert config.worker.other == 10
    assert config.worker.model_jobs == 1


def test_load_config_reads_worker_section(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "worker:\n  youtube: 4\n  bilibili: 8\n  default: 6\n  model_jobs: 2\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.worker.youtube == 4
    assert config.worker.bilibili == 8
    assert config.worker.other == 6
    assert config.worker.model_jobs == 2


def test_persist_worker_config_keeps_other_comments(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "# keep me\nserver:\n  host: 127.0.0.1\n\npaths:\n  vault: /tmp/vault\n",
        encoding="utf-8",
    )
    config = load_config(path)
    config.worker.set_limits(youtube=3, bilibili=12, model_jobs=2)
    persist_worker_config(config)
    saved = path.read_text(encoding="utf-8")
    assert "# keep me" in saved
    assert "vault: /tmp/vault" in saved
    assert "youtube: 3" in saved
    assert "bilibili: 12" in saved
    assert "model_jobs: 2" in saved
    reloaded = load_config(path)
    assert reloaded.worker.youtube == 3
    assert reloaded.worker.bilibili == 12
    assert reloaded.worker.model_jobs == 2


def test_load_config_defaults_dashboard_view(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("server:\n  host: 127.0.0.1\n", encoding="utf-8")
    config = load_config(path)
    assert config.dashboard.filter == "today"
    assert config.dashboard.group == "source"
    assert config.dashboard.order == "requested_desc"


def test_load_config_reads_dashboard_section(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "dashboard:\n  filter: all\n  group: none\n  order: title\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.dashboard.filter == "all"
    assert config.dashboard.group == "none"
    assert config.dashboard.order == "title"


def test_persist_dashboard_config_keeps_other_comments(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "# keep me\nserver:\n  host: 127.0.0.1\n\npaths:\n  vault: /tmp/vault\n",
        encoding="utf-8",
    )
    config = load_config(path)
    config.dashboard.set_view(filter="7d", group="status", order="requested_asc")
    persist_dashboard_config(config)
    saved = path.read_text(encoding="utf-8")
    assert "# keep me" in saved
    assert "vault: /tmp/vault" in saved
    assert "filter: 7d" in saved
    assert "group: status" in saved
    assert "order: requested_asc" in saved
    reloaded = load_config(path)
    assert reloaded.dashboard.filter == "7d"
    assert reloaded.dashboard.group == "status"
    assert reloaded.dashboard.order == "requested_asc"


def test_persist_replaces_existing_worker_block(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "server:\n  host: 127.0.0.1\n\nworker:\n  youtube: 10\n  bilibili: 10\n\nvisual:\n  sample_interval_sec: 12\n",
        encoding="utf-8",
    )
    config = load_config(path)
    config.worker.set_limits(youtube=5)
    persist_worker_config(config)
    saved = path.read_text(encoding="utf-8")
    assert saved.count("worker:") == 1
    assert "youtube: 5" in saved
    assert "visual:" in saved
    assert "sample_interval_sec: 12" in saved
