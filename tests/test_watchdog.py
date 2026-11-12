from pathlib import Path

from media_pipeline.bulk import open_job
from media_pipeline.config import AppConfig, PathsConfig
from media_pipeline.watchdog import SERVER_LABEL, bulk_label, check, last_result

NOW = 2_000_000_000.0


def _config(tmp_path: Path) -> AppConfig:
    config = AppConfig(
        paths=PathsConfig(
            videos=tmp_path / "videos", audio=tmp_path / "audio", artifacts=tmp_path / "artifacts",
            logs=tmp_path / "logs", vault=tmp_path / "vault", db=tmp_path / "tasks.sqlite3", bulk=tmp_path / "bulk",
        )
    )
    config.ensure_directories()
    return config


class FakeLaunchd:
    def __init__(self, installed=(), running=()):
        self.installed_labels = set(installed)
        self.running_labels = set(running)
        self.restarts: list[str] = []

    def installed(self, label):
        return label in self.installed_labels

    def running(self, label):
        return label in self.running_labels

    def restart(self, label):
        self.restarts.append(label)
        return True


def _job(config, beat_age: float, statuses=("pending", "done"), **meta):
    store = open_job(config, "aie", create=True)
    store.add_videos([{"id": f"v{i}", "duration": 600} for i in range(len(statuses))])
    for i, status in enumerate(statuses):
        store.execute("UPDATE videos SET status=? WHERE id=?", (status, f"v{i}"))
    store.set_meta("heartbeat", NOW - beat_age)
    store.set_meta("heartbeat_download", NOW - beat_age)
    store.set_meta("state", "running")
    for key, value in meta.items():
        store.set_meta(key, value)
    return store


def test_healthy_system_is_left_alone(tmp_path: Path):
    config = _config(tmp_path)
    _job(config, beat_age=20)
    launchd = FakeLaunchd(installed={bulk_label("aie")}, running={bulk_label("aie")})
    result = check(config, launchd=launchd, api_ok=lambda url: True, now=NOW)
    assert result["ok"] and launchd.restarts == []
    assert "后端正常" in result["notes"] and "aie: 运行正常" in result["notes"]
    assert last_result(config)["at"] == NOW


def test_api_down_twice_in_a_row_restarts_the_server_once(tmp_path: Path):
    config = _config(tmp_path)
    launchd = FakeLaunchd()
    first = check(config, launchd=launchd, api_ok=lambda url: False, now=NOW)
    assert not first["ok"] and first["problems"] == ["后端无响应"] and launchd.restarts == []
    second = check(config, launchd=launchd, api_ok=lambda url: False, now=NOW + 300)
    assert launchd.restarts == [SERVER_LABEL] and second["actions"]
    check(config, launchd=launchd, api_ok=lambda url: True, now=NOW + 600)
    assert last_result(config)["strikes"] == {}
    # One bad check after recovery does not restart again straight away.
    check(config, launchd=launchd, api_ok=lambda url: False, now=NOW + 900)
    assert launchd.restarts == [SERVER_LABEL]


def test_a_stuck_download_stage_restarts_the_job_and_says_so(tmp_path: Path):
    config = _config(tmp_path)
    store = _job(config, beat_age=20 * 60)
    label = bulk_label("aie")
    launchd = FakeLaunchd(installed={label}, running={label})
    first = check(config, launchd=launchd, api_ok=lambda url: True, now=NOW)
    assert "aie: 下载 20 分钟没有动静" in first["problems"] and launchd.restarts == []
    check(config, launchd=launchd, api_ok=lambda url: True, now=NOW + 300)
    assert launchd.restarts == [label]
    assert any(event["kind"] == "watchdog" for event in store.events())


def test_a_job_with_work_left_but_no_runner_is_started_again(tmp_path: Path):
    config = _config(tmp_path)
    _job(config, beat_age=3600, asr_model="fun-asr", statuses=("done", "done"))  # downloads done, ASR left
    label = bulk_label("aie")
    launchd = FakeLaunchd(installed={label})
    check(config, launchd=launchd, api_ok=lambda url: True, now=NOW)
    check(config, launchd=launchd, api_ok=lambda url: True, now=NOW + 300)
    assert launchd.restarts == [label]


def test_finished_paused_or_manual_jobs_are_not_restarted(tmp_path: Path):
    config = _config(tmp_path)
    store = _job(config, beat_age=3600, statuses=("done", "skipped"))
    label = bulk_label("aie")
    launchd = FakeLaunchd(installed={label})
    for offset in (0, 300, 600):
        assert check(config, launchd=launchd, api_ok=lambda url: True, now=NOW + offset)["ok"]
    store.execute("UPDATE videos SET status='pending' WHERE id='v1'")
    store.set_meta("control", "pause")
    for offset in (900, 1200):
        check(config, launchd=launchd, api_ok=lambda url: True, now=NOW + offset)
    store.set_meta("control", "")
    launchd.installed_labels.clear()  # a trial run from the terminal, not a LaunchAgent
    for offset in (1500, 1800):
        check(config, launchd=launchd, api_ok=lambda url: True, now=NOW + offset)
    assert launchd.restarts == []
