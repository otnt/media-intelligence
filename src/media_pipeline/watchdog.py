"""Periodic self-check (LaunchAgent ``com.media-pipeline.watchdog``, every 5 minutes).

launchd already restarts processes that crash; this catches the ones that are alive but stuck, and
bulk jobs whose runner is not running although work is left. Checks are local and cheap (one HTTP call
and a few SQLite reads); YouTube is never contacted, since the bulk runner already sees every download
and extra requests would only count against YouTube's limits.

A problem must be seen on two checks in a row before anything is restarted, so a slow moment does not
cause a restart loop; recovery takes at most ~10 minutes. The latest result is written to
``<paths.logs>/watchdog.json`` (shown on /bulk); restarts are also added to the job's event list.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from media_pipeline.config import AppConfig

logger = logging.getLogger(__name__)

SERVER_LABEL = "com.media-pipeline.server"
DOWNLOAD_STALE_SEC = 15 * 60
ASR_STALE_SEC = 30 * 60
CONFIRM_CHECKS = 2


def state_path(config: AppConfig) -> Path:
    return config.paths.logs / "watchdog.json"


def last_result(config: AppConfig) -> dict[str, Any]:
    try:
        return json.loads(state_path(config).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def bulk_label(name: str) -> str:
    return f"com.media-pipeline.bulk.{name}"


class Launchd:
    """The few launchctl calls the watchdog needs (replaced by a fake in tests)."""

    def __init__(self) -> None:
        self.domain = f"gui/{os.getuid()}"

    def plist(self, label: str) -> Path:
        return Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"

    def installed(self, label: str) -> bool:
        return self.plist(label).is_file()

    def running(self, label: str) -> bool:
        result = subprocess.run(["launchctl", "print", f"{self.domain}/{label}"], capture_output=True, text=True)
        return result.returncode == 0 and "state = running" in result.stdout

    def restart(self, label: str) -> bool:
        result = subprocess.run(["launchctl", "kickstart", "-k", f"{self.domain}/{label}"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            return True
        # Not loaded (e.g. after an unload): load it again, which also starts it (RunAtLoad).
        result = subprocess.run(["launchctl", "bootstrap", self.domain, str(self.plist(label))],
                                capture_output=True, text=True)
        return result.returncode == 0


def _api_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status == 200 and bool(json.loads(response.read() or b"{}").get("ok"))
    except Exception:  # noqa: BLE001 - any failure means "not healthy"
        return False


def check(
    config: AppConfig,
    *,
    launchd: Any = None,
    api_ok: Callable[[str], bool] = _api_ok,
    now: float | None = None,
) -> dict[str, Any]:
    from media_pipeline import bulk

    now = time.time() if now is None else now
    launchd = launchd or Launchd()
    previous = last_result(config)
    strikes: dict[str, int] = dict(previous.get("strikes") or {})
    problems: list[str] = []
    notes: list[str] = []
    actions: list[str] = []

    def strike(key: str, problem: str, label: str, job: Any = None) -> None:
        problems.append(problem)
        strikes[key] = strikes.get(key, 0) + 1
        if strikes[key] < CONFIRM_CHECKS:
            return
        strikes[key] = 0
        done = launchd.restart(label)
        action = f"{'已重启' if done else '重启失败'} {label}（{problem}）"
        actions.append(action)
        logger.warning("watchdog: %s", action)
        if job is not None:
            job.event("watchdog", action, at=now)

    seen: set[str] = set()
    url = f"http://127.0.0.1:{config.server.port}/v1/health"
    if api_ok(url):
        notes.append("后端正常")
    else:
        seen.add("api")
        strike("api", "后端无响应", SERVER_LABEL)

    for name in bulk.list_jobs(config):
        try:
            job = bulk.open_job(config, name)
            summary = job.summary(now=now)
            meta = job.all_meta()
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{name}: 读取进度失败（{exc}）")
            continue
        label = bulk_label(name)
        downloads_left, asr_left = job.work_left()
        if summary["state"] == "limited":
            notes.append(f"{name}: YouTube 限制中，{time.strftime('%H:%M', time.localtime(summary['pause_until']))} 自动探测")
        if summary["asr_pause_until"]:
            notes.append(f"{name}: 转写暂停（{summary['asr_pause_reason'][:80]}）")
        if summary["free_gb"] is not None and summary["min_free_gb"] and summary["free_gb"] < summary["min_free_gb"] + 10:
            problems.append(f"{name}: Mac 只剩 {summary['free_gb']} GB")
        if not launchd.installed(label) or summary["control"] == "pause":
            continue  # not run as a LaunchAgent (e.g. a manual trial), or paused on purpose
        if not (downloads_left or asr_left):
            continue
        if not launchd.running(label):
            seen.add(f"{name}:agent")
            strike(f"{name}:agent", f"{name}: 还有活没干完，但下载进程没在运行", label, job)
            continue
        download_beat = float(meta.get("heartbeat_download") or meta.get("heartbeat") or 0)
        asr_beat = float(meta.get("heartbeat_asr") or meta.get("heartbeat") or 0)
        if downloads_left and now - download_beat > DOWNLOAD_STALE_SEC:
            seen.add(f"{name}:download")
            strike(f"{name}:download", f"{name}: 下载 {round((now - download_beat) / 60)} 分钟没有动静", label, job)
        elif asr_left and now - asr_beat > ASR_STALE_SEC:
            seen.add(f"{name}:asr")
            strike(f"{name}:asr", f"{name}: 转写 {round((now - asr_beat) / 60)} 分钟没有动静", label, job)
        else:
            notes.append(f"{name}: 运行正常")

    for key in list(strikes):  # a problem that went away starts counting from zero again
        if key not in seen:
            strikes.pop(key)
    result = {
        "at": now,
        "ok": not problems,
        "problems": problems,
        "notes": notes,
        "actions": actions,
        "strikes": strikes,
        "interval_sec": 300,
    }
    path = state_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result
