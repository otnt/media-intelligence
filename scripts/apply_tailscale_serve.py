#!/usr/bin/env python3
"""Apply scripts/tailscale-serve.json using node-level `tailscale serve` commands."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "scripts" / "tailscale-serve.json"
LOG = "[tailscale-serve]"


class ApplyError(RuntimeError):
    pass


def find_cli() -> str:
    override = os.environ.get("TAILSCALE_CLI", "").strip()
    if override and os.access(override, os.X_OK):
        return override
    found = shutil.which("tailscale")
    if found:
        return found
    for candidate in (
        Path("/Applications/Tailscale.app/Contents/MacOS/Tailscale"),
        Path.home() / "Applications" / "Tailscale.app" / "Contents" / "MacOS" / "Tailscale",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise ApplyError("Tailscale CLI not found. Install: brew install --cask tailscale-app")


def run(cli: str, args: list[str], *, timeout: float = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [cli, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def wait_for_tailscale(cli: str) -> None:
    attempts = int(os.environ.get("TAILSCALE_WAIT_ATTEMPTS", "90"))
    delay = float(os.environ.get("TAILSCALE_WAIT_DELAY_SEC", "2"))
    for _ in range(attempts):
        proc = run(cli, ["status", "--json"], timeout=10)
        if proc.returncode == 0:
            try:
                state = json.loads(proc.stdout).get("BackendState", "")
            except json.JSONDecodeError:
                state = ""
            if state == "Running":
                return
        time.sleep(delay)
    raise ApplyError(f"Tailscale is not connected after {attempts * delay:.0f}s")


def load_dns_name(cli: str) -> str:
    proc = run(cli, ["status", "--json"], timeout=10)
    if proc.returncode != 0:
        raise ApplyError(proc.stderr or proc.stdout or "tailscale status failed")
    data = json.loads(proc.stdout)
    dns = str(data.get("Self", {}).get("DNSName", "")).rstrip(".")
    if not dns:
        raise ApplyError("Could not read MagicDNS name from tailscale status")
    return dns


def load_serve_status(cli: str) -> dict:
    proc = run(cli, ["serve", "status", "--json"], timeout=10)
    if proc.returncode != 0:
        return {}
    try:
        return json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return {}


def wait_for_backend(proxy: str) -> None:
    attempts = int(os.environ.get("BACKEND_WAIT_ATTEMPTS", "90"))
    delay = float(os.environ.get("BACKEND_WAIT_DELAY_SEC", "2"))
    health = proxy.rstrip("/") + "/v1/health"
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(health, timeout=2) as resp:
                if 200 <= resp.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(delay)
    raise ApplyError(f"Backend {proxy} did not become healthy in time")


def expand_host(host_key: str, dns_name: str) -> str:
    if host_key.startswith(":"):
        return f"{dns_name}{host_key}"
    if host_key.startswith(dns_name):
        return host_key
    return host_key


def protocol_for_port(tcp: dict, port: str) -> str:
    entry = tcp.get(port) or tcp.get(str(port)) or {}
    if entry.get("HTTP"):
        return "http"
    if entry.get("HTTPS"):
        return "https"
    return "https"


def handler_matches(status: dict, host: str, path: str, proxy: str) -> bool:
    handlers = status.get("Web", {}).get(host, {}).get("Handlers", {})
    return handlers.get(path, {}).get("Proxy") == proxy


def remove_legacy_https_443(cli: str, status: dict) -> None:
    if not status.get("TCP", {}).get("443"):
        return
    print(f"{LOG} Removing legacy HTTPS :443 handler")
    run(cli, ["serve", "--https=443", "off"], timeout=15)


def publish_route(cli: str, *, scheme: str, port: str, path: str, proxy: str, status: dict, host: str) -> None:
    wait_for_backend(proxy)
    if handler_matches(status, host, path, proxy):
        print(f"{LOG} {scheme.upper()} :{port}{path} already proxies {proxy}")
        return
    args = ["serve", "--bg", "--yes"]
    if scheme == "http":
        args.extend([f"--http={port}"])
    else:
        args.extend([f"--https={port}"])
    if path != "/":
        args.extend(["--set-path", path])
    args.append(proxy)
    print(f"{LOG} Publishing {scheme.upper()} :{port}{path} -> {proxy}")
    proc = run(cli, args, timeout=30)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "tailscale serve failed").strip()
        raise ApplyError(detail)


def apply_config(cli: str, config_path: Path) -> None:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    tcp = raw.get("TCP") or {}
    web = raw.get("Web") or {}
    dns_name = load_dns_name(cli)
    status = load_serve_status(cli)
    remove_legacy_https_443(cli, status)
    status = load_serve_status(cli)

    for host_key, host_cfg in web.items():
        host = expand_host(str(host_key), dns_name)
        port = host.rsplit(":", 1)[-1]
        scheme = protocol_for_port(tcp, port)
        handlers = host_cfg.get("Handlers") or {}
        for path, handler in sorted(handlers.items()):
            proxy = str(handler.get("Proxy") or "").strip()
            if not proxy:
                continue
            publish_route(
                cli,
                scheme=scheme,
                port=port,
                path=str(path),
                proxy=proxy,
                status=status,
                host=host,
            )
            status = load_serve_status(cli)

    print(f"{LOG} Done. Example: http://{dns_name}:8875/")
    proc = run(cli, ["serve", "status"], timeout=10)
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip(), file=sys.stderr)


def main() -> int:
    config = Path(os.environ.get("TAILSCALE_SERVE_CONFIG", str(DEFAULT_CONFIG)))
    if not config.is_file():
        print(f"{LOG} Config not found: {config}", file=sys.stderr)
        return 1
    try:
        cli = find_cli()
        wait_for_tailscale(cli)
        apply_config(cli, config)
    except ApplyError as exc:
        print(f"{LOG} {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
