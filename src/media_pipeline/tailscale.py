"""Publish the local dashboard on a Tailscale tailnet.

Uses Tailscale Serve so the API can stay on 127.0.0.1. Other devices on the
same tailnet open an HTTPS MagicDNS URL. Funnel (public internet) is never
enabled here.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

MAC_APP_CLI = Path("/Applications/Tailscale.app/Contents/MacOS/Tailscale")
USER_APP_CLI = Path.home() / "Applications" / "Tailscale.app" / "Contents" / "MacOS" / "Tailscale"
MISSING_CLI = (
    "Tailscale CLI not found. Install the Mac app with: brew install --cask tailscale-app"
)
LOGIN_HINT = (
    "Open the Tailscale app and sign in, or run: tailscale up. "
    "Then retry: media-pipeline serve --tailscale"
)
_HTTPS_RE = re.compile(r"https://[A-Za-z0-9.-]+\.ts\.net/?")


@dataclass(frozen=True)
class TailscaleBackend:
    state: str
    dns_name: str
    ipv4: str | None


@dataclass(frozen=True)
class PublishResult:
    ok: bool
    detail: str
    url: str = ""
    ipv4: str | None = None
    dns_name: str = ""
    cli: str = ""


class TailscaleError(RuntimeError):
    pass


def find_cli() -> str | None:
    found = shutil.which("tailscale")
    if found:
        return found
    for path in (
        MAC_APP_CLI,
        USER_APP_CLI,
        Path("/opt/homebrew/bin/tailscale"),
        Path("/usr/local/bin/tailscale"),
    ):
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


def local_http_target(port: int) -> str:
    return f"http://127.0.0.1:{int(port)}"


def serve_background_argsets(port: int) -> list[list[str]]:
    """Candidate argv lists for `tailscale serve`. None of these enable Funnel."""
    port_s = str(int(port))
    target = local_http_target(port)
    return [
        ["serve", "--bg", "--yes", port_s],
        ["serve", "--bg", port_s],
        ["serve", "--bg", "--yes", target],
        ["serve", "--bg", target],
    ]


def login_detail(cli: str, state: str) -> str:
    quoted = cli if " " not in cli else f'"{cli}"'
    return (
        f"Tailscale is {state}. Sign in with: {quoted} up\n"
        "Or open the Tailscale app on this Mac, then retry. "
        "Install Tailscale on your phone and use the same account. Do not enable Funnel."
    )


def _run(cli: str, args: list[str], timeout: float = 30) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [cli, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return subprocess.CompletedProcess(
            args=[cli, *args],
            returncode=1,
            stdout=stdout,
            stderr=stderr or f"timed out after {timeout:.0f}s",
        )


def load_backend(cli: str) -> TailscaleBackend:
    proc = _run(cli, ["status", "--json"], timeout=10)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "tailscale status failed").strip()
        raise TailscaleError(err)
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise TailscaleError(f"tailscale status is not JSON: {exc}") from exc
    return backend_from_status(data)


def backend_from_status(data: dict) -> TailscaleBackend:
    self = data.get("Self") or {}
    dns_name = str(self.get("DNSName") or "").rstrip(".")
    ipv4 = None
    for ip in self.get("TailscaleIPs") or []:
        if isinstance(ip, str) and ip and ":" not in ip:
            ipv4 = ip
            break
    return TailscaleBackend(
        state=str(data.get("BackendState") or ""),
        dns_name=dns_name,
        ipv4=ipv4,
    )


def https_dashboard_url(stdout: str, dns_name: str) -> str:
    match = _HTTPS_RE.search(stdout or "")
    if match:
        url = match.group(0)
        return url if url.endswith("/") else f"{url}/"
    if dns_name:
        return f"https://{dns_name}/"
    return ""


def url_from_serve_status_json(data: dict, dns_name: str = "") -> str:
    web = data.get("Web") if isinstance(data, dict) else None
    if isinstance(web, dict):
        for key in web:
            host = str(key).split("://")[-1].split("/")[0].split(":")[0].rstrip(".")
            if host:
                return f"https://{host}/"
    return https_dashboard_url("", dns_name)


def _start_mac_app() -> None:
    for app in (
        Path("/Applications/Tailscale.app"),
        Path.home() / "Applications" / "Tailscale.app",
    ):
        if app.is_dir():
            subprocess.run(
                ["open", "-a", str(app)],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return


def ensure_backend(cli: str, *, attempts: int = 5, delay: float = 1.0) -> TailscaleBackend:
    """Load Tailscale status, starting the Mac app if the daemon is down.

    Does not run `tailscale up` (that can hang on an auth URL). NeedsLogin fails fast.
    """
    last_error = "Tailscale is not running"
    started = False
    for attempt in range(max(1, attempts)):
        try:
            backend = load_backend(cli)
        except (TailscaleError, OSError, subprocess.TimeoutExpired) as exc:
            last_error = str(exc)
            if not started:
                _start_mac_app()
                started = True
            if attempt < attempts - 1 and delay > 0:
                time.sleep(delay)
            continue
        if backend.state == "Running":
            return backend
        if backend.state == "NeedsLogin":
            raise TailscaleError(login_detail(cli, backend.state))
        last_error = login_detail(cli, backend.state or "not running")
        if not started:
            _start_mac_app()
            started = True
        if attempt < attempts - 1 and delay > 0:
            time.sleep(delay)
    raise TailscaleError(last_error)


def _combined_output(proc: subprocess.CompletedProcess[str]) -> str:
    return f"{proc.stdout or ''}\n{proc.stderr or ''}"


def _unknown_flag(text: str) -> bool:
    lower = text.lower()
    return "unknown flag" in lower or "flag provided but not defined" in lower


def _fatal_serve_error(text: str) -> bool:
    lower = text.lower()
    return any(
        needle in lower
        for needle in (
            "logged out",
            "not running",
            "needs login",
            "access denied",
            "failed to connect",
            "no current user",
            "serve is not enabled",
            "to enable, visit",
        )
    )


def _run_serve(cli: str, port: int) -> subprocess.CompletedProcess[str]:
    last = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="tailscale serve failed")
    for argv in serve_background_argsets(port):
        last = _run(cli, argv, timeout=8)
        if last.returncode == 0:
            return last
        output = _combined_output(last)
        if _fatal_serve_error(output):
            return last
        if "--yes" in argv and _unknown_flag(output):
            continue
        if argv[-1].startswith("http://") and ("usage:" in output.lower() or "invalid" in output.lower()):
            continue
        return last
    return last


def load_serve_url(cli: str, dns_name: str) -> str:
    text_proc = _run(cli, ["serve", "status"], timeout=10)
    url = https_dashboard_url(_combined_output(text_proc), dns_name)
    if url:
        return url
    json_proc = _run(cli, ["serve", "status", "--json"], timeout=10)
    if json_proc.returncode != 0:
        return https_dashboard_url("", dns_name)
    try:
        data = json.loads(json_proc.stdout or "{}")
    except json.JSONDecodeError:
        return https_dashboard_url(json_proc.stdout or "", dns_name)
    return url_from_serve_status_json(data, dns_name)


def publish_local_http(port: int, *, attempts: int = 5, delay: float = 1.0) -> PublishResult:
    """Point Tailscale Serve at a local HTTP port (background, tailnet only)."""
    cli = find_cli()
    if not cli:
        return PublishResult(ok=False, detail=MISSING_CLI)
    try:
        backend = ensure_backend(cli, attempts=attempts, delay=delay)
    except (TailscaleError, OSError, subprocess.TimeoutExpired) as exc:
        return PublishResult(ok=False, detail=str(exc), cli=cli)
    try:
        proc = _run_serve(cli, port)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return PublishResult(
            ok=False,
            detail=str(exc),
            cli=cli,
            dns_name=backend.dns_name,
            ipv4=backend.ipv4,
        )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "tailscale serve failed").strip()
        if "funnel" in detail.lower() and "access denied" in detail.lower():
            detail = f"{detail}\nThis command never enables Funnel; check Tailscale Serve ACL on the tailnet."
        return PublishResult(
            ok=False,
            detail=detail,
            cli=cli,
            dns_name=backend.dns_name,
            ipv4=backend.ipv4,
        )
    url = https_dashboard_url(_combined_output(proc), backend.dns_name) or load_serve_url(cli, backend.dns_name)
    if not url:
        return PublishResult(
            ok=False,
            detail="tailscale serve started but no MagicDNS URL was returned",
            cli=cli,
            dns_name=backend.dns_name,
            ipv4=backend.ipv4,
        )
    return PublishResult(
        ok=True,
        detail="ok",
        url=url,
        ipv4=backend.ipv4,
        dns_name=backend.dns_name,
        cli=cli,
    )


def doctor_status() -> tuple[bool, str]:
    cli = find_cli()
    if not cli:
        return False, "not found; brew install --cask tailscale-app, then serve --tailscale"
    try:
        backend = load_backend(cli)
    except (TailscaleError, OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if backend.state != "Running":
        state = backend.state or "not running"
        return False, f"{state}; sign in with: {cli} up"
    if backend.dns_name and backend.ipv4:
        return True, f"{backend.dns_name} ({backend.ipv4})"
    if backend.dns_name:
        return True, backend.dns_name
    return True, backend.ipv4 or "running"


def doctor_serve_status() -> tuple[bool, str]:
    cli = find_cli()
    if not cli:
        return False, "CLI missing; serve --tailscale after installing Tailscale"
    try:
        backend = load_backend(cli)
    except (TailscaleError, OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if backend.state != "Running":
        state = backend.state or "not running"
        return False, f"{state}; not publishing"
    text_proc = _run(cli, ["serve", "status"], timeout=10)
    text = _combined_output(text_proc).strip()
    lower = text.lower()
    if text_proc.returncode != 0:
        return False, text or "serve status failed"
    if not text or "no serve" in lower or "not configured" in lower:
        return False, "not publishing; run media-pipeline serve --tailscale"
    url = https_dashboard_url(text, backend.dns_name)
    if not url:
        json_proc = _run(cli, ["serve", "status", "--json"], timeout=10)
        try:
            data = json.loads(json_proc.stdout or "{}")
        except json.JSONDecodeError:
            data = {}
        url = url_from_serve_status_json(data, backend.dns_name)
    if not url:
        return False, text.splitlines()[0][:200]
    return True, url


def doctor_checks() -> list[tuple[str, bool, str]]:
    return [
        ("tailscale", *doctor_status()),
        ("tailscale serve", *doctor_serve_status()),
    ]
