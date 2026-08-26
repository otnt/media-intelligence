from types import SimpleNamespace

from media_pipeline.tailscale import (
    backend_from_status,
    https_dashboard_url,
    publish_local_http,
    serve_background_argsets,
    url_from_serve_status_json,
)


def test_backend_from_status_picks_ipv4_and_strips_dns_dot():
    backend = backend_from_status(
        {
            "BackendState": "Running",
            "Self": {
                "DNSName": "mac.tail08155b.ts.net.",
                "TailscaleIPs": ["100.64.1.2", "fd7a:115c::1"],
            },
        }
    )
    assert backend.state == "Running"
    assert backend.dns_name == "mac.tail08155b.ts.net"
    assert backend.ipv4 == "100.64.1.2"


def test_https_dashboard_url_from_stdout_and_dns():
    assert https_dashboard_url("https://mac.tail08155b.ts.net/", "") == "https://mac.tail08155b.ts.net/"
    assert https_dashboard_url("ready https://mac.tail08155b.ts.net", "") == "https://mac.tail08155b.ts.net/"
    assert https_dashboard_url("", "mac.tail08155b.ts.net") == "https://mac.tail08155b.ts.net/"


def test_url_from_serve_status_json():
    url = url_from_serve_status_json(
        {"Web": {"https://mac.tail08155b.ts.net:443": {}}},
        "ignored.example.ts.net",
    )
    assert url == "https://mac.tail08155b.ts.net/"


def test_serve_argsets_never_enable_funnel():
    for argv in serve_background_argsets(8765):
        joined = " ".join(argv).lower()
        assert "funnel" not in joined
        assert "--bg" in argv


def test_publish_reports_missing_cli(monkeypatch):
    monkeypatch.setattr("media_pipeline.tailscale.find_cli", lambda: None)
    result = publish_local_http(8765)
    assert result.ok is False
    assert "not found" in result.detail.lower()


def test_publish_reports_login_needed(monkeypatch):
    from media_pipeline.tailscale import TailscaleError

    monkeypatch.setattr("media_pipeline.tailscale.find_cli", lambda: "/usr/bin/tailscale")

    def boom(cli, **kwargs):
        raise TailscaleError("Tailscale is NeedsLogin. Sign in with: /usr/bin/tailscale up")

    monkeypatch.setattr("media_pipeline.tailscale.ensure_backend", boom)
    result = publish_local_http(8765, attempts=1, delay=0)
    assert result.ok is False
    assert "NeedsLogin" in result.detail
    assert result.cli == "/usr/bin/tailscale"


def test_publish_starts_serve_without_funnel(monkeypatch):
    from media_pipeline.tailscale import TailscaleBackend

    monkeypatch.setattr("media_pipeline.tailscale.find_cli", lambda: "/usr/bin/tailscale")
    monkeypatch.setattr(
        "media_pipeline.tailscale.ensure_backend",
        lambda cli, **kwargs: TailscaleBackend(
            state="Running",
            dns_name="mac.tail08155b.ts.net",
            ipv4="100.64.1.2",
        ),
    )

    def fake_run(cli, args, timeout=30):
        assert "funnel" not in args
        joined = " ".join(args)
        if args[:2] == ["serve", "--bg"]:
            return SimpleNamespace(returncode=0, stdout="https://mac.tail08155b.ts.net/\n", stderr="")
        raise AssertionError(f"unexpected tailscale args: {joined}")

    monkeypatch.setattr("media_pipeline.tailscale._run", fake_run)
    result = publish_local_http(8765, attempts=1, delay=0)
    assert result.ok is True
    assert result.url == "https://mac.tail08155b.ts.net/"
    assert result.ipv4 == "100.64.1.2"


def test_publish_surfaces_serve_not_enabled(monkeypatch):
    from media_pipeline.tailscale import TailscaleBackend

    monkeypatch.setattr("media_pipeline.tailscale.find_cli", lambda: "/usr/bin/tailscale")
    monkeypatch.setattr(
        "media_pipeline.tailscale.ensure_backend",
        lambda cli, **kwargs: TailscaleBackend(
            state="Running",
            dns_name="mac.tail08155b.ts.net",
            ipv4="100.64.1.2",
        ),
    )
    calls: list[list[str]] = []

    def fake_run(cli, args, timeout=30):
        calls.append(args)
        return SimpleNamespace(
            returncode=1,
            stdout="Serve is not enabled on your tailnet.\nTo enable, visit:\nhttps://login.tailscale.com/f/serve?node=example\n",
            stderr="",
        )

    monkeypatch.setattr("media_pipeline.tailscale._run", fake_run)
    result = publish_local_http(8765, attempts=1, delay=0)
    assert result.ok is False
    assert "Serve is not enabled" in result.detail
    assert len(calls) == 1
    assert calls[0][-1] == "8765"
