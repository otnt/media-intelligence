from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.apply_tailscale_serve import expand_host, handler_matches, protocol_for_port


def test_expand_host_placeholder():
    assert expand_host(":8875", "macbook-pro.tail08155b.ts.net") == (
        "macbook-pro.tail08155b.ts.net:8875"
    )


def test_protocol_for_port_http():
    tcp = {"8875": {"HTTP": True}}
    assert protocol_for_port(tcp, "8875") == "http"


def test_handler_matches():
    status = {
        "Web": {
            "mac.example.ts.net:8875": {
                "Handlers": {"/": {"Proxy": "http://127.0.0.1:8875"}}
            }
        }
    }
    assert handler_matches(status, "mac.example.ts.net:8875", "/", "http://127.0.0.1:8875")
    assert not handler_matches(status, "mac.example.ts.net:8875", "/", "http://127.0.0.1:3000")


def test_tailscale_serve_json_is_valid():
    raw = json.loads(
        (Path(__file__).resolve().parents[1] / "scripts" / "tailscale-serve.json").read_text(
            encoding="utf-8"
        )
    )
    assert raw["TCP"]["8875"]["HTTP"] is True
    assert raw["Web"][":8875"]["Handlers"]["/"]["Proxy"] == "http://127.0.0.1:8875"
