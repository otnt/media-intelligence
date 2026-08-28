from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPLY_SCRIPT = ROOT / "scripts" / "tailscale" / "apply-serve.py"
EXAMPLE_CONFIG = ROOT / "scripts" / "tailscale" / "serve.example.json"


def _load_apply_module():
    spec = importlib.util.spec_from_file_location("apply_serve", APPLY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["apply_serve"] = module
    spec.loader.exec_module(module)
    return module


def test_expand_host_placeholder():
    mod = _load_apply_module()
    assert mod.expand_host(":8875", "macbook-pro.tail08155b.ts.net") == (
        "macbook-pro.tail08155b.ts.net:8875"
    )


def test_protocol_for_port_http():
    mod = _load_apply_module()
    tcp = {"8875": {"HTTP": True}}
    assert mod.protocol_for_port(tcp, "8875") == "http"


def test_handler_matches():
    mod = _load_apply_module()
    status = {
        "Web": {
            "mac.example.ts.net:8875": {
                "Handlers": {"/": {"Proxy": "http://127.0.0.1:8875"}}
            }
        }
    }
    assert mod.handler_matches(status, "mac.example.ts.net:8875", "/", "http://127.0.0.1:8875")
    assert not mod.handler_matches(status, "mac.example.ts.net:8875", "/", "http://127.0.0.1:3000")


def test_default_config_path_is_user_config_dir():
    mod = _load_apply_module()
    assert mod.DEFAULT_CONFIG == mod.CONFIG_DIR / "serve.json"
    assert mod.CONFIG_DIR.name == "tailscale"


def test_serve_example_json_is_valid():
    raw = json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert raw["TCP"]["8875"]["HTTP"] is True
    assert raw["Web"][":8875"]["Handlers"]["/"]["Proxy"] == "http://127.0.0.1:8875"
