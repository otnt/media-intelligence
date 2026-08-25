import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "extension" / "shared.js"
NOTE_ID = "64aaaaaaaaaaaaaaaaaaaaaa"


def _eval(cases: list[dict]) -> list:
    if not shutil.which("node"):
        pytest.skip("node is required to test extension URL helpers")
    script = """
const fs = require("fs");
const vm = require("vm");
const ctx = { URL, console };
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), ctx);
const cases = JSON.parse(process.argv[2]);
const out = cases.map((item) => {
  if (item.fn === "parse") return ctx.mdpParseXiaohongshuNote(item.url);
  if (item.fn === "supported") return ctx.mdpIsSupportedVideoUrl(item.url);
  if (item.fn === "host") return ctx.mdpIsXiaohongshuHost(item.url);
  if (item.fn === "tokens") return ctx.mdpCollectXhsTokens(item.state, item.scripts || []);
  return null;
});
console.log(JSON.stringify(out));
"""
    result = subprocess.run(
        ["node", "-e", script, str(SHARED), json.dumps(cases)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_extension_parses_feed_note_links_and_keeps_xsec_token():
    url = f"https://www.xiaohongshu.com/explore/{NOTE_ID}?xsec_token=abc&xsec_source=pc_feed"
    parsed, feed_supported, feed_host, relative = _eval(
        [
            {"fn": "parse", "url": url},
            {"fn": "supported", "url": "https://www.xiaohongshu.com/explore"},
            {"fn": "host", "url": "https://www.xiaohongshu.com/explore"},
            {"fn": "parse", "url": f"/explore/{NOTE_ID}?xsec_token=abc&xsec_source=pc_feed"},
        ]
    )
    assert parsed["noteId"] == NOTE_ID
    assert parsed["token"] == "abc"
    assert "xsec_token=abc" in parsed["url"]
    assert feed_supported is False
    assert feed_host is True
    assert relative["noteId"] == NOTE_ID
    assert relative["token"] == "abc"


def test_extension_parses_rednote_feed_links():
    url = f"https://www.rednote.com/explore/{NOTE_ID}?xsec_token=abc&xsec_source=pc_feed"
    parsed, feed_supported, feed_host = _eval(
        [
            {"fn": "parse", "url": url},
            {"fn": "supported", "url": "https://www.rednote.com/"},
            {"fn": "host", "url": "https://www.rednote.com/explore"},
        ]
    )
    assert parsed["noteId"] == NOTE_ID
    assert parsed["token"] == "abc"
    assert feed_supported is False
    assert feed_host is True


def test_extension_collects_feed_xsec_tokens_from_state():
    collected = _eval(
        [
            {
                "fn": "tokens",
                "state": {
                    "feed": {
                        "feeds": [
                            {
                                "id": NOTE_ID,
                                "modelType": "note",
                                "xsecToken": "from-state",
                                "xsecSource": "pc_feed",
                                "noteCard": {"noteId": NOTE_ID},
                            }
                        ]
                    }
                },
            }
        ]
    )[0]
    assert collected[NOTE_ID]["token"] == "from-state"
    assert collected[NOTE_ID]["source"] == "pc_feed"
