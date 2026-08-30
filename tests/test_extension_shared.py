import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "extension" / "shared.js"
BACKGROUND = ROOT / "extension" / "background.js"
MANIFEST = ROOT / "extension" / "manifest.json"
NOTE_ID = "64aaaaaaaaaaaaaaaaaaaaaa"


def test_extension_defaults_to_api_port_8875():
    background = BACKGROUND.read_text(encoding="utf-8")
    manifest = MANIFEST.read_text(encoding="utf-8")
    assert 'const DEFAULT_BASE = "http://127.0.0.1:8875"' in background
    assert '"http://127.0.0.1:8765"' in background  # migrated away at runtime
    assert "/v1/health" in background
    assert "http://127.0.0.1:8875/*" in manifest
    assert '"http://127.0.0.1/*"' in manifest


def test_floating_overlay_not_injected():
    content = (ROOT / "extension" / "content.js").read_text(encoding="utf-8")
    assert "No floating panel" in content
    assert "existing?.remove();" in content
    assert "watchVideoCards();" in content
    assert "scanVideoCards();" in content
    assert "scanYoutubeCards();" in content
    assert "scanBilibiliCards();" in content
    assert 'host.classList.add("mdp-floating");' not in content


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
  if (item.fn === "ytHost") return ctx.mdpIsYoutubeHost(item.url);
  if (item.fn === "biliHost") return ctx.mdpIsBilibiliHost(item.url);
  if (item.fn === "parseYt") return ctx.mdpParseYoutubeVideo(item.url, item.base);
  if (item.fn === "parseBili") return ctx.mdpParseBilibiliVideo(item.url, item.base);
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


def test_extension_parses_youtube_card_links():
    watch, shorts, live, feed_host, feed_parsed = _eval(
        [
            {"fn": "parseYt", "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
            {"fn": "parseYt", "url": "https://www.youtube.com/shorts/abc123_-XYZ"},
            {"fn": "parseYt", "url": "https://www.youtube.com/live/abc123_-XYZ"},
            {"fn": "ytHost", "url": "https://www.youtube.com/feed/subscriptions"},
            {"fn": "parseYt", "url": "/watch?v=dQw4w9WgXcQ", "base": "https://www.youtube.com/results?search_query=test"},
        ]
    )
    assert watch["videoId"] == "dQw4w9WgXcQ"
    assert watch["url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert shorts["videoId"] == "abc123_-XYZ"
    assert live["videoId"] == "abc123_-XYZ"
    assert feed_host is True
    assert feed_parsed["videoId"] == "dQw4w9WgXcQ"


def test_extension_parses_bilibili_card_links():
    video, watchlater, av, feed_host = _eval(
        [
            {"fn": "parseBili", "url": "https://www.bilibili.com/video/BV181KNeuEi2/?spm_id_from=333"},
            {
                "fn": "parseBili",
                "url": "https://www.bilibili.com/list/watchlater/?bvid=BV1fP4y1j76Q&oid=893222135",
            },
            {"fn": "parseBili", "url": "https://www.bilibili.com/video/av893222135"},
            {"fn": "biliHost", "url": "https://www.bilibili.com/"},
        ]
    )
    assert video["videoId"] == "BV181KNeuEi2"
    assert video["url"] == "https://www.bilibili.com/video/BV181KNeuEi2"
    assert watchlater["videoId"] == "BV1fP4y1j76Q"
    assert av["videoId"] == "av893222135"
    assert feed_host is True
