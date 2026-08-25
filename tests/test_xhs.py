import json

import pytest

from media_pipeline.config import AppConfig
from media_pipeline.xhs import (
    _POST_CACHE,
    _image_urls,
    _note_from_html,
    _video_urls,
    canonicalize_xhs_url,
    extract_xhs_post,
    parse_xhs_ref,
    render_image_post,
)


NOTE_ID = "64aaaaaaaaaaaaaaaaaaaaaa"


@pytest.fixture(autouse=True)
def _clear_cache():
    _POST_CACHE.clear()
    yield
    _POST_CACHE.clear()


def _html_for(note: dict) -> str:
    state = {"note": {"noteDetailMap": {note["noteId"]: {"note": note}}}}
    return f"<html><script>window.__INITIAL_STATE__={json.dumps(state)}</script></html>"


def test_parse_xhs_urls():
    assert parse_xhs_ref("https://www.xiaohongshu.com/explore/" + NOTE_ID) == ("Xiaohongshu", NOTE_ID)
    assert parse_xhs_ref(f"https://www.xiaohongshu.com/discovery/item/{NOTE_ID}") == ("Xiaohongshu", NOTE_ID)
    assert parse_xhs_ref(f"https://www.xiaohongshu.com/user/profile/123/{NOTE_ID}") == ("Xiaohongshu", NOTE_ID)
    assert parse_xhs_ref("http://xhslink.com/o/2x5jqGA2hr6") == ("Xiaohongshu", "2x5jqGA2hr6")


def test_canonicalize_keeps_xsec_token():
    url = f"https://www.xiaohongshu.com/discovery/item/{NOTE_ID}?xsec_token=abc&xsec_source=pc_feed"
    canonical = canonicalize_xhs_url(url, NOTE_ID)
    assert canonical.startswith(f"https://www.xiaohongshu.com/explore/{NOTE_ID}?")
    assert "xsec_token=abc" in canonical
    assert canonicalize_xhs_url("http://xhslink.com/o/2x5jqGA2hr6") == "http://xhslink.com/o/2x5jqGA2hr6"


def test_parse_video_note_from_initial_state():
    note = {
        "noteId": NOTE_ID,
        "type": "video",
        "title": "A clip",
        "desc": "hello",
        "time": 1700000000000,
        "user": {"nickname": "Author"},
        "imageList": [
            {
                "urlDefault": "http://sns-webpic-qc.xhscdn.com/20240101/hash/notes_pre_post/coverid!nd",
            }
        ],
        "video": {
            "capa": {"duration": 12},
            "consumer": {"originVideoKey": "notes_pre_post/videokey"},
            "media": {
                "stream": {
                    "h264": [
                        {
                            "height": 720,
                            "masterUrl": "https://sns-video-bd.xhscdn.com/h264",
                            "backupUrls": ["https://backup.example/v.mp4"],
                        }
                    ]
                }
            },
        },
    }
    parsed = _note_from_html(_html_for(note))
    assert parsed is not None
    assert parsed["noteId"] == NOTE_ID
    urls = _video_urls(parsed)
    assert urls[0] == "https://sns-video-bd.xhscdn.com/notes_pre_post/videokey"
    assert "https://backup.example/v.mp4" in urls


def test_parse_image_note_and_classify_video_album_as_image():
    images = [
        {"urlDefault": "http://sns-webpic-qc.xhscdn.com/20240101/h/notes_pre_post/one!nd"},
        {"urlDefault": "http://sns-webpic-qc.xhscdn.com/20240101/h/notes_pre_post/two!nd"},
    ]
    urls = _image_urls(images)
    assert urls == [
        "https://ci.xiaohongshu.com/notes_pre_post/one?imageView2/format/jpeg",
        "https://ci.xiaohongshu.com/notes_pre_post/two?imageView2/format/jpeg",
    ]
    album = {
        "noteId": NOTE_ID,
        "type": "video",
        "title": "album",
        "desc": "",
        "imageList": images,
        "user": {},
        "video": {"consumer": {"originVideoKey": "notes_pre_post/videokey"}},
    }
    parsed = _note_from_html(_html_for(album))
    assert parsed is not None
    assert len(parsed["imageList"]) == 2


def test_extract_uses_html_and_caches(monkeypatch):
    note = {
        "noteId": NOTE_ID,
        "type": "normal",
        "title": "Photos",
        "desc": "a walk",
        "time": 1700000000000,
        "user": {"nickname": "Nana"},
        "imageList": [
            {"urlDefault": "http://sns-webpic-qc.xhscdn.com/20240101/h/notes_pre_post/one!nd"},
            {"urlDefault": "http://sns-webpic-qc.xhscdn.com/20240101/h/notes_pre_post/two!nd"},
        ],
    }

    class FakeResp:
        status_code = 200
        text = _html_for(note)
        url = f"https://www.xiaohongshu.com/explore/{NOTE_ID}?xsec_token=tok"
        headers = {}
        content = b""

    calls = {"n": 0}

    def fake_get(url, config, stream=False):
        calls["n"] += 1
        return FakeResp()

    monkeypatch.setattr("media_pipeline.xhs._http_get", fake_get)
    monkeypatch.setattr("media_pipeline.xhs._extract_with_xhs_downloader", lambda url: None)
    post = extract_xhs_post("http://xhslink.com/o/21VSXCZSZ9s", AppConfig())
    assert post.media_kind == "image"
    assert post.note_id == NOTE_ID
    assert post.author == "Nana"
    assert len(post.download_urls) == 2
    extract_xhs_post("http://xhslink.com/o/21VSXCZSZ9s", AppConfig())
    extract_xhs_post(f"https://www.xiaohongshu.com/explore/{NOTE_ID}", AppConfig())
    assert calls["n"] == 1


def test_render_image_post():
    body = render_image_post(NOTE_ID, "a walk", ["01.jpg", "02.jpg"])
    assert "## Post" in body
    assert "a walk" in body
    assert f"![[attachments/{NOTE_ID}/01.jpg]]" in body
    assert f"![[attachments/{NOTE_ID}/02.jpg]]" in body
