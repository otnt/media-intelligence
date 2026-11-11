import json
from pathlib import Path

import pytest

from media_pipeline.media import UnsupportedURLError, canonicalize_url, extract_supported_url, parse_video_ref
from media_pipeline.models import source_key, source_label
from media_pipeline.threads import ThreadsError, parse_threads_page

FIXTURES = Path(__file__).parent / "fixtures"


def test_threads_urls_are_recognized_and_cleaned():
    assert parse_video_ref("https://www.threads.com/@essenhwv/post/Dfj5fhollvJ?xmt=abc&slof=1") == ("Threads", "th_Dfj5fhollvJ")
    assert parse_video_ref("https://www.threads.net/@a.b/post/Xyz_-123") == ("Threads", "th_Xyz_-123")
    assert parse_video_ref("https://www.threads.com/share/_8NWxZ8gy") == ("Threads", "th_share__8NWxZ8gy")
    assert canonicalize_url("https://www.threads.com/@essenhwv/post/Dfj5fhollvJ?xmt=abc") == \
        "https://www.threads.com/@essenhwv/post/Dfj5fhollvJ"
    shared = "看看这个 https://www.threads.com/share/_8NWxZ8gy 很有意思"
    assert extract_supported_url(shared) == "https://www.threads.com/share/_8NWxZ8gy"
    assert source_key("Threads", "") == "threads" and source_label("", "https://www.threads.com/x") == "Threads"
    with pytest.raises(UnsupportedURLError):
        parse_video_ref("https://www.threads.com/@essenhwv")  # a profile, not a post


def test_real_text_post_page_ignores_recommended_posts():
    html = (FIXTURES / "threads_post_text.html").read_text()
    post = parse_threads_page(html, "https://www.threads.com/@essenhwv/post/Dfj5fhollvJ?xmt=x")
    assert post.video_id == "th_Dfj5fhollvJ" and post.username == "essenhwv"
    assert post.text.startswith("互联网最暴利的生意之一") and "边际成本为 0" in post.text
    assert post.media_kind == "image" and post.image_urls == [] and post.published.startswith("2026-")
    assert post.url == "https://www.threads.com/@essenhwv/post/Dfj5fhollvJ"


def _page(*posts, chain=None):
    data = {"thread_items": [{"post": p} for p in (chain or posts)]}
    return '<script type="application/json" data-sjs>' + json.dumps({"x": [data, {"more": list(posts)}]}) + "</script>"


def _post(code, user="u1", text="t", **extra):
    return {"code": code, "caption": {"text": text}, "user": {"pk": user, "username": user, "full_name": user.upper()},
            "taken_at": 1794313423, **extra}


def test_carousel_video_and_author_continuation():
    image = {"image_versions2": {"candidates": [{"url": "small", "width": 100, "height": 100},
                                                {"url": "big", "width": 1080, "height": 1350}]}}
    video = {"video_versions": [{"url": "https://cdn/v.mp4"}], "video_duration": 61.5}
    carousel = _post("C1", text="第一条", carousel_media=[image, image])
    reply = _post("C2", text="接着说", **image)
    stranger = _post("C3", user="someone", text="路人评论")
    post = parse_threads_page(_page(chain=[carousel, reply, stranger]), "https://www.threads.com/@u1/post/C1")
    assert post.text == "第一条\n\n接着说" and post.image_urls == ["big", "big", "big"]
    clip = parse_threads_page(_page(_post("V1", **video)), "https://www.threads.com/@u1/post/V1")
    assert clip.media_kind == "video" and clip.video_url == "https://cdn/v.mp4" and clip.duration == 61.5
    with pytest.raises(ThreadsError):
        parse_threads_page(_page(_post("other")), "https://www.threads.com/@u1/post/MISSING")
