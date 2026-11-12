import html
import json

import pytest

from media_pipeline import linkedin
from media_pipeline.config import AppConfig
from media_pipeline.linkedin import LinkedInError, canonicalize_linkedin_url, parse_linkedin_page, parse_linkedin_ref
from media_pipeline.media import UnsupportedURLError, extract_supported_url, fetch_metadata, parse_video_ref
from media_pipeline.models import source_key, source_label

ACTIVITY_URL = "https://www.linkedin.com/posts/ada-l_building-agents-activity-7526345581150195712-QKwo"


def _page(ld: dict, og_url: str = ACTIVITY_URL, extra: str = "") -> str:
    return (
        f'<html><head><meta property="og:url" content="{og_url}">'
        '<meta property="og:image" content="https://static.licdn.com/aero-v1/sc/h/card">'
        f'<script type="application/ld+json">{json.dumps(ld)}</script></head><body>{extra}</body></html>'
    )


TEXT_POST = {
    "@type": "SocialMediaPosting",
    "articleBody": "If you're building with agents, read this.\n\nTwo cool things here. Check it out: https://lnkd.in/gAbc123",
    "author": {"@type": "Person", "name": "Ada Lovelace"},
    "datePublished": "2026-11-11T18:10:19.284Z",
    "sharedContent": {"@type": "WebPage", "author": {"name": "Grace Hopper"},
                      "headline": "We just launched an emulator, built entirely by agents."},
}


def test_post_share_and_feed_links_parse_to_ids():
    assert parse_video_ref("https://lnkd.in/p/gkHgqR-n") == ("LinkedIn", "li_lnkd_gkHgqR-n")
    share = ("https://www.linkedin.com/posts/ada-l_building-agents-share-7526345580344856576-Lx3s/"
             "?utm_source=share&utm_medium=member_ios&rcm=ACoAAB")
    assert parse_linkedin_ref(share) == ("LinkedIn", "li_share_7526345580344856576")
    assert parse_linkedin_ref(ACTIVITY_URL) == ("LinkedIn", "li_7526345581150195712")
    assert parse_linkedin_ref("https://www.linkedin.com/feed/update/urn:li:activity:7526345581150195712/") == (
        "LinkedIn", "li_7526345581150195712")
    assert canonicalize_linkedin_url(share) == (
        "https://www.linkedin.com/posts/ada-l_building-agents-share-7526345580344856576-Lx3s/")
    with pytest.raises(UnsupportedURLError):
        parse_video_ref("https://www.linkedin.com/in/someone")
    blob = "Worth a read 👇 https://lnkd.in/p/gkHgqR-n (shared from LinkedIn)"
    assert extract_supported_url(blob) == "https://lnkd.in/p/gkHgqR-n"
    assert source_key("LinkedIn") == "linkedin" and source_label("", "https://lnkd.in/p/x") == "LinkedIn"


def test_a_reshare_keeps_both_posts_and_skips_the_generic_card_image():
    post = parse_linkedin_page(_page(TEXT_POST), "https://www.linkedin.com/posts/x-share-1-Lx3s")
    assert post.video_id == "li_7526345581150195712" and post.url == ACTIVITY_URL
    assert post.author == "Ada Lovelace" and post.published == "2026-11-11"
    assert post.text.startswith("If you're building with agents, read this.")
    assert post.text.endswith("转发自 Grace Hopper：\nWe just launched an emulator, built entirely by agents.")
    assert post.media_kind == "image" and post.image_urls == []


def test_video_posts_take_the_best_mp4_and_their_duration():
    sources = [{"src": "https://dms.licdn.com/playlist/vid/low.mp4", "type": "video/mp4", "data-bitrate": 400000},
               {"src": "https://dms.licdn.com/playlist/vid/high.mp4", "type": "video/mp4", "data-bitrate": 2000000}]
    player = f'<video data-sources="{html.escape(json.dumps(sources))}"></video>'
    video = {"@type": "VideoObject", "description": "A demo of the emulator.", "author": {"name": "Grace Hopper"},
             "uploadDate": "2026-11-10", "duration": "PT1M30S", "thumbnailUrl": "https://media.licdn.com/thumb.jpg"}
    post = parse_linkedin_page(_page(video, extra=player), ACTIVITY_URL)
    assert post.media_kind == "video" and post.video_url.endswith("high.mp4") and post.duration == 90.0
    assert post.text == "A demo of the emulator." and post.thumbnail_url == "https://media.licdn.com/thumb.jpg"


def test_image_posts_list_their_images():
    ld = dict(TEXT_POST, image=[{"@type": "ImageObject", "url": "https://media.licdn.com/dms/image/a.jpg"},
                                {"@type": "ImageObject", "url": "https://media.licdn.com/dms/image/b.jpg"}])
    post = parse_linkedin_page(_page(ld), ACTIVITY_URL)
    assert post.image_urls == ["https://media.licdn.com/dms/image/a.jpg", "https://media.licdn.com/dms/image/b.jpg"]


def test_a_login_wall_is_a_clear_error():
    with pytest.raises(LinkedInError, match="may need a login"):
        parse_linkedin_page("<html><title>Sign in</title></html>", ACTIVITY_URL)
    with pytest.raises(LinkedInError, match="did not open a post"):
        parse_linkedin_page("<html></html>", "https://www.linkedin.com/feed/")


def test_metadata_resolves_links_inside_the_text(monkeypatch):
    linkedin._CACHE.clear()
    monkeypatch.setattr(linkedin, "_fetch", lambda url: (_page(TEXT_POST), ACTIVITY_URL + "?rcm=x"))
    monkeypatch.setattr(linkedin, "_redirect_target", lambda url: "https://aws.amazon.com/blogs/emulator")
    metadata = fetch_metadata("https://lnkd.in/p/gkHgqR-n", "fun-asr", AppConfig())
    assert metadata.platform == "LinkedIn" and metadata.video_id == "li_7526345581150195712"
    assert metadata.media_kind == "image" and metadata.title == "If you're building with agents, read this."
    assert "https://lnkd.in/gAbc123（→ https://aws.amazon.com/blogs/emulator）" in metadata.description
    assert metadata.url == ACTIVITY_URL
    linkedin._CACHE.clear()



def test_short_links_resolve_through_the_interstitial_page(monkeypatch):
    class Response:
        headers: dict = {}
        text = ('<a data-tracking-control-name="learn_more_click" href="https://www.linkedin.com/help">x</a>'
                '<a class="artdeco-button" data-tracking-control-name="external_url_click" '
                'href="https://www.aboutamazon.com/news/aws/emulator?a=1&amp;b=2">Continue</a>')

    import curl_cffi.requests as cf_requests

    monkeypatch.setattr(cf_requests, "get", lambda *args, **kwargs: Response())
    assert linkedin._redirect_target("https://lnkd.in/gnepDsZ2") == "https://www.aboutamazon.com/news/aws/emulator?a=1&b=2"
