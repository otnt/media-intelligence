from media_pipeline.media import UnsupportedURLError, parse_video_ref
from media_pipeline.models import format_duration, format_timestamp

import pytest


@pytest.mark.parametrize(
    ("url", "platform", "video_id"),
    [
        ("https://www.youtube.com/watch?v=dQw4w9wgGcQ", "YouTube", "dQw4w9wgGcQ"),
        ("https://youtu.be/dQw4w9wgGcQ", "YouTube", "dQw4w9wgGcQ"),
        ("https://www.bilibili.com/video/BV181KNeuEi2", "Bilibili", "BV181KNeuEi2"),
        ("https://www.bilibili.com/video/BV181KNeuEi2/?spm_id_from=333", "Bilibili", "BV181KNeuEi2"),
        (
            "https://www.bilibili.com/list/watchlater/?bvid=BV1fP4y1j76Q&oid=893222135",
            "Bilibili",
            "BV1fP4y1j76Q",
        ),
        ("https://www.xiaohongshu.com/explore/64aaaaaaaaaaaaaaaaaaaaaa", "Xiaohongshu", "64aaaaaaaaaaaaaaaaaaaaaa"),
        ("http://xhslink.com/o/2x5jqGA2hr6", "Xiaohongshu", "2x5jqGA2hr6"),
        ("https://www.xiaohongshu.com/discovery/item/64aaaaaaaaaaaaaaaaaaaaaa", "Xiaohongshu", "64aaaaaaaaaaaaaaaaaaaaaa"),
        (
            "https://www.xiaohongshu.com/search_result/64aaaaaaaaaaaaaaaaaaaaaa?xsec_token=abc",
            "Xiaohongshu",
            "64aaaaaaaaaaaaaaaaaaaaaa",
        ),
        (
            "https://www.rednote.com/explore/64aaaaaaaaaaaaaaaaaaaaaa?xsec_token=abc",
            "Xiaohongshu",
            "64aaaaaaaaaaaaaaaaaaaaaa",
        ),
    ],
)
def test_parse_supported_urls(url, platform, video_id):
    assert parse_video_ref(url) == (platform, video_id)


def test_canonicalizes_watchlater_to_video_page():
    from media_pipeline.media import canonicalize_url

    url = "https://www.bilibili.com/list/watchlater/?bvid=BV1fP4y1j76Q&oid=893222135"
    assert canonicalize_url(url) == "https://www.bilibili.com/video/BV1fP4y1j76Q"


def test_parse_rejects_other_sites():
    with pytest.raises(UnsupportedURLError):
        parse_video_ref("https://twitter.com/x/status/1")


def test_duration_and_timestamp_formatting():
    assert format_duration(1471) == "24:31"
    assert format_duration(3723) == "1:02:03"
    assert format_timestamp(8) == "00:00:08"
    assert format_timestamp(3723) == "01:02:03"
