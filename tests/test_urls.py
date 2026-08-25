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
    ],
)
def test_parse_supported_urls(url, platform, video_id):
    assert parse_video_ref(url) == (platform, video_id)


def test_parse_rejects_other_sites():
    with pytest.raises(UnsupportedURLError):
        parse_video_ref("https://www.xiaohongshu.com/explore/123")


def test_duration_and_timestamp_formatting():
    assert format_duration(1471) == "24:31"
    assert format_duration(3723) == "1:02:03"
    assert format_timestamp(8) == "00:00:08"
    assert format_timestamp(3723) == "01:02:03"
