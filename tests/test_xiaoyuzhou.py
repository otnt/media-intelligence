import json

import pytest

from media_pipeline.media import UnsupportedURLError, extract_supported_url, parse_video_ref
from media_pipeline.models import source_label
from media_pipeline.xiaoyuzhou import XiaoyuzhouError, parse_episode_page, shownotes_text

EID = "6afd12c6a4a79edf137afd61"


def _page(episode):
    data = {"props": {"pageProps": {"episode": episode}}}
    return f'<html><script id="__NEXT_DATA__" type="application/json">{json.dumps(data)}</script></html>'


def test_episode_links_parse():
    url = f"https://www.xiaoyuzhoufm.com/episode/{EID}?s=eyJ1IjoiNjM"
    assert parse_video_ref(url) == ("Xiaoyuzhou", f"xyz_{EID}")
    assert extract_supported_url(f"【小宇宙】157. 复盘 {url} 来自小宇宙") == f"https://www.xiaoyuzhoufm.com/episode/{EID}"
    assert source_label("Xiaoyuzhou") == "小宇宙"
    with pytest.raises(UnsupportedURLError):
        parse_video_ref("https://www.xiaoyuzhoufm.com/podcast/626b46ea9cbbf0451cf5a962")


def test_the_page_gives_audio_notes_and_podcast():
    notes = ('<img src="x.png"><p><span>今天的嘉宾是蔡伟。</span></p><ul><li>02:15 为什么是现在</li>'
             '<li>1:05:30 入口之争</li></ul><p><a href="https://example.com/a">相关链接</a></p>')
    episode = parse_episode_page(_page({
        "eid": EID, "title": "157. 复盘Personal Agent", "duration": 9820, "pubDate": "2026-11-18T03:08:54.630Z",
        "shownotes": notes, "description": "short", "podcast": {"title": "张小珺Jùn｜商业访谈录", "author": "张小珺",
                                                                "image": {"picUrl": "https://image.xyzcdn.net/p.png"}},
        "media": {"source": {"url": "https://media.xyzcdn.net/a/b.m4a"}},
    }), f"https://www.xiaoyuzhoufm.com/episode/{EID}")
    assert episode.video_id == f"xyz_{EID}" and episode.audio_url.endswith("b.m4a") and episode.duration == 9820
    assert episode.published == "2026-11-18" and episode.podcast == "张小珺Jùn｜商业访谈录"
    assert episode.notes == "今天的嘉宾是蔡伟。\n- 02:15 为什么是现在\n- 1:05:30 入口之争\n相关链接 (https://example.com/a)"


def test_paid_or_missing_episodes_are_clear_errors():
    with pytest.raises(XiaoyuzhouError, match="没有找到"):
        parse_episode_page("<html></html>", "u")
    with pytest.raises(XiaoyuzhouError, match="付费"):
        parse_episode_page(_page({"eid": EID, "title": "x"}), "u")
    assert shownotes_text("<p>a&amp;b</p><p></p><p>c</p>") == "a&b\n\nc"
