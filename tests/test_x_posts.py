from media_pipeline import x_posts
from media_pipeline.media import parse_video_ref


def _fx(**tweet):
    base = {"id": "123", "author": {"name": "Addy", "screen_name": "addy"}, "created_timestamp": 1794900029,
            "text": "hello world", "media": {}}
    base.update(tweet)
    return {"code": 200, "tweet": base}


def test_parse_urls():
    for url in ("https://x.com/addyosmani/status/2122590683055993308?a=k&s=46",
                "https://twitter.com/addyosmani/status/2122590683055993308",
                "https://mobile.twitter.com/i/web/status/2122590683055993308"):
        assert parse_video_ref(url) == ("X", "x_2122590683055993308")
    assert x_posts.canonicalize_x_url("https://twitter.com/a/status/9?s=1") == "https://x.com/a/status/9"


def test_article_becomes_markdown_with_cover():
    article = {"title": "Verification", "cover_media": {"media_info": {"original_img_url": "https://img/cover.jpg"}},
               "content": {"blocks": [
                   {"type": "unstyled", "text": "Intro paragraph."},
                   {"type": "header-two", "text": "Loops"},
                   {"type": "unordered-list-item", "text": "first"},
                   {"type": "atomic", "text": " "},
               ]}}
    post = x_posts.parse_fxtwitter(_fx(text="https://t.co/x", article=article), "https://x.com/addy/status/123")
    assert post.title == "Verification"
    assert post.media_kind == "image" and post.image_urls == ["https://img/cover.jpg"]
    assert post.text == "# Verification\n\nIntro paragraph.\n\n### Loops\n\n- first"
    assert post.author == "Addy (@addy)" and post.published == "2026-11-17"


def test_plain_tweet_with_quote():
    post = x_posts.parse_fxtwitter(_fx(text="My take\nmore", quote={"text": "original", "author": {"screen_name": "b"}}),
                                   "https://x.com/addy/status/123")
    assert post.title == "My take"
    assert post.text.endswith("引用 @b：\noriginal")


def test_video_tweet_is_transcribed(monkeypatch):
    monkeypatch.setattr(x_posts, "has_audio", lambda video: True)
    post = x_posts.parse_fxtwitter(_fx(media={"videos": [{"url": "https://video/v.mp4", "duration": 42.5}]}),
                                   "https://x.com/addy/status/123")
    assert post.media_kind == "video" and post.video_url == "https://video/v.mp4" and post.duration == 42.5


def test_falls_back_to_embed_endpoint(monkeypatch):
    x_posts._CACHE.clear()
    calls = []

    def fake_get(url):
        calls.append(url)
        if "fxtwitter" in url:
            raise x_posts.XError("fetching_metadata", "down")
        return {"id_str": "777", "user": {"name": "A", "screen_name": "a"}, "text": "embedded text",
                "created_at": "2026-11-17T00:00:00.000Z",
                "mediaDetails": [{"type": "photo", "media_url_https": "https://img/p.jpg"}]}

    monkeypatch.setattr(x_posts, "_get_json", fake_get)
    post = x_posts.extract_x_post("https://x.com/a/status/777")
    assert "syndication" in calls[-1] and "token=" in calls[-1]
    assert post.text == "embedded text" and post.image_urls == ["https://img/p.jpg"]
    x_posts._CACHE.clear()


def test_silent_video_becomes_text_post(monkeypatch):
    monkeypatch.setattr(x_posts, "_get_text", lambda url: "#EXTM3U\n#EXT-X-STREAM-INF:CODECS=\"avc1.4D4015\"\nv.m3u8")
    video = {"url": "https://video/v.mp4", "duration": 40, "thumbnail_url": "https://img/thumb.jpg",
             "formats": [{"url": "https://video/pl/master.m3u8?tag=29", "container": "m3u8"}]}
    post = x_posts.parse_fxtwitter(_fx(media={"videos": [video]}), "https://x.com/addy/status/123")
    assert post.media_kind == "image" and post.image_urls == ["https://img/thumb.jpg"] and not post.video_url


def test_has_audio_reads_playlist(monkeypatch):
    monkeypatch.setattr(x_posts, "_get_text", lambda url: '#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="a"')
    assert x_posts.has_audio({"formats": [{"url": "https://v/m.m3u8", "container": "m3u8"}]})
