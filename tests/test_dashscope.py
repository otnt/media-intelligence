import io
import json
import threading
import urllib.request
from pathlib import Path

from media_pipeline.artifacts import ArtifactStore
from media_pipeline.asr.dashscope import DashScopeASRProvider, parse_transcription
from media_pipeline.config import AppConfig, DashScopeConfig, load_config
from media_pipeline.dashscope import DashScopeClient, api_host
from media_pipeline.diarization import diarization_from_transcript
from media_pipeline.models import ASROptions, Task, TaskStatus, Transcript, TranscriptSegment, VideoMetadata
from media_pipeline.summary_llm import _qwen_max_base_url, _qwen_max_key, resolve_summary_backends
from media_pipeline.visual.filtering import filter_keyframes
from media_pipeline.visual.models import Keyframe
from media_pipeline.visual.vlm import DashScopeVisionProvider, build_vision_provider

def _fake_download(url, vid, dest, cfg):
    path = dest / f"{vid}.mp4"
    path.write_bytes(b"v")
    return path


def _fake_extract(video, audio):
    audio.write_bytes(b"a")
    return audio


RESULT = {
    "transcripts": [
        {
            "channel_id": 0,
            "sentences": [
                {"begin_time": 100, "end_time": 3820, "text": "大家好，今天聊 AI agent。", "speaker_id": 3,
                 "words": [{"begin_time": 100, "end_time": 600, "text": "大家", "punctuation": ""}]},
                {"begin_time": 3820, "end_time": 6500, "text": "好的，我先说。", "speaker_id": 1},
                {"begin_time": 6500, "end_time": 7000, "text": "  ", "speaker_id": 1},
                {"begin_time": 7000, "end_time": 9000, "text": "Thanks.", "speaker_id": 3},
            ],
        }
    ]
}


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpener:
    """Scripted DashScope endpoints: policy → OSS upload → submit → poll → result."""

    def __init__(self):
        self.calls = []
        self.polls = 0

    def __call__(self, request: urllib.request.Request, timeout=None):
        url = request.full_url
        headers = {k.lower(): v for k, v in request.header_items()}
        self.calls.append((request.get_method(), url, headers, request.data))
        if "/api/v1/uploads" in url:
            return _Resp(json.dumps({"data": {
                "upload_host": "https://oss.example.com", "upload_dir": "dashscope-instant/abc",
                "oss_access_key_id": "id", "signature": "sig", "policy": "pol",
                "x_oss_object_acl": "private", "x_oss_forbid_overwrite": "true", "max_file_size_mb": "1024",
            }}).encode())
        if url == "https://oss.example.com":
            return _Resp(b"")
        if url.endswith("/services/audio/asr/transcription"):
            return _Resp(json.dumps({"output": {"task_id": "t-1", "task_status": "PENDING"}}).encode())
        if "/api/v1/tasks/t-1" in url:
            self.polls += 1
            status = "RUNNING" if self.polls == 1 else "SUCCEEDED"
            return _Resp(json.dumps({"output": {"task_status": status, "results": [
                {"subtask_status": "SUCCEEDED", "transcription_url": "https://result.example.com/r.json"}]}}).encode())
        if url == "https://result.example.com/r.json":
            return _Resp(json.dumps(RESULT).encode())
        if url.endswith("/compatible-mode/v1/chat/completions"):
            return _Resp(json.dumps({"choices": [{"message": {"content": '{"informative": true, "score": 0.9}'}}]}).encode())
        raise AssertionError(f"unexpected url {url}")


def _client(opener):
    return DashScopeClient(api_key="sk-test", host="https://dashscope.aliyuncs.com", sleep=lambda s: None, opener=opener)


def test_api_host_strips_paths():
    assert api_host("https://dashscope-intl.aliyuncs.com/compatible-mode/v1") == "https://dashscope-intl.aliyuncs.com"
    assert api_host("") == "https://dashscope.aliyuncs.com"


def test_parse_transcription_normalizes_speakers_and_ms():
    transcript = parse_transcription(RESULT, provider="dashscope", model="fun-asr")
    assert [s.speaker for s in transcript.segments] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert transcript.segments[0].start == 0.1 and transcript.segments[0].end == 3.82
    assert transcript.segments[0].words[0].text == "大家"
    round_trip = Transcript.from_dict(transcript.to_dict())
    assert round_trip.segments[1].speaker == "SPEAKER_01"


def test_fun_asr_uploads_submits_with_diarization_and_polls(tmp_path: Path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF....")
    opener = FakeOpener()
    config = AppConfig(dashscope=DashScopeConfig(api_key="sk-test", speaker_count=2))
    provider = DashScopeASRProvider(config=config, client=_client(opener))
    transcript = provider.transcribe(audio, ASROptions(language="zh"))
    assert len(transcript.segments) == 3
    submit = next(c for c in opener.calls if c[1].endswith("/transcription"))
    body = json.loads(submit[3])
    assert body["model"] == "fun-asr"
    assert body["input"]["file_urls"][0].startswith("oss://dashscope-instant/abc/")
    assert body["parameters"]["diarization_enabled"] is True
    assert body["parameters"]["speaker_count"] == 2
    assert body["parameters"]["language_hints"] == ["zh"]
    assert submit[2]["x-dashscope-async"] == "enable"
    assert submit[2]["x-dashscope-ossresourceresolve"] == "enable"
    assert submit[2]["authorization"] == "Bearer sk-test"
    assert opener.polls == 2


def test_diarization_from_transcript_requires_every_segment_tagged():
    tagged = parse_transcription(RESULT, provider="dashscope", model="fun-asr")
    result = diarization_from_transcript(tagged)
    assert result is not None and result.provider == "asr:dashscope:fun-asr"
    assert [s.speaker for s in result.segments] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    untagged = Transcript(language=None, provider="x", model="y", segments=[TranscriptSegment(0, 1, "hi")])
    assert diarization_from_transcript(untagged) is None


def test_pipeline_uses_cloud_speakers_and_skips_diarizer(tmp_path: Path, monkeypatch):
    from tests.test_pipeline import FakeVision, FakeVisual, _config
    from media_pipeline.pipeline import Pipeline
    from media_pipeline.store import TaskStore

    config = _config(tmp_path)
    store = TaskStore(config.paths.db)
    metadata = VideoMetadata(
        url="https://www.bilibili.com/video/BVcloud", title="t", platform="Bilibili", author="a",
        video_id="BVcloud", duration=9.0, published="", description="", thumbnail_url="", asr_model="fun-asr",
    )

    class CloudASR:
        local = False
        bound = None

        def bind_config(self, cfg):
            self.bound = cfg

        def transcribe(self, audio_path, options=None):
            assert not slot_state["held"], "cloud ASR must not take the local model slot"
            return parse_transcription(RESULT, provider="dashscope", model="fun-asr")

    class ExplodingDiarizer:
        name = "boom"

        def diarize(self, audio_path):
            raise AssertionError("pyannote should not run when ASR already tagged speakers")

    slot_state = {"held": False}

    class TrackingSlot:
        def __enter__(self):
            slot_state["held"] = True

        def __exit__(self, *exc):
            slot_state["held"] = False
            return False

    asr = CloudASR()
    monkeypatch.setattr("media_pipeline.pipeline.fetch_metadata", lambda url, model, cfg: metadata)
    monkeypatch.setattr(
        "media_pipeline.pipeline.download_video",
        _fake_download,
    )
    monkeypatch.setattr(
        "media_pipeline.pipeline.extract_audio", _fake_extract
    )
    monkeypatch.setattr("media_pipeline.pipeline.require_provider", lambda model_id: asr)
    pipeline = Pipeline(
        config, store, diarization=ExplodingDiarizer(), visual=FakeVisual(), vision=FakeVision(), model_lock=TrackingSlot()
    )
    task = store.insert(Task(id="c1", url=metadata.url, asr_model="fun-asr", video_id="BVcloud", extra={}))
    result = pipeline.run(task)
    assert result.status == TaskStatus.completed, result.error
    assert asr.bound is config
    named = ArtifactStore(config.paths.artifacts, "BVcloud").load_named()
    assert len({seg.speaker_id for seg in named}) == 2


def test_cloud_vision_judges_frames_in_parallel_without_model_lock(tmp_path: Path):
    from tests.test_pipeline import _config

    config = _config(tmp_path)
    artifacts = ArtifactStore(config.paths.artifacts, "BVv")
    frames = []
    for i in range(6):
        rel = f"keyframes/f{i}.jpg"
        (artifacts.root / "keyframes").mkdir(parents=True, exist_ok=True)
        (artifacts.root / rel).write_bytes(b"\xff\xd8jpeg")
        frames.append(Keyframe(timestamp=float(i), image_path=rel, sources=["periodic"]))

    active = {"now": 0, "peak": 0}
    gate = threading.Barrier(3, timeout=5)
    lock = threading.Lock()

    class SlowClient:
        def chat(self, model, prompt, images, max_tokens=256, extra=None):
            with lock:
                active["now"] += 1
                active["peak"] = max(active["peak"], active["now"])
            try:
                gate.wait()
            except threading.BrokenBarrierError:
                pass
            with lock:
                active["now"] -= 1
            if images[0].name == "f5.jpg":
                raise RuntimeError("HTTP 500")
            return '{"informative": true, "score": 0.9, "category": "slide"}'

    class NoLock:
        def __enter__(self):
            raise AssertionError("cloud vision must not take the local model slot")

        def __exit__(self, *exc):
            return False

    provider = DashScopeVisionProvider(SlowClient(), "qwen3.8-flash", concurrency=3)
    metadata = VideoMetadata(url="u", title="t", platform="Bilibili", author="a", video_id="BVv", duration=6.0,
                             published="", description="", thumbnail_url="", asr_model="fun-asr")
    verdicts, selected = filter_keyframes(frames, artifacts, metadata, [], {}, provider, model_lock=NoLock())
    assert active["peak"] >= 2
    assert len(selected) == 6
    assert verdicts[5].decision == "parse_error"
    # The failed frame is retried next run, the good ones come from cache.
    calls = {"n": 0}

    class CountingClient:
        def chat(self, *args, **kwargs):
            calls["n"] += 1
            return '{"informative": false, "score": 0.1}'

    provider._client = CountingClient()
    verdicts, _ = filter_keyframes(frames, artifacts, metadata, [], {}, provider)
    assert calls["n"] == 1
    assert verdicts[5].kept is False


def test_build_vision_provider_dashscope_and_summary_share_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    path = tmp_path / "config.yaml"
    path.write_text(
        "analysis:\n  provider: dashscope\n"
        "dashscope:\n  api_key: sk-shared\n  base_url: https://dashscope-intl.aliyuncs.com\n"
        "  vision_model: qwen3.8-flash\n  speaker_count: 3\n"
        "summary:\n  providers: [qwen3.8-max]\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.dashscope.speaker_count == 3
    vision = build_vision_provider(config)
    assert isinstance(vision, DashScopeVisionProvider)
    assert vision._client.host == "https://dashscope-intl.aliyuncs.com"
    assert _qwen_max_key(config) == "sk-shared"
    assert _qwen_max_base_url(config) == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    backends = resolve_summary_backends(config, vision)
    assert [b.key for b in backends] == ["qwen3.8-max"]
    # A cloud judge must not masquerade as the local Qwen3.8-27B summary backend.
    assert not any(b.local for b in resolve_summary_backends(config, vision, "qwen"))


def test_legacy_summary_qwen_key_feeds_dashscope(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    path = tmp_path / "config.yaml"
    path.write_text("summary:\n  qwen_api_key: sk-legacy\n", encoding="utf-8")
    config = load_config(path)
    assert config.dashscope.api_key == "sk-legacy"
    assert config.analysis.provider == "local"


def test_fun_asr_splits_long_audio_and_links_speakers_across_windows(tmp_path: Path, monkeypatch):
    audio = tmp_path / "long.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr("media_pipeline.asr.dashscope._probe_duration", lambda path: 400.0)
    cuts = []

    def fake_cut(path, tmp_dir, start=None, end=None):
        cuts.append((start, end))
        clip = tmp_dir / f"clip-{int(start)}.m4a"
        clip.write_bytes(b"x")
        return clip

    monkeypatch.setattr("media_pipeline.asr.dashscope._compress_for_upload", fake_cut)

    class Client:
        def upload_file(self, path, model):
            return f"oss://{path.name}"

        def transcribe(self, model, url, params):
            assert params["diarization_enabled"] is True
            # Window 1 (0-150): host=0, guest=1. Window 2 (120-...): labels swapped; overlap 120-150 has both.
            if "clip-0" in url:
                sentences = [(0, 100_000, 0, "host intro"), (110_000, 128_000, 1, "guest one"), (130_000, 149_000, 0, "host two")]
            elif "clip-120" in url:
                sentences = [(0, 8_000, 0, "guest one"), (10_000, 29_000, 1, "host two"), (40_000, 100_000, 0, "guest three"),
                             (125_000, 135_000, 0, "guest tail"), (136_000, 149_000, 1, "host tail")]
            else:
                sentences = [(5_000, 15_000, 0, "guest tail"), (16_000, 29_000, 1, "host tail"),
                             (35_000, 50_000, 1, "host four"), (60_000, 140_000, 0, "guest five")]
            return {"transcripts": [{"sentences": [
                {"begin_time": b, "end_time": e, "speaker_id": spk, "text": t} for b, e, spk, t in sentences]}]}

    config = AppConfig(dashscope=DashScopeConfig(api_key="sk", chunk_sec=150, chunk_overlap_sec=30))
    provider = DashScopeASRProvider(config=config, client=Client())
    transcript = provider.transcribe(audio, ASROptions())
    assert cuts == [(0.0, 150.0), (120.0, 270.0), (240.0, 400.0)]
    by_text = {s.text: s.speaker for s in transcript.segments}
    assert by_text["host intro"] == by_text["host two"] == by_text["host four"]
    assert by_text["guest one"] == by_text["guest three"] == by_text["guest five"]
    assert by_text["host intro"] != by_text["guest one"]
    assert [s.text for s in transcript.segments].count("guest one") == 1
    assert [s.text for s in transcript.segments].count("guest tail") == 1


def test_switching_asr_model_rebuilds_the_speaker_aligned_transcript(tmp_path: Path, monkeypatch):
    from tests.test_pipeline import FakeVision, FakeVisual, _config
    from media_pipeline.diarization import NullDiarizationProvider
    from media_pipeline.pipeline import Pipeline
    from media_pipeline.store import TaskStore

    config = _config(tmp_path)
    store = TaskStore(config.paths.db)
    metadata = VideoMetadata(url="https://www.bilibili.com/video/BVswap", title="t", platform="Bilibili", author="a",
                             video_id="BVswap", duration=9.0, published="", description="", thumbnail_url="", asr_model="x")

    class ASR:
        def __init__(self, text, speaker):
            self.text, self.speaker = text, speaker

        def transcribe(self, audio_path, options=None):
            return Transcript(language=None, provider="p", model="m",
                              segments=[TranscriptSegment(0.0, 4.0, self.text, speaker=self.speaker)])

    asrs = {"whisper-large-v3-turbo": ASR("old model text", None), "fun-asr": ASR("new model text", "SPEAKER_00")}
    monkeypatch.setattr("media_pipeline.pipeline.fetch_metadata", lambda url, model, cfg: metadata)
    monkeypatch.setattr("media_pipeline.pipeline.download_video",
                        _fake_download)
    monkeypatch.setattr("media_pipeline.pipeline.extract_audio", _fake_extract)
    monkeypatch.setattr("media_pipeline.pipeline.require_provider", lambda model_id: asrs[model_id])
    pipeline = Pipeline(config, store, diarization=NullDiarizationProvider(), visual=FakeVisual(), vision=FakeVision())
    for index, model in enumerate(["whisper-large-v3-turbo", "fun-asr"]):
        task = store.insert(Task(id=f"t{index}", url=metadata.url, asr_model=model, video_id="BVswap", extra={}))
        assert pipeline.run(task).status == TaskStatus.completed
    artifacts = ArtifactStore(config.paths.artifacts, "BVswap")
    assert [seg.text for seg in artifacts.load_named()] == ["new model text"]
    # The summary was regenerated from the new transcript, not reused from the first run.
    assert pipeline.vision.calls == 2
