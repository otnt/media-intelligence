from types import SimpleNamespace
from pathlib import Path

from media_pipeline.asr.qwen3 import Qwen3ASRProvider
from media_pipeline.models import ASROptions
from media_pipeline.vad import AsrChunk


def test_qwen_transcribes_vad_chunks_without_forced_aligner(tmp_path: Path, monkeypatch):
    path = tmp_path / "talk.wav"
    _write_tone_wav(path, silence_sec=0.8, tone_sec=1.2)

    captured: list[dict] = []

    class FakeSession:
        def transcribe(self, audio, **kwargs):
            captured.append(kwargs)
            samples, sample_rate = audio
            assert kwargs.get("return_timestamps") is False
            assert sample_rate == 16000
            assert len(samples) < 16000 * 8
            return SimpleNamespace(text="hello there", language="English")

    provider = Qwen3ASRProvider()
    provider._model = FakeSession()
    transcript = provider.transcribe(path, ASROptions(language=None))

    assert captured
    assert all(item.get("return_timestamps") is False for item in captured)
    assert transcript.segments
    assert transcript.segments[0].text == "hello there"
    assert transcript.segments[0].words is None
    assert transcript.segments[0].start < transcript.segments[0].end
    assert transcript.language == "English"


def test_qwen_uniques_repeated_chunk_languages(tmp_path: Path, monkeypatch):
    path = tmp_path / "talk.wav"
    _write_tone_wav(path, silence_sec=0.2, tone_sec=2.6)

    class FakeSession:
        def transcribe(self, audio, **kwargs):
            return SimpleNamespace(text="hello", language="Chinese")

    monkeypatch.setattr(
        "media_pipeline.asr.qwen3.build_asr_chunks",
        lambda audio_path: [AsrChunk(0.0, 1.0), AsrChunk(1.2, 2.2)],
    )
    provider = Qwen3ASRProvider()
    provider._model = FakeSession()
    transcript = provider.transcribe(path, ASROptions(language=None))
    assert transcript.language == "Chinese"
    assert len(transcript.segments) == 2


def _write_tone_wav(path: Path, silence_sec: float, tone_sec: float, sample_rate: int = 16000) -> None:
    import array
    import math
    import wave

    samples: list[float] = [0.0] * int(silence_sec * sample_rate)
    samples.extend(
        0.35 * math.sin(2 * math.pi * 220.0 * index / sample_rate)
        for index in range(int(tone_sec * sample_rate))
    )
    samples.extend([0.0] * int(silence_sec * sample_rate))
    pcm = array.array("h", [int(sample * 32767) for sample in samples])
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
