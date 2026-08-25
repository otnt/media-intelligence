from media_pipeline.cli import cmd_doctor, main
from media_pipeline.config import AppConfig


def test_transcribe_rejects_unsupported_url(capsys):
    assert main(["transcribe", "https://example.com/watch?v=nope"]) == 2
    err = capsys.readouterr().err
    assert "Only Bilibili, YouTube, and Xiaohongshu" in err


def test_transcribe_rejects_unknown_model(capsys):
    assert main(["transcribe", "https://www.bilibili.com/video/BV181KNeuEi2", "--asr-model", "whisper-tiny"]) == 2
    err = capsys.readouterr().err
    assert "Unknown ASR model" in err


def test_doctor_reports_qwen_analysis(capsys):
    cmd_doctor(AppConfig())
    out = capsys.readouterr().out
    assert "analysis:qwen3.8" in out
    assert "curl-cffi" in out

