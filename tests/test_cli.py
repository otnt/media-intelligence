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


def test_transcribe_help_documents_keyframes_flag(capsys):
    try:
        main(["transcribe", "-h"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "--keyframes" in out
    assert "slow" in out
    assert "default" in out


def test_doctor_reports_qwen_analysis(capsys, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cmd_doctor(AppConfig())
    out = capsys.readouterr().out
    assert "analysis:qwen3.8" in out
    assert "curl-cffi" in out
    assert "summary:gemini" in out
    assert "summary:openai" in out
    assert "WARN" in out

