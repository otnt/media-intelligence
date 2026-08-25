from media_pipeline.cli import main


def test_transcribe_rejects_unsupported_url(capsys):
    assert main(["transcribe", "https://example.com/watch?v=nope"]) == 2
    err = capsys.readouterr().err
    assert "Only Bilibili and YouTube" in err


def test_transcribe_rejects_unknown_model(capsys):
    assert main(["transcribe", "https://www.bilibili.com/video/BV181KNeuEi2", "--asr-model", "whisper-tiny"]) == 2
    err = capsys.readouterr().err
    assert "Unknown ASR model" in err
