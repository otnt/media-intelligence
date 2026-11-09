import json
import subprocess
from pathlib import Path

import pytest

from media_pipeline.asr.agy import (
    AgyASRProvider,
    AgyError,
    AgySettings,
    Piece,
    parse_segments,
    plan_windows,
    run_agy,
    stitch_windows,
)
from media_pipeline.models import ASROptions


def _stream(*events):
    return "\n".join(json.dumps(e) for e in events)


def _tool(name, path=""):
    return {"step_update": {"step_type": "tool", "state": "ACTIVE", "tool_info": {"name": name, "parameters": {"Path": path}}}}


def _result(response, status="SUCCESS"):
    return {"event": "result", "result": {"status": status, "response": response}}


def _runner(stdout, calls=None):
    def run(command, **kwargs):
        if calls is not None:
            calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    return run


def test_run_agy_locks_down_the_agent_and_returns_response(tmp_path: Path):
    clip = tmp_path / "audio.wav"
    clip.write_bytes(b"RIFF")
    calls = []
    out = run_agy(
        "transcribe",
        workspace=tmp_path,
        allowed_reads={clip},
        settings=AgySettings(),
        schema={"type": "object"},
        runner=_runner(_stream(_tool("read_file", str(clip)), _tool("finish"), _result('{"segments": []}')), calls),
    )
    assert out == '{"segments": []}'
    command, kwargs = calls[0]
    assert "--sandbox" in command
    assert command[command.index("--mode") + 1] == "accept-edits"
    assert json.loads(command[command.index("--settings") + 1]) == {"allowNonWorkspaceAccess": False}
    assert kwargs["cwd"] == str(tmp_path)


@pytest.mark.parametrize("event", [_tool("run_command"), _tool("read_file", "/etc/passwd")])
def test_run_agy_rejects_any_other_tool_use(tmp_path: Path, event):
    clip = tmp_path / "audio.wav"
    clip.write_bytes(b"RIFF")
    with pytest.raises(AgyError, match="unexpected tool"):
        run_agy("x", workspace=tmp_path, allowed_reads={clip}, settings=AgySettings(),
                runner=_runner(_stream(event, _result("{}"))))


def test_run_agy_surfaces_timeouts(tmp_path: Path):
    with pytest.raises(AgyError, match="ERROR"):
        run_agy("x", workspace=tmp_path, allowed_reads=set(), settings=AgySettings(),
                runner=_runner(_stream(_result("", status="ERROR"))))


def test_parse_segments_tolerates_code_fences():
    pieces = parse_segments('```json\n{"segments":[{"start":1,"end":2,"speaker":"SPEAKER_00","text":" 你好  world "}]}\n```')
    assert pieces == [Piece(1.0, 2.0, "你好 world", "SPEAKER_00")]


def test_plan_windows_overlaps_and_absorbs_short_tail():
    assert plan_windows(700, 600, 30) == [(0.0, 700)]
    windows = plan_windows(1500, 600, 30)
    assert windows == [(0.0, 600.0), (570.0, 1170.0), (1140.0, 1500)]


def test_stitch_maps_speakers_across_windows_by_overlap():
    first = [Piece(0, 285, "a", "w0:SPEAKER_00"), Piece(285, 300, "b", "w0:SPEAKER_01")]
    # Window 2 swapped the labels: its SPEAKER_00 is window 1's SPEAKER_01 and vice versa.
    second = [
        Piece(280, 285, "a-tail", "w1:SPEAKER_01"),
        Piece(285, 299, "b", "w1:SPEAKER_00"),
        Piece(300, 400, "c", "w1:SPEAKER_01"),
        Piece(400, 450, "d", "w1:SPEAKER_02"),
    ]
    out = stitch_windows([(0, 300, first), (280, 450, second)])
    assert [(p.text, p.speaker) for p in out] == [
        ("a", "SPEAKER_00"), ("b", "SPEAKER_01"), ("c", "SPEAKER_00"), ("d", "SPEAKER_02"),
    ]


def test_provider_end_to_end_with_fake_agy(tmp_path: Path, monkeypatch):
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr("media_pipeline.asr.agy._probe_duration", lambda path: 40.0)
    monkeypatch.setattr("media_pipeline.asr.agy.snap_to_speech", lambda pieces, path: pieces)
    prompts = []

    def run(command, **kwargs):
        prompts.append(command[command.index("-p") + 1])
        clip = Path(kwargs["cwd"]) / "audio.wav"
        assert clip.exists()
        body = {"segments": [
            {"start": 0.5, "end": 4, "speaker": "SPEAKER_00", "text": "大家好"},
            {"start": 4.5, "end": 9, "speaker": "SPEAKER_01", "text": "Hi, 我是 Eddy"},
        ]}
        return subprocess.CompletedProcess(command, 0, stdout=_stream(_tool("read_file", str(clip)), _result(json.dumps(body))), stderr="")

    provider = AgyASRProvider(runner=run)
    transcript = provider.transcribe(audio, ASROptions(language="zh", context="Title: 对话 Eddy"))
    assert [(s.speaker, s.text) for s in transcript.segments] == [("SPEAKER_00", "大家好"), ("SPEAKER_01", "Hi, 我是 Eddy")]
    assert transcript.provider == "agy"
    assert "mostly Chinese" in prompts[0] and "对话 Eddy" in prompts[0]
