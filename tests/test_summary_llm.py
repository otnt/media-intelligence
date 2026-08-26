import json
from pathlib import Path

from media_pipeline.config import AppConfig, SummaryConfig, load_config
from media_pipeline.models import NamedSegment, VideoMetadata
from media_pipeline.summary import render_summary_runs, run_summary_backends
from media_pipeline.summary_llm import (
    SummaryBackend,
    _gemini_generate,
    _openai_generate,
    _redact_secrets,
    catalog_summary_backends,
    resolve_summary_backends,
)


class FakeVision:
    name = "mlx_vlm"
    model_id = "local-qwen"

    def generate(self, prompt, images=None, max_tokens=None) -> str:
        return "本地要点"


def _metadata() -> VideoMetadata:
    return VideoMetadata(
        url="https://example.com",
        title="T",
        platform="Bilibili",
        author="A",
        video_id="BVxxxx",
        duration=1,
        published="",
        description="",
        thumbnail_url="",
        asr_model="",
    )


def _store_with_transcript(tmp_path: Path):
    from media_pipeline.artifacts import ArtifactStore

    artifacts = ArtifactStore(tmp_path / "artifacts", "BVxxxx")
    artifacts.save_named(
        [NamedSegment(start=0.0, end=1.0, speaker_id="A", speaker_label="Speaker 1", text="hello")]
    )
    return artifacts


def test_load_config_reads_summary_providers(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "summary:\n  providers: [qwen, gemini]\n  gemini_model: gemini-2.5-pro\n  openai_model: gpt-4.1\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.summary.providers == ["qwen", "gemini"]
    assert config.summary.gemini_model == "gemini-2.5-pro"
    assert config.summary.openai_model == "gpt-4.1"


def test_catalog_probes_qwen_without_loaded_vision(monkeypatch):
    monkeypatch.setattr(
        "media_pipeline.visual.vlm.probe_vlm",
        lambda *_args, **_kwargs: (True, "mlx-community/Qwen3.8-27B-4bit"),
    )
    rows = catalog_summary_backends(AppConfig(), None)
    by_key = {item["key"]: item for item in rows}
    assert by_key["qwen"]["available"] is True


def test_catalog_marks_cloud_unavailable_without_keys(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    rows = catalog_summary_backends(AppConfig(), FakeVision())
    by_key = {item["key"]: item for item in rows}
    assert by_key["qwen"]["available"] is True
    assert by_key["gemini"]["available"] is False
    assert by_key["openai"]["available"] is False


def test_resolve_all_uses_only_available_backends(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = AppConfig(summary=SummaryConfig(providers=["qwen"], gemini_api_key="secret"))
    backends = resolve_summary_backends(config, FakeVision(), "all")
    assert [item.key for item in backends] == ["qwen", "gemini"]


def test_resolve_default_expands_all_providers():
    config = AppConfig(summary=SummaryConfig(providers=["all"], gemini_api_key="g", openai_api_key="o"))
    backends = resolve_summary_backends(config, FakeVision(), "")
    assert [item.key for item in backends] == ["qwen", "gemini", "openai"]


def test_render_summary_runs_compares_models():
    text = render_summary_runs(
        [
            {"key": "qwen", "label": "Qwen3.8 (local)", "markdown": "本地版"},
            {"key": "gemini", "label": "Gemini (gemini-2.5-flash)", "markdown": "云端版"},
        ]
    )
    assert "### Qwen3.8 (local)" in text
    assert "本地版" in text
    assert "### Gemini (gemini-2.5-flash)" in text
    assert "云端版" in text


def test_run_summary_backends_keeps_success_when_one_fails(tmp_path: Path):
    artifacts = _store_with_transcript(tmp_path)

    def boom(*_args, **_kwargs):
        raise RuntimeError("quota")

    good = SummaryBackend(key="qwen", label="Qwen", model_id="qwen", generate_fn=lambda *_args, **_kwargs: "好")
    bad = SummaryBackend(key="gemini", label="Gemini", model_id="gemini", generate_fn=boom)
    result = run_summary_backends(artifacts, [good, bad], prompt="总结", metadata=_metadata())
    assert result["status"] == "completed"
    assert "好" in result["markdown"]
    assert "quota" in result["markdown"]
    assert result["runs"][0]["status"] == "completed"
    assert result["runs"][1]["status"] == "failed"


def test_run_summary_backends_merges_a_single_model(tmp_path: Path):
    artifacts = _store_with_transcript(tmp_path)
    qwen = SummaryBackend(key="qwen", label="Qwen", model_id="qwen", generate_fn=lambda *_a, **_k: "本地")
    gemini = SummaryBackend(
        key="gemini", label="Gemini", model_id="gemini", generate_fn=lambda *_a, **_k: "云端"
    )
    first = run_summary_backends(artifacts, [qwen], prompt="总结", metadata=_metadata())
    second = run_summary_backends(
        artifacts,
        [gemini],
        prompt="总结",
        metadata=_metadata(),
        existing_runs=list(first["runs"]),
        replace=False,
    )
    assert [item["key"] for item in second["runs"]] == ["qwen", "gemini"]
    assert "本地" in second["markdown"]
    assert "云端" in second["markdown"]


def test_gemini_generate_reads_candidates(monkeypatch, tmp_path: Path):
    calls = {}

    class FakeResp:
        def read(self):
            return json.dumps(
                {"candidates": [{"content": {"parts": [{"text": "Gemini briefing"}]}}]}
            ).encode()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_open(request, timeout=0):
        calls["url"] = request.full_url
        headers = {key.lower(): value for key, value in request.header_items()}
        calls["key"] = headers.get("x-goog-api-key")
        return FakeResp()

    monkeypatch.setattr("media_pipeline.summary_llm.urllib.request.urlopen", fake_open)
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpeg-bytes")
    text = _gemini_generate("secret-key", "gemini-2.5-flash", "hello", [image], 128)
    assert text == "Gemini briefing"
    assert "gemini-2.5-flash" in calls["url"]
    assert "key=" not in calls["url"]
    assert calls["key"] == "secret-key"


def test_openai_generate_reads_message(monkeypatch):
    class FakeResp:
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "OpenAI briefing"}}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr("media_pipeline.summary_llm.urllib.request.urlopen", lambda *args, **kwargs: FakeResp())
    text = _openai_generate("sk-test", "gpt-4.1-mini", "hello", [], 128)
    assert text == "OpenAI briefing"


def test_redact_secrets_strips_bearer_tokens():
    text = _redact_secrets("failed sk-secret12 leaked", {"Authorization": "Bearer sk-secret12"})
    assert "sk-secret12" not in text
    assert "[redacted]" in text
