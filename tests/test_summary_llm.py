import json
from pathlib import Path

from media_pipeline.config import AppConfig, SummaryConfig, load_config
from media_pipeline.models import NamedSegment, VideoMetadata
from media_pipeline.summary import (
    coerce_summary_runs,
    format_run_duration,
    parse_summary_output,
    render_summary_runs,
    run_summary_backends,
    split_summary_thinking,
)
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

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, prompt, images=None, max_tokens=None, **kwargs) -> str:
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens, **kwargs})
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
        "summary:\n  providers: [qwen, gemini]\n  qwen_8bit_model: org/custom-8bit\n  gemini_model: gemini-2.5-pro\n  openai_model: gpt-4.1\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.summary.providers == ["qwen", "gemini"]
    assert config.summary.gemini_model == "gemini-2.5-pro"
    assert config.summary.openai_model == "gpt-4.1"
    assert config.summary.qwen_8bit_model == "org/custom-8bit"


def test_catalog_probes_qwen_without_loaded_vision(monkeypatch):
    monkeypatch.setattr(
        "media_pipeline.visual.vlm.probe_vlm",
        lambda *_args, **_kwargs: (True, "mlx-community/Qwen3.8-27B-4bit"),
    )
    rows = catalog_summary_backends(AppConfig(), None)
    by_key = {item["key"]: item for item in rows}
    assert by_key["qwen"]["available"] is True
    assert by_key["qwen-low"]["available"] is True
    assert by_key["qwen-xhigh"]["available"] is True
    assert "qwen-8bit" in by_key
    assert "qwen-8bit-xhigh" in by_key
    assert by_key["qwen-8bit"]["model"] == "mlx-community/Qwen3.8-27B-8bit"


def test_catalog_marks_cloud_unavailable_without_keys(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    rows = catalog_summary_backends(AppConfig(), FakeVision())
    by_key = {item["key"]: item for item in rows}
    assert by_key["qwen"]["available"] is True
    assert by_key["qwen-low"]["available"] is True
    assert by_key["qwen-medium"]["available"] is True
    assert by_key["qwen-xhigh"]["available"] is True
    assert by_key["gemini"]["available"] is False
    assert by_key["openai"]["available"] is False


def test_resolve_all_uses_only_available_backends(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = AppConfig(summary=SummaryConfig(providers=["qwen"], gemini_api_key="secret"))
    backends = resolve_summary_backends(config, FakeVision(), "all")
    assert [item.key for item in backends] == ["qwen", "gemini"]


def test_resolve_default_uses_qwen_xhigh():
    vision = FakeVision()
    backends = resolve_summary_backends(AppConfig(), vision, "")
    assert [item.key for item in backends] == ["qwen-xhigh"]
    backends[0].generate("hello", [], 128)
    assert vision.calls[-1]["enable_thinking"] is True
    assert vision.calls[-1]["reasoning_effort"] == "xhigh"


def test_resolve_default_expands_all_providers():
    config = AppConfig(summary=SummaryConfig(providers=["all"], gemini_api_key="g", openai_api_key="o"))
    backends = resolve_summary_backends(config, FakeVision(), "")
    assert [item.key for item in backends] == ["qwen", "gemini", "openai"]


def test_resolve_qwen_8bit_xhigh_uses_8bit_vision():
    four = FakeVision()
    eight = FakeVision()
    eight.model_id = "mlx-community/Qwen3.8-27B-8bit"
    backends = resolve_summary_backends(AppConfig(), four, "qwen-8bit-xhigh", vision_8bit=eight)
    assert [item.key for item in backends] == ["qwen-8bit-xhigh"]
    assert backends[0].label == "qwen-3.8-27B 8bit thinking xhigh"
    backends[0].generate("hello", [], 128)
    assert four.calls == []
    assert eight.calls[-1]["enable_thinking"] is True
    assert eight.calls[-1]["reasoning_effort"] == "xhigh"
    assert eight.calls[-1]["max_tokens"] >= 8192


def test_catalog_lists_8bit_even_without_weights(monkeypatch):
    monkeypatch.setattr(
        "media_pipeline.visual.vlm.probe_vlm",
        lambda *_args, **_kwargs: (True, "mlx-community/Qwen3.8-27B-4bit"),
    )
    monkeypatch.setattr(
        "media_pipeline.summary_llm._probe_qwen_8bit",
        lambda _config: (False, "weights missing. hf download mlx-community/Qwen3.8-27B-8bit"),
    )
    rows = catalog_summary_backends(AppConfig(), None)
    by_key = {item["key"]: item for item in rows}
    assert by_key["qwen-8bit"]["available"] is False
    assert by_key["qwen-8bit-xhigh"]["available"] is False
    assert "hf download" in by_key["qwen-8bit"]["detail"]


def test_resolve_qwen_xhigh_enables_thinking():
    vision = FakeVision()
    backends = resolve_summary_backends(AppConfig(), vision, "qwen-xhigh")
    assert [item.key for item in backends] == ["qwen-xhigh"]
    assert backends[0].label == "qwen-3.8-27B 4bit thinking xhigh"
    backends[0].generate("hello", [], 128)
    assert vision.calls[-1]["enable_thinking"] is True
    assert vision.calls[-1]["reasoning_effort"] == "xhigh"
    assert vision.calls[-1]["max_tokens"] >= 8192


def test_render_summary_runs_compares_models():
    text = render_summary_runs(
        [
            {"key": "qwen", "label": "Qwen3.8 (local)", "markdown": "本地版", "duration_sec": 1.2},
            {"key": "gemini", "label": "Gemini (gemini-2.5-flash)", "markdown": "云端版", "duration_sec": 12},
        ]
    )
    assert text.index("### Gemini (gemini-2.5-flash) · 12s") < text.index("### Qwen3.8 (local) · 1.2s")
    assert "本地版" in text
    assert "云端版" in text


def test_format_run_duration_buckets():
    assert format_run_duration(3.4) == "3.4s"
    assert format_run_duration(12) == "12s"
    assert format_run_duration(63) == "1m 3s"
    assert format_run_duration(None) == ""


def test_coerce_summary_runs_seeds_legacy_markdown():
    runs = coerce_summary_runs(
        {
            "markdown": "旧简报",
            "model": "local-qwen",
            "label": "Qwen3.8 (local)",
            "updated_at": "2026-08-25T00:00:00+00:00",
        }
    )
    assert len(runs) == 1
    assert runs[0]["markdown"] == "旧简报"
    assert runs[0]["label"] == "Qwen3.8 (local)"
    assert runs[0]["model"] == "local-qwen"
    assert runs[0]["thinking"] == ""


def test_split_summary_thinking_keeps_answer():
    thinking, answer = split_summary_thinking("<think>plan</think>\n\n核心结论：完成。\n")
    assert thinking == "plan"
    assert "核心结论：完成。" in answer
    parsed_think, parsed_answer = parse_summary_output("draft</think>\n\n核心结论：拆开。\n")
    assert "draft" in parsed_think
    assert parsed_answer == "核心结论：拆开。"
    assert "<think>" not in parsed_answer
    assert "</think>" not in parsed_answer


def test_coerce_summary_runs_extracts_orphan_thinking():
    runs = coerce_summary_runs(
        {
            "runs": [
                {
                    "key": "qwen-xhigh",
                    "label": "Qwen3.8 27B thinking xhigh",
                    "markdown": "先列提纲\n</think>\n\n核心结论：正文。\n",
                }
            ]
        }
    )
    assert runs[0]["thinking"] == "先列提纲"
    assert runs[0]["markdown"] == "核心结论：正文。"


def test_render_summary_runs_folds_thinking():
    text = render_summary_runs(
        [
            {
                "label": "Qwen3.8 27B thinking xhigh",
                "markdown": "正文",
                "thinking": "推理过程",
            }
        ]
    )
    assert "<details>" in text
    assert "<summary>Thinking</summary>" in text
    assert "推理过程" in text
    assert text.index("推理过程") < text.index("正文")


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
    assert result["runs"][0]["duration_sec"] >= 0
    assert result["runs"][1]["duration_sec"] >= 0
    assert " · " in result["markdown"]


def test_run_summary_backends_appends_history(tmp_path: Path):
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
    )
    third = run_summary_backends(
        artifacts,
        [qwen],
        prompt="总结",
        metadata=_metadata(),
        existing_runs=list(second["runs"]),
    )
    assert [item["key"] for item in second["runs"]] == ["qwen", "gemini"]
    assert [item["key"] for item in third["runs"]] == ["qwen", "gemini", "qwen"]
    assert "本地" in third["markdown"]
    assert "云端" in third["markdown"]
    assert all(item["duration_sec"] >= 0 for item in third["runs"])
    assert third["markdown"].index("### Qwen") < third["markdown"].index("### Gemini")


def test_run_summary_backends_keeps_failed_attempt(tmp_path: Path):
    artifacts = _store_with_transcript(tmp_path)

    def boom(*_args, **_kwargs):
        raise RuntimeError("quota")

    bad = SummaryBackend(key="gemini", label="Gemini", model_id="gemini", generate_fn=boom)
    result = run_summary_backends(artifacts, [bad], prompt="总结", metadata=_metadata())
    assert result["status"] == "failed"
    assert result["runs"][0]["status"] == "failed"
    assert result["runs"][0]["duration_sec"] >= 0
    assert "quota" in result["markdown"]


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


def _stub_chat_template(monkeypatch, fake_apply):
    import sys
    from types import ModuleType

    pkg = ModuleType("mlx_vlm")
    sub = ModuleType("mlx_vlm.prompt_utils")
    sub.apply_chat_template = fake_apply
    pkg.prompt_utils = sub
    monkeypatch.setitem(sys.modules, "mlx_vlm", pkg)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", sub)


def test_format_prompt_passes_reasoning_effort(monkeypatch):
    seen = {}

    def fake_apply(processor, config, prompt, **kwargs):
        seen.update(kwargs)
        return "PROMPT"

    _stub_chat_template(monkeypatch, fake_apply)
    from media_pipeline.visual.vlm import _format_prompt

    text = _format_prompt(
        object(),
        {"model_type": "qwen3_5"},
        "hi",
        num_images=0,
        enable_thinking=True,
        reasoning_effort="medium",
    )
    assert text == "PROMPT"
    assert seen["enable_thinking"] is True
    assert seen["reasoning_effort"] == "medium"


def test_format_prompt_instruct_disables_thinking(monkeypatch):
    seen = {}

    def fake_apply(processor, config, prompt, **kwargs):
        seen.update(kwargs)
        return "PROMPT"

    _stub_chat_template(monkeypatch, fake_apply)
    from media_pipeline.visual.vlm import _format_prompt

    _format_prompt(object(), {"model_type": "qwen3_5"}, "hi", num_images=1)
    assert seen["enable_thinking"] is False
    assert "reasoning_effort" not in seen
