from pathlib import Path

from media_pipeline.artifacts import ArtifactStore
from media_pipeline.models import NamedSegment, VideoMetadata
from media_pipeline.summary import summarize_task, take_speaker_names

LABELS = ["Speaker 1", "Speaker 2"]


def test_the_trailing_json_is_parsed_and_removed():
    text = '核心结论：……\n\n```json\n{"speakers": {"Speaker 1": "黄仁勋", "Speaker 2": null, "Speaker 9": "x"}}\n```'
    body, names = take_speaker_names(text, LABELS)
    assert body == "核心结论：……" and names == {"Speaker 1": "黄仁勋"}
    assert take_speaker_names("没有 JSON 的总结", LABELS) == ("没有 JSON 的总结", {})
    assert take_speaker_names("```json\n{broken\n```", LABELS)[1] == {}
    for tail in ('json\n{"speakers": {"Speaker 2": "周默"}}', '```json\n{"speakers": {"Speaker 2": "周默"}}',
                 '{"speakers": {"Speaker 2": "周默"}}\n'):
        body, names = take_speaker_names("总结正文\n\n" + tail, LABELS)
        assert body == "总结正文" and names == {"Speaker 2": "周默"}


class Provider:
    key = "fake"
    model_id = "fake-model"
    label = "Fake"
    accepts_images = False
    max_source_chars = 100_000

    def __init__(self):
        self.prompt = ""

    def generate(self, prompt, images, max_tokens):
        self.prompt = prompt
        return '核心结论：主持人采访黄仁勋。\n\n```json\n{"speakers": {"Speaker 1": "主持人", "Speaker 2": "黄仁勋"}}\n```'


def test_the_summary_model_names_the_speakers_in_transcript_and_note(tmp_path: Path):
    from media_pipeline.summary import apply_speaker_names

    artifacts = ArtifactStore(tmp_path, "v1")
    artifacts.save_named([NamedSegment(0, 5, "SPEAKER_00", "Speaker 1", "欢迎黄总"),
                          NamedSegment(5, 9, "SPEAKER_01", "Speaker 2", "谢谢")])
    artifacts.set_named_source("fun-asr")
    note = tmp_path / "note.md"
    note.write_text("# t\n\n## Transcript\n\n### [00:00:00] Speaker 1\n\n欢迎黄总\n", encoding="utf-8")
    artifacts.remember_note(note)
    meta = VideoMetadata(url="u", title="AI 是泡沫吗", platform="Bilibili", author="a", video_id="v1", duration=9.0,
                         published="", description="", thumbnail_url="", asr_model="fun-asr")
    provider = Provider()
    result = summarize_task(artifacts, provider, prompt="总结", metadata=meta)
    assert '"Speaker 1"、"Speaker 2"' in provider.prompt
    assert result["markdown"] == "核心结论：主持人采访黄仁勋。" and result["speakers"] == {"Speaker 1": "主持人", "Speaker 2": "黄仁勋"}
    assert apply_speaker_names(artifacts, result["speakers"])
    assert [s.speaker_label for s in artifacts.load_named()] == ["主持人", "黄仁勋"]
    assert artifacts.named_source() == "fun-asr"
    assert "黄仁勋" in note.read_text(encoding="utf-8") and "Speaker 2" not in note.read_text(encoding="utf-8")



def test_the_prompt_carries_the_description_date_and_length():
    from media_pipeline.summary import build_model_prompt

    meta = VideoMetadata(url="u", title="E251 推理芯片之战", platform="YouTube", author="硅谷101", video_id="v",
                         duration=3600.0, published="2026-11-01", description="本期嘉宾：红军、子阳。\n\n时间轴…",
                         thumbnail_url="", asr_model="fun-asr")
    prompt = build_model_prompt("总结", meta, "text")
    assert "Source: YouTube · 2026-11-01 · 60 min" in prompt and "本期嘉宾：红军、子阳。 时间轴…" in prompt
    meta.media_kind = "image"
    assert "Description" not in build_model_prompt("总结", meta, "text")
