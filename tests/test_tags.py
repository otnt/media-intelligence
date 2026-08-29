from media_pipeline.models import Task, TaskStatus
from media_pipeline.tags import (
    collect_manual_tags,
    normalize_tag,
    normalize_tags,
    payload_has_tag,
    set_task_tags,
    tags_from_extra,
    task_tags,
)


def test_normalize_tags_strips_dedupes_and_caps_length():
    assert normalize_tag("  hello   world  ") == "hello world"
    assert normalize_tag("") == ""
    assert len(normalize_tag("x" * 100)) == 40
    assert normalize_tags(["Foo", "foo", " Bar ", "", "Baz"]) == ["Foo", "Bar", "Baz"]


def test_task_tags_round_trip_via_extra():
    task = Task(id="t1", url="https://youtu.be/a", asr_model="qwen3-asr-1.7b")
    assert task_tags(task) == []
    assert set_task_tags(task, ["research", " research ", "CLIP"]) == ["research", "CLIP"]
    assert task.extra["tags"] == ["research", "CLIP"]
    assert tags_from_extra(task.extra) == ["research", "CLIP"]
    assert task.to_public()["tags"] == ["research", "CLIP"]


def test_collect_manual_tags_sorted_unique():
    a = Task(id="a", url="u", asr_model="qwen3-asr-1.7b", status=TaskStatus.queued, extra={"tags": ["zeta", "Alpha"]})
    b = Task(id="b", url="u", asr_model="qwen3-asr-1.7b", status=TaskStatus.queued, extra={"tags": ["alpha", "beta"]})
    assert collect_manual_tags([a, b]) == ["Alpha", "beta", "zeta"]


def test_payload_has_tag_case_insensitive():
    assert payload_has_tag({"tags": ["Deep Dive"]}, "deep dive")
    assert not payload_has_tag({"tags": ["Deep Dive"]}, "other")
    assert payload_has_tag({"tags": ["x"]}, None)
    assert payload_has_tag({"tags": ["x"]}, "")
