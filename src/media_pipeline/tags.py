"""User-managed manual tags on tasks (not platform/status chips)."""

from __future__ import annotations

from typing import Any

from media_pipeline.models import Task

TAG_MAX_LEN = 40
TAGS_EXTRA_KEY = "tags"


def normalize_tag(value: str | None) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return ""
    return text[:TAG_MAX_LEN]


def normalize_tags(values: list[str] | tuple[str, ...] | None) -> list[str]:
    """Deduplicate tags case-insensitively while preserving first-seen spelling/order."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in values or []:
        tag = normalize_tag(raw)
        if not tag:
            continue
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(tag)
    return out


def tags_from_extra(extra: dict[str, Any] | None) -> list[str]:
    payload = extra or {}
    raw = payload.get(TAGS_EXTRA_KEY)
    if not isinstance(raw, list):
        return []
    return normalize_tags([str(item) for item in raw])


def task_tags(task: Task) -> list[str]:
    return tags_from_extra(task.extra)


def set_task_tags(task: Task, values: list[str] | tuple[str, ...] | None) -> list[str]:
    tags = normalize_tags(values)
    task.extra[TAGS_EXTRA_KEY] = tags
    return tags


def payload_has_tag(payload: dict[str, Any], tag: str | None) -> bool:
    wanted = normalize_tag(tag)
    if not wanted:
        return True
    needle = wanted.casefold()
    tags = payload.get("tags")
    if not isinstance(tags, list):
        return False
    return any(normalize_tag(str(item)).casefold() == needle for item in tags)


def collect_manual_tags(tasks: list[Task]) -> list[str]:
    """Unique manual tags across tasks, sorted case-insensitively."""
    seen: dict[str, str] = {}
    for task in tasks:
        for tag in task_tags(task):
            key = tag.casefold()
            seen.setdefault(key, tag)
    return [seen[key] for key in sorted(seen.keys())]
