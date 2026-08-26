from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from media_pipeline.models import SOURCE_LABELS, source_key, source_label


DASHBOARD_FILTERS = ("today", "7d", "all")
DASHBOARD_GROUPS = ("source", "none", "status", "day")
DASHBOARD_ORDERS = ("requested_desc", "requested_asc", "updated_desc", "title")
DASHBOARD_FILTER_OPTIONS = (
    ("today", "Today"),
    ("7d", "Last 7 days"),
    ("all", "All"),
)
DASHBOARD_GROUP_OPTIONS = (
    ("source", "Source"),
    ("none", "None"),
    ("status", "Status"),
    ("day", "Day"),
)
DASHBOARD_ORDER_OPTIONS = (
    ("requested_desc", "Requested newest"),
    ("requested_asc", "Requested oldest"),
    ("updated_desc", "Updated newest"),
    ("title", "Title"),
)


def normalize_dashboard_filter(value: str | None, default: str = "today") -> str:
    return _pick(value, DASHBOARD_FILTERS, default)


def normalize_dashboard_group(value: str | None, default: str = "source") -> str:
    return _pick(value, DASHBOARD_GROUPS, default)


def normalize_dashboard_order(value: str | None, default: str = "requested_desc") -> str:
    return _pick(value, DASHBOARD_ORDERS, default)


def dashboard_options() -> dict[str, list[dict[str, str]]]:
    return {
        "filters": [{"id": key, "label": label} for key, label in DASHBOARD_FILTER_OPTIONS],
        "groups": [{"id": key, "label": label} for key, label in DASHBOARD_GROUP_OPTIONS],
        "orders": [{"id": key, "label": label} for key, label in DASHBOARD_ORDER_OPTIONS],
    }


def organize_task_payloads(
    payloads: list[dict[str, Any]],
    *,
    time_filter: str = "today",
    group: str = "source",
    order: str = "requested_desc",
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Filter, sort, and group public task payloads for the dashboard sidebar."""
    current = now or datetime.now()
    filtered = _filter_payloads(payloads, normalize_dashboard_filter(time_filter), current)
    sorted_payloads = _sort_payloads(filtered, normalize_dashboard_order(order))
    return _group_payloads(sorted_payloads, normalize_dashboard_group(group))


def flatten_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for group in groups:
        tasks.extend(list(group.get("tasks") or []))
    return tasks


def _filter_payloads(payloads: list[dict[str, Any]], time_filter: str, now: datetime) -> list[dict[str, Any]]:
    if time_filter == "all":
        return list(payloads)
    today = now.strftime("%Y-%m-%d")
    if time_filter == "today":
        return [item for item in payloads if _stamp_date(item) == today]
    start = (now.date() - timedelta(days=6)).isoformat()
    return [item for item in payloads if _stamp_date(item) >= start]


def _sort_payloads(payloads: list[dict[str, Any]], order: str) -> list[dict[str, Any]]:
    if order == "requested_asc":
        return sorted(payloads, key=lambda item: (_requested_at(item), str(item.get("id") or "")))
    if order == "title":
        return sorted(payloads, key=lambda item: (_title_key(item), str(item.get("id") or "")))
    if order == "updated_desc":
        return sorted(
            payloads,
            key=lambda item: (_updated_at(item), str(item.get("id") or "")),
            reverse=True,
        )
    return sorted(
        payloads,
        key=lambda item: (_requested_at(item), str(item.get("id") or "")),
        reverse=True,
    )


def _group_payloads(payloads: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    if group == "none":
        return [{"key": "all", "label": "", "tasks": payloads}]
    groups: list[dict[str, Any]] = []
    index: dict[str, int] = {}
    for item in payloads:
        key, label = _group_identity(item, group)
        slot = index.get(key)
        if slot is None:
            index[key] = len(groups)
            groups.append({"key": key, "label": label, "tasks": []})
            slot = index[key]
        groups[slot]["tasks"].append(item)
    return groups


def _group_identity(item: dict[str, Any], group: str) -> tuple[str, str]:
    if group == "status":
        status = str(item.get("status") or "unknown")
        return status, status.replace("_", " ")
    if group == "day":
        day = _stamp_date(item) or "unknown"
        return day, day
    key = str(item.get("source") or source_key(str(item.get("platform") or ""), str(item.get("url") or "")))
    label = str(item.get("source_label") or SOURCE_LABELS.get(key) or source_label(str(item.get("platform") or "")))
    return key, label


def _requested_at(item: dict[str, Any]) -> str:
    return str(item.get("requested_at") or item.get("created_at") or "")


def _updated_at(item: dict[str, Any]) -> str:
    return str(item.get("updated_at") or _requested_at(item))


def _stamp_date(item: dict[str, Any]) -> str:
    return _requested_at(item)[:10]


def _title_key(item: dict[str, Any]) -> str:
    return str(item.get("title") or item.get("video_id") or item.get("url") or "").lower()


def _pick(value: str | None, allowed: tuple[str, ...], default: str) -> str:
    key = (value or "").strip().lower()
    if key in allowed:
        return key
    return default
