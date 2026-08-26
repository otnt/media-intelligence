from datetime import datetime

from media_pipeline.dashboard_view import flatten_groups, organize_task_payloads
from media_pipeline.models import source_key, source_label


def test_xiaohongshu_and_rednote_are_the_same_source():
    assert source_key("Xiaohongshu") == "rednote"
    assert source_key("RedNote") == "rednote"
    assert source_key("", "https://www.rednote.com/explore/abc") == "rednote"
    assert source_key("", "https://www.xiaohongshu.com/explore/abc") == "rednote"
    assert source_label("Xiaohongshu") == "RedNote"
    assert source_key("YouTube") == "youtube"
    assert source_key("Bilibili") == "bilibili"
    assert source_label("YouTube") == "YouTube"


def test_organize_defaults_today_grouped_by_source_newest_first():
    now = datetime(2026, 8, 25, 21, 0, 0)
    payloads = [
        _task("old-yt", "YouTube", "2026-08-24 09:00", "Yesterday YT"),
        _task("new-xhs", "Xiaohongshu", "2026-08-25 18:12", "New XHS"),
        _task("new-bili", "Bilibili", "2026-08-25 21:03", "New Bili"),
        _task("new-red", "Xiaohongshu", "2026-08-25 20:01", "New Red"),
        _task("new-yt", "YouTube", "2026-08-25 10:00", "New YT"),
    ]
    groups = organize_task_payloads(payloads, now=now)
    assert [item["key"] for item in groups] == ["bilibili", "rednote", "youtube"]
    assert [item["id"] for item in groups[0]["tasks"]] == ["new-bili"]
    assert [item["id"] for item in groups[1]["tasks"]] == ["new-red", "new-xhs"]
    assert flatten_groups(groups)[0]["id"] == "new-bili"
    assert all(item["requested_at"].startswith("2026-08-25") for item in flatten_groups(groups))


def test_organize_all_none_requested_asc():
    payloads = [
        _task("b", "Bilibili", "2026-08-25 12:00", "B"),
        _task("a", "YouTube", "2026-08-24 09:00", "A"),
    ]
    groups = organize_task_payloads(
        payloads,
        time_filter="all",
        group="none",
        order="requested_asc",
        now=datetime(2026, 8, 25),
    )
    assert len(groups) == 1
    assert [item["id"] for item in groups[0]["tasks"]] == ["a", "b"]


def _task(task_id: str, platform: str, created_at: str, title: str) -> dict:
    return {
        "id": task_id,
        "platform": platform,
        "source": source_key(platform),
        "source_label": source_label(platform),
        "requested_at": created_at,
        "created_at": created_at,
        "updated_at": created_at,
        "title": title,
        "status": "completed",
        "url": "",
    }
