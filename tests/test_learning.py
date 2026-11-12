import json
from pathlib import Path

from fastapi.testclient import TestClient

from media_pipeline import learning
from media_pipeline.learning import run_learning
from media_pipeline.models import Task, TaskStatus
from media_pipeline.store import TaskStore
from tests.test_bulk import _config, _job

LIBRARY = {
    "talks": [
        {"slug": "s1", "title": "Talk One", "videoId": "v1", "event": "WF 2026", "durationMs": 1500000,
         "speakers": [{"name": "Ada"}], "topics": [{"name": "Evals"}], "summary": "Verifiable evals win."},
        {"slug": "s2", "title": "Talk Two", "videoId": "v2", "event": "WF 2026", "durationMs": 900000,
         "speakers": [{"name": "Bo"}], "topics": [{"name": "Evals"}], "summary": "Graders and reward hacking."},
        {"slug": "s3", "title": "Talk Three", "videoId": "v3", "event": "Summit 2025", "durationMs": 600000,
         "speakers": [], "topics": [{"name": "Evals"}], "summary": "Eval-driven development."},
        {"slug": "x9", "title": "Not in the job", "videoId": "zz", "summary": "?"},
    ],
    "topics": [
        {"slug": "evals", "name": "Evals", "description": "Talks on evals.", "talkSlugs": ["s1", "s2", "s3", "x9"]},
        {"slug": "tiny", "name": "Tiny", "description": "", "talkSlugs": ["s3"]},
    ],
}


class FakeModel:
    def __init__(self):
        self.prompts = []

    def __call__(self, prompt, max_tokens=16000):
        self.prompts.append(prompt)
        if "逐场打分" in prompt:
            return '好的：[{"id": "T1", "score": 5, "track": "评测、验证器与奖励", "why": "核心"},' \
                   ' {"id": "T2", "score": 4, "track": "评测、验证器与奖励", "why": "防作弊"},' \
                   ' {"id": "T3", "score": 2, "track": "其他", "why": "一般"}]'
        if "8 周学习路线" in prompt:
            return "# 总览\n第一周看 [T1]，再看 [T2, T3]。"
        if "分批提炼" in prompt or "各场演讲的总结" in prompt:
            return "## 一句话结论\n评测要可验证 [T1]、[T2]\n\n## 方法谱系\n- 用独立 grader [T2]"
        return "### 子主题\n- 可验证评测 [T1]"


def _setup(tmp_path):
    config = _config(tmp_path)
    store = _job(config, ids=("v1", "v2", "v3"), title="AI Engineer")
    art = config.paths.artifacts / "v1"
    art.mkdir(parents=True)
    (art / "summary.json").write_text(json.dumps({"status": "completed", "markdown": "### Flash · 20s\n核心结论：评测要可验证。"}))
    (config.paths.artifacts / "v2").mkdir(parents=True)
    (config.paths.artifacts / "v2" / "aie.json").write_text(json.dumps({"chapters": [{"start": 75, "title": "Why graders"}]}))
    tasks = TaskStore(config.paths.db)
    notes = config.notes_dir() / "AI Engineer"
    tasks.insert(Task(id="t1", url="https://www.youtube.com/watch?v=v1", asr_model="ai-engineer", video_id="v1",
                      status=TaskStatus.completed, note_path=str(notes / "Talk One.md"),
                      extra={"collection": "AI Engineer"}))
    store.execute("UPDATE videos SET task_id='t1' WHERE id='v1'")
    return config, store, tasks


def test_topics_scores_and_plan_cite_talks_with_links(tmp_path: Path):
    config, store, _tasks = _setup(tmp_path)
    out = config.notes_dir() / "AI Engineer" / "_学习"
    model = FakeModel()
    stats = run_learning(config, store, LIBRARY, out, generate=model, min_topic_talks=2)
    assert stats["topics"] == 1 and stats["skipped_topics"] == 1 and stats["scored"] == 3
    topic = (out / "主题" / "Evals.md").read_text()
    assert topic.startswith("# Evals\n\n> AI Engineer 主题综述 · 3 场演讲")
    assert "[[Talk One]]、[Talk Two](https://www.youtube.com/watch?v=v2)" in topic  # note link, else YouTube
    assert "核心结论：评测要可验证。" in model.prompts[0] and "Graders and reward hacking." in model.prompts[0]
    assert "### Flash" not in model.prompts[0]  # our run header is not fed back
    plan = (out / "学习路线.md").read_text()
    assert "第一周看 [[Talk One]]" in plan and "[Talk Three](https://www.youtube.com/watch?v=v3)" in plan
    plan_prompt = next(p for p in model.prompts if "8 周学习路线" in p)
    assert "Talk Two" in plan_prompt and "Talk Three" not in plan_prompt  # only the high scorers
    assert "01:15 Why graders" in plan_prompt  # chapters help pick what to watch first
    scores = json.loads((out / "scores.json").read_text())
    assert scores["T1"]["score"] == 5 and scores["T1"]["video_id"] == "v1"


def test_big_topics_are_read_in_batches_then_merged(tmp_path: Path, monkeypatch):
    config, store, _tasks = _setup(tmp_path)
    monkeypatch.setattr(learning, "CHUNK_CHARS", 120)
    model = FakeModel()
    run_learning(config, store, LIBRARY, tmp_path / "out", generate=model, min_topic_talks=2, workers=1)
    batches = [p for p in model.prompts if "批演讲" in p]
    assert len(batches) >= 2 and f"第 1/{len(batches)} 批" in batches[0]
    assert any("分批提炼" in p for p in model.prompts)


def test_learn_pages_list_and_render_the_material(tmp_path: Path):
    from media_pipeline.api import create_app
    from tests.test_api import DummyWorker

    config, store, tasks = _setup(tmp_path)
    out = config.notes_dir() / "AI Engineer" / "_学习"
    run_learning(config, store, LIBRARY, out, generate=FakeModel(), min_topic_talks=2)
    client = TestClient(create_app(config, store=tasks, worker=DummyWorker()))
    index = client.get("/v1/learn").json()["collections"]
    assert index[0]["collection"] == "AI Engineer" and index[0]["plan"] == "学习路线.md"
    assert index[0]["topics"] == [{"name": "Evals", "path": "主题/Evals.md", "talks": 3}]
    doc = client.get("/v1/learn/doc", params={"collection": "AI Engineer", "path": "学习路线.md"}).json()
    assert doc["title"] == "学习路线" and doc["links"] == {"Talk One": "/#/n/t1"}
    assert client.get("/v1/learn/doc", params={"collection": "AI Engineer", "path": "../../tasks.sqlite3"}).status_code == 404
    assert client.get("/v1/learn/doc", params={"collection": "Nope", "path": "学习路线.md"}).status_code == 404
    assert client.get("/learn").status_code == 200
