"""Turn a bulk collection (the AI Engineer talks) into study material: topic syntheses and a learning plan.

``media-pipeline bulk learn <job>`` writes, under ``<notes>/<collection>/_学习/``:

* ``主题/<topic>.md``: one synthesis per official topic (problems, approaches by sub-theme with numbers,
  consensus, disagreements, the talks worth watching, actions). Big topics are read in batches first
  (map), then merged (reduce).
* ``scores.json``: every talk scored 0-5 for the reader's goal, with a learning track and a reason.
* ``学习路线.md``: an 8-week plan built from the best-scoring talks and the topic conclusions.

The model cites talks by short ids (``[T12]``); ids are replaced afterwards with Obsidian links to the talk
notes (or YouTube links), so a citation can never point at the wrong talk. Inputs are the talk summaries
already written (``summary.json``), falling back to the organizers' one-paragraph summary.
No embeddings: at ~1,500 talks a model reads the summaries directly, which picks better than similarity.
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from media_pipeline.config import AppConfig

logger = logging.getLogger(__name__)

LEARN_DIR = "_学习"
TOPICS_DIR = "主题"
PLAN_NAME = "学习路线"
CHUNK_CHARS = 60_000
MIN_TOPIC_TALKS = 8
PROFILE = (
    "读者是资深分布式系统工程师（大型科技公司的存储/流计算方向，Staff 级别），正在转向前沿 AI 实验室的研究工程岗位，"
    "最关注递归自我改进（RSI）、Agent 与 harness 工程、评测与验证、强化学习环境与后训练基础设施、推理基础设施。"
    "母语中文；英文听力是弱项，习惯先读中文总结再看原视频。时间有限，偏好能动手、能放进作品集的练习。"
)
TRACKS = (
    "自我改进与 harness 工程", "评测、验证器与奖励", "强化学习环境与后训练基础设施", "编码 Agent",
    "推理与服务基础设施", "Agent 基础设施与沙箱", "上下文、记忆与检索", "安全与对齐", "其他",
)
_CITE = re.compile(r"\[(T\d+(?:\s*[,，、]\s*T\d+)*)\]")


@dataclass
class Talk:
    tid: str
    video_id: str
    slug: str
    title: str
    speakers: str
    event: str
    minutes: int
    topics: list[str]
    official_summary: str
    summary: str = ""  # ours (summary.json), when written
    note: str = ""  # note file name without .md, for [[links]]
    task_id: str = ""
    chapters: list[dict[str, Any]] = field(default_factory=list)

    def entry(self, with_summary: bool = True) -> str:
        head = f"[{self.tid}] {self.title} — {self.speakers or '?'} · {self.event or '?'} · {self.minutes} 分钟"
        body = self.summary or self.official_summary
        return f"{head}\n{body.strip()}" if with_summary and body else head


def generate_fn(config: AppConfig, model: str = "deepseek-v4.1-flash") -> Callable[[str, int], str]:
    from media_pipeline.summary_llm import resolve_summary_backends, strip_think

    backends = resolve_summary_backends(config, None, selection=model)
    if not backends:
        raise RuntimeError(f"Summary model {model!r} is not available")
    backend = backends[0]
    return lambda prompt, max_tokens=16000: strip_think(backend.generate(prompt, [], max_tokens))


def load_talks(config: AppConfig, store: Any, library: dict[str, Any]) -> list[Talk]:
    """The collection's talks: official records joined with our summaries, notes and tasks."""
    from media_pipeline.store import TaskStore

    rows = {row["id"]: row for row in store.rows()}
    tasks = TaskStore(config.paths.db)
    talks: list[Talk] = []
    for number, record in enumerate(library["talks"], start=1):
        vid = str(record.get("videoId") or "")
        if vid not in rows:
            continue
        artifacts = config.paths.artifacts / vid
        summary = ""
        summary_path = artifacts / "summary.json"
        if summary_path.is_file():
            try:
                data = json.loads(summary_path.read_text(encoding="utf-8"))
                if data.get("status") == "completed":
                    summary = _strip_run_header(str(data.get("markdown") or ""))
            except (OSError, ValueError):
                pass
        task_id = str(rows[vid].get("task_id") or "")
        note = ""
        if task_id:
            task = tasks.get(task_id)
            if task is not None and task.note_path:
                note = Path(task.note_path).stem
        chapters = []
        aie_path = artifacts / "aie.json"
        if aie_path.is_file():
            try:
                chapters = list(json.loads(aie_path.read_text(encoding="utf-8")).get("chapters") or [])
            except (OSError, ValueError):
                chapters = []
        talks.append(Talk(
            tid=f"T{number}", video_id=vid, slug=str(record.get("slug") or ""), title=str(record.get("title") or vid),
            speakers=", ".join(s.get("name", "") for s in record.get("speakers") or [] if s.get("name")),
            event=str(record.get("event") or ""), minutes=round(float(record.get("durationMs") or 0) / 60000),
            topics=[t.get("name", "") for t in record.get("topics") or [] if t.get("name")],
            official_summary=str(record.get("summary") or ""), summary=summary, note=note, task_id=task_id,
            chapters=chapters,
        ))
    return talks


def _strip_run_header(markdown: str) -> str:
    """Our summaries start with '### <model> · 27s'; drop that line."""
    lines = markdown.strip().splitlines()
    if lines and lines[0].startswith("### ") and "·" in lines[0]:
        lines = lines[1:]
    return "\n".join(lines).strip()


def chunks(talks: list[Talk], limit: int | None = None) -> list[list[Talk]]:
    limit = limit or CHUNK_CHARS
    out: list[list[Talk]] = [[]]
    size = 0
    for talk in talks:
        length = len(talk.entry())
        if out[-1] and size + length > limit:
            out.append([])
            size = 0
        out[-1].append(talk)
        size += length
    return [chunk for chunk in out if chunk]


TOPIC_SECTIONS = """## 一句话结论
## 这个领域在解决什么问题
## 方法谱系（按子主题：核心思路、代表做法、关键数据）
## 共识
## 分歧与未解决的问题
## 最值得看的 5–8 场（每场一句为什么）
## 给工程师的行动建议（3–5 条，可以直接上手）"""


def _map_prompt(topic: str, description: str, batch: list[Talk], index: int, total: int) -> str:
    talks = "\n\n".join(talk.entry() for talk in batch)
    return (
        f"你是资深 AI 工程研究员，在整理 AI Engineer 大会（2023–2026）的演讲。下面是主题「{topic}」"
        f"（官方说明：{description}）下的第 {index}/{total} 批演讲，共 {len(batch)} 场，每场有编号、讲者、场合和总结。\n"
        "请提炼这批演讲，供之后汇总成主题综述：\n"
        "1. 子主题分组：按子主题归类（每组列出编号）；\n"
        "2. 每组的核心思路、具体做法和关键数据（数字必须来自材料）；\n"
        "3. 演讲之间的共识与分歧；\n"
        "4. 这批里最值得精看的演讲（最多 5 场，说明理由）。\n"
        "只用材料里的信息，不要编造。每个论断都用编号引用，例如 [T12]。用中文，信息密度高，不复述原文。\n\n"
        f"{talks}"
    )


def _reduce_prompt(topic: str, description: str, material: str, count: int, partial: bool) -> str:
    source = "若干份分批提炼" if partial else "各场演讲的总结"
    return (
        f"你是资深 AI 工程研究员。下面是 AI Engineer 大会主题「{topic}」（官方说明：{description}）"
        f"共 {count} 场演讲的{source}。请写一份主题综述（中文 Markdown），用这些小节：\n{TOPIC_SECTIONS}\n\n"
        f"读者：{PROFILE}\n"
        "要求：信息密度高、不复述材料；数字和论断都要有编号出处，例如 [T12]；只用材料里的信息，不要编造；"
        "子主题要真正归纳出做法的差异，而不是逐场罗列。\n\n"
        f"{material}"
    )


def synthesize_topic(generate: Callable[[str, int], str], topic: str, description: str, talks: list[Talk],
                     pool: ThreadPoolExecutor | None = None) -> str:
    batches = chunks(talks)
    if len(batches) == 1:
        material = "\n\n".join(talk.entry() for talk in talks)
        return generate(_reduce_prompt(topic, description, material, len(talks), partial=False), 16000)
    prompts = [_map_prompt(topic, description, batch, i, len(batches)) for i, batch in enumerate(batches, start=1)]
    partials = list(pool.map(lambda p: generate(p, 12000), prompts)) if pool else [generate(p, 12000) for p in prompts]
    material = "\n\n".join(f"### 第 {i} 批\n{text}" for i, text in enumerate(partials, start=1))
    return generate(_reduce_prompt(topic, description, material, len(talks), partial=True), 16000)


def score_talks(generate: Callable[[str, int], str], talks: list[Talk], pool: ThreadPoolExecutor | None = None
                ) -> dict[str, dict[str, Any]]:
    """{tid: {"score", "track", "why"}} for every talk (official one-paragraph summaries keep it compact)."""
    def compact(talk: Talk) -> str:
        return f"[{talk.tid}] {talk.title} | 主题: {', '.join(talk.topics[:4])} | {talk.event}\n{talk.official_summary[:600]}"

    batches: list[list[Talk]] = [talks[i:i + 120] for i in range(0, len(talks), 120)]

    def run(batch: list[Talk]) -> list[dict[str, Any]]:
        listing = "\n\n".join(compact(t) for t in batch)
        prompt = (
            f"读者画像：{PROFILE}\n\n"
            "下面是 AI Engineer 大会的演讲列表（编号、标题、主题、场合、摘要）。请逐场打分：对这位读者的价值 0–5 分"
            "（5 = 必看，0 = 无关），给出所属学习方向（只能从这些里选一个："
            f"{'、'.join(TRACKS)}），以及不超过 20 字的理由。\n"
            '只输出 JSON 数组，不要别的文字：[{"id": "T12", "score": 4, "track": "评测、验证器与奖励", "why": "..."}]\n\n'
            f"{listing}"
        )
        return _json_list(generate(prompt, 16000))

    results = list(pool.map(run, batches)) if pool else [run(b) for b in batches]
    scores: dict[str, dict[str, Any]] = {}
    for items in results:
        for item in items:
            tid = str(item.get("id") or "")
            if tid:
                scores[tid] = {"score": int(item.get("score") or 0), "track": str(item.get("track") or "其他"),
                               "why": str(item.get("why") or "")}
    return scores


def plan(generate: Callable[[str, int], str], talks: list[Talk], scores: dict[str, dict[str, Any]],
         topic_conclusions: dict[str, str]) -> str:
    from media_pipeline.models import format_timestamp

    best = sorted((t for t in talks if scores.get(t.tid, {}).get("score", 0) >= 4),
                  key=lambda t: -scores[t.tid]["score"])[:120]
    by_track: dict[str, list[Talk]] = {}
    for talk in best:
        by_track.setdefault(scores[talk.tid]["track"], []).append(talk)
    sections = []
    for track, items in by_track.items():
        lines = [f"## {track}"]
        for talk in items:
            info = scores[talk.tid]
            chapters = "；".join(f"{format_timestamp(float(c.get('start') or 0))} {c.get('title', '')}"
                                for c in talk.chapters[:8])
            body = (talk.summary or talk.official_summary)[:700]
            lines.append(f"[{talk.tid}] {talk.title}（{info['score']} 分：{info['why']}）\n章节：{chapters or '无'}\n{body}")
        sections.append("\n\n".join(lines))
    conclusions = "\n".join(f"- {name}：{text}" for name, text in topic_conclusions.items() if text)
    prompt = (
        f"读者画像：{PROFILE}\n\n"
        "下面是为这位读者筛出的高价值演讲（按学习方向分组：编号、标题、得分和理由、章节、总结），以及相关主题综述的一句话结论。\n"
        "请设计一份 8 周学习路线（中文 Markdown）：\n"
        "# 总览：8 周的顺序和理由（先打基础，再到 RSI）\n"
        "# 第 N 周：<主题>（每周都包含）\n"
        "- 本周目标：学完能做到什么\n- 必看 3–5 场：编号 + 为什么看 + 先看哪几个章节/时间点\n"
        "- 动手项目：具体、一周内能做完、尽量能放进作品集\n- 设计笔记：一个 Staff 级别的题目，要讲清取舍\n"
        "- 自测：3 个问题\n"
        "# 最后：怎样把这些变成面试里的故事（系统设计、研究工程问答）\n"
        "要求：具体可执行；只推荐材料里的演讲；用编号引用，例如 [T12]。\n\n"
        f"主题结论：\n{conclusions}\n\n" + "\n\n".join(sections)
    )
    return generate(prompt, 16000)


def link_citations(markdown: str, talks: dict[str, Talk]) -> str:
    """[T12] → [[note]] (Obsidian) or [title](YouTube) when the talk has no note yet."""
    def one(tid: str) -> str:
        talk = talks.get(tid.strip())
        if talk is None:
            return tid
        if talk.note:
            return f"[[{talk.note}]]"
        return f"[{talk.title}](https://www.youtube.com/watch?v={talk.video_id})"

    return _CITE.sub(lambda m: "、".join(one(part) for part in re.split(r"\s*[,，、]\s*", m.group(1))), markdown)


def _json_list(text: str) -> list[dict[str, Any]]:
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        return []
    return [item for item in data if isinstance(item, dict)]


def first_line_after(markdown: str, heading: str) -> str:
    lines = markdown.splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith("##") and heading in line:
            for following in lines[index + 1:]:
                if following.strip() and not following.startswith("#"):
                    return following.strip()[:300]
    return ""


def run_learning(
    config: AppConfig,
    store: Any,
    library: dict[str, Any],
    out_dir: Path,
    *,
    generate: Callable[[str, int], str] | None = None,
    min_topic_talks: int = MIN_TOPIC_TALKS,
    workers: int = 4,
    log: Callable[[str], None] = logger.info,
) -> dict[str, Any]:
    generate = generate or generate_fn(config)
    talks = load_talks(config, store, library)
    by_tid = {talk.tid: talk for talk in talks}
    by_slug = {talk.slug: talk for talk in talks}
    topics_dir = out_dir / TOPICS_DIR
    topics_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {"talks": len(talks), "with_our_summary": sum(1 for t in talks if t.summary),
                             "topics": 0, "skipped_topics": 0, "failed": []}
    log(f"learning: {len(talks)} talks ({stats['with_our_summary']} with our summary)")
    conclusions: dict[str, str] = {}
    topics = sorted(library["topics"], key=lambda t: -len(t.get("talkSlugs") or []))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for topic in topics:
            members = [by_slug[s] for s in topic.get("talkSlugs") or [] if s in by_slug]
            name = str(topic.get("name") or topic.get("slug"))
            if len(members) < min_topic_talks:
                stats["skipped_topics"] += 1
                continue
            target = topics_dir / f"{_file_name(name)}.md"
            try:
                text = synthesize_topic(generate, name, str(topic.get("description") or ""), members, pool)
            except Exception as exc:  # noqa: BLE001 - one topic must not stop the rest
                stats["failed"].append(name)
                log(f"topic failed {name}: {exc}")
                continue
            conclusions[name] = first_line_after(text, "一句话结论")
            header = (f"# {name}\n\n> AI Engineer 主题综述 · {len(members)} 场演讲 · "
                      f"{time.strftime('%Y-%m-%d')} · 官方说明：{topic.get('description', '')}\n\n")
            target.write_text(header + link_citations(text, by_tid) + "\n", encoding="utf-8")
            stats["topics"] += 1
            log(f"topic {stats['topics']}: {name} ({len(members)} talks)")
        scores = score_talks(generate, talks, pool)
    (out_dir / "scores.json").write_text(json.dumps(
        {t.tid: {**scores.get(t.tid, {}), "title": t.title, "video_id": t.video_id} for t in talks},
        ensure_ascii=False, indent=1), encoding="utf-8")
    text = plan(generate, talks, scores, conclusions)
    header = (f"# {PLAN_NAME}\n\n> 基于 {len(talks)} 场 AI Engineer 演讲为你定制 · {time.strftime('%Y-%m-%d')} · "
              f"主题综述在「{TOPICS_DIR}」文件夹\n\n")
    (out_dir / f"{PLAN_NAME}.md").write_text(header + link_citations(text, by_tid) + "\n", encoding="utf-8")
    stats["scored"] = len(scores)
    log(f"learning done: {stats}")
    return stats


def _file_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in " -_&" else "-" for ch in name).strip()[:100] or "topic"
