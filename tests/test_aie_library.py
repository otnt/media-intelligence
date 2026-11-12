import io
import json
import urllib.error
from pathlib import Path

from fastapi.testclient import TestClient

from media_pipeline.aie_library import BASE, Library, import_job, pick_spread, slide_entries, to_transcript
from media_pipeline.artifacts import ArtifactStore
from media_pipeline.store import TaskStore
from tests.test_bulk import _config, _job, _row

VID = "CEvIs9y1uog"
MANIFEST_URL = "https://assets.ai.engineer/objects/manifest.json"
TALK = {
    "slug": "agent-skills", "title": "Don't Build Agents, Build Skills Instead", "url": "https://ai.engineer/talks/x",
    "videoId": VID, "event": "AI Engineer Code 2025", "durationMs": 982000,
    "speakers": [{"name": "Barry Zhang"}, {"name": "Mahesh Murag"}], "topics": [{"name": "Agent engineering"}],
    "summary": "Build reusable skills.", "slideText": "...", "slides": {"url": MANIFEST_URL},
}
MANIFEST = {"index": {"groups": [
    {"title": "Year of agents", "kind": "slide", "frames": [
        {"src": "https://assets.ai.engineer/a.webp", "captureMs": 70500, "alt": "Timeline", "text": "Year of agents FEB — TODAY"}]},
    {"title": "Code is the universal interface", "kind": "slide", "frames": [
        {"src": "https://assets.ai.engineer/b.webp", "captureMs": 100500, "alt": "Agent icons", "text": "Code is the universal interface"}]},
    {"title": "Skills", "kind": "slide", "frames": [
        {"src": "https://assets.ai.engineer/c.webp", "captureMs": 300000, "alt": "Folder", "text": "Skills are folders"}]},
    {"title": "Speaker", "kind": "speaker", "frames": [{"src": "https://assets.ai.engineer/d.webp", "captureMs": 1}]},
]}}


class FakeSite:
    def __init__(self, throttle_once=False):
        self.calls: list[str] = []
        self.throttle_once = throttle_once
        self.pages = {
            f"{BASE}/talks?format=json": [TALK],
            f"{BASE}/chapters?format=json": [
                {"talkSlug": "agent-skills", "title": "Why skills", "startMs": 181000, "endMs": 400000},
                {"talkSlug": "agent-skills", "title": "Intro", "startMs": 0, "endMs": 181000}],
            f"{BASE}/transcripts?format=json": [
                {"talkSlug": "agent-skills", "jsonUrl": f"{BASE}/transcript?slug=agent-skills&format=json"}],
            f"{BASE}/transcript?slug=agent-skills&format=json": [
                {"startMs": 21270, "text": "Alright, good morning."}, {"startMs": 80790, "text": "We used to think  agents differ."}],
            MANIFEST_URL: MANIFEST,
        }

    def __call__(self, url):
        self.calls.append(url)
        if self.throttle_once and len(self.calls) == 1:
            self.throttle_once = False
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", {"Retry-After": "2"}, io.BytesIO(b""))
        if url.endswith(".webp"):
            return b"RIFF....WEBP" + url.encode()
        if url.endswith("format=csv"):
            return b"slug,title\n"
        if url.endswith(".md"):
            return b"# talk page"
        if url.endswith("/status"):
            return b'{"status": "available"}'
        if url.endswith("format=json") and url.split("/")[-1].split("?")[0] in ("speakers", "topics", "organizations"):
            return b"[]"
        return json.dumps(self.pages[url]).encode()


def test_paragraphs_become_segments_and_slides_are_spread_over_the_talk():
    transcript = to_transcript([{"startMs": 80790, "text": "b"}, {"startMs": 21270, "text": "a"}], 982000)
    assert [(s.start, s.end, s.text) for s in transcript.segments] == [(21.27, 80.79, "a"), (80.79, 982.0, "b")]
    assert {s.speaker for s in transcript.segments} == {"SPEAKER_00"} and transcript.model == "ai-engineer"
    slides = slide_entries(MANIFEST)
    assert [s["title"] for s in slides] == ["Year of agents", "Code is the universal interface", "Skills"]  # no speaker shots
    assert [s["title"] for s in pick_spread(slides, 2)] == ["Year of agents", "Skills"]


def test_429_is_retried_after_retry_after():
    naps = []
    site = FakeSite(throttle_once=True)
    library = Library(fetch=site, sleep=naps.append, api_interval=0, asset_interval=0)
    assert library.json(f"{BASE}/talks?format=json")[0]["slug"] == "agent-skills"
    assert 2.0 in naps and len(site.calls) == 2


def test_import_saves_what_the_pipeline_reads_and_finishes_the_transcript_stage(tmp_path: Path):
    from media_pipeline.summary import _load_chapters, _load_slides, build_model_prompt, collect_stills

    config = _config(tmp_path)
    store = _job(config, ids=(VID, "notATalk001"), asr_model="fun-asr", youtube_extras=True, title="AI Engineer")
    site = FakeSite()
    stats = import_job(config, store, library=Library(fetch=site, sleep=lambda s: None, api_interval=0, asset_interval=0),
                       slides_per_talk=2)
    assert stats == {"talks": 1, "matched": 1, "transcripts": 1, "slides": 1, "images": 2, "failed": 0}
    row, other = _row(store, VID), _row(store, "notATalk001")
    assert row["asr_status"] == "done" and row["transcript_source"] == "ai-engineer" and row["segments"] == 2
    assert row["captions_status"] == "skipped" and row["chapters"] == 2 and row["status"] == "pending"
    assert other["asr_status"] == "pending" and other["transcript_source"] == ""

    artifacts = ArtifactStore(config.paths.artifacts, VID)
    assert artifacts.load_transcript("ai-engineer").segments[1].text == "We used to think agents differ."
    metadata = artifacts.load_metadata()
    assert metadata.author == "Barry Zhang, Mahesh Murag" and metadata.duration == 982.0
    assert [c["title"] for c in _load_chapters(artifacts)] == ["Intro", "Why skills"]
    assert sorted(p.name for p in (artifacts.root / "slides").iterdir()) == ["a.webp", "c.webp", "slides.json"]
    stills = collect_stills(artifacts)
    assert [s.caption for s in stills] == ["Year of agents — Timeline", "Skills — Folder"] and stills[1].timestamp == 300.0
    prompt = build_model_prompt("Summarize.", metadata, "text", chapters=_load_chapters(artifacts),
                                slides=_load_slides(artifacts))
    assert "Slides (title and on-slide text" in prompt and "- [00:01:40] Code is the universal interface" in prompt

    # A second run fetches nothing per talk again: only the three library indexes.
    again = FakeSite()
    import_job(config, store, library=Library(fetch=again, sleep=lambda s: None, api_interval=0, asset_interval=0),
               slides_per_talk=2)
    assert len(again.calls) == 3

    from media_pipeline.api import create_app
    from tests.test_api import DummyWorker

    tasks = TaskStore(config.paths.db)
    client = TestClient(create_app(config, store=tasks, worker=DummyWorker()))
    created = client.post(f"/v1/bulk/aie/videos/{VID}/process")
    assert created.status_code == 202 and tasks.get(created.json()["id"]).asr_model == "ai-engineer"
    text = client.get(f"/v1/bulk/aie/transcript/{VID}").text
    assert "[00:00:21] SPEAKER_00\nAlright, good morning. We used to think agents differ." in text



def test_archive_keeps_every_published_file_and_reuses_downloaded_slides(tmp_path: Path):
    from media_pipeline.aie_library import archive_library

    artifacts_root = tmp_path / "artifacts"
    (artifacts_root / VID / "slides").mkdir(parents=True)
    (artifacts_root / VID / "slides" / "a.webp").write_bytes(b"already here")
    site = FakeSite()
    dest = tmp_path / "nas" / "official"
    stats = archive_library(dest, library=Library(fetch=site, sleep=lambda s: None, api_interval=0, asset_interval=0),
                            artifacts_root=artifacts_root)
    talk = dest / "talks" / f"{VID}-agent-skills"
    assert sorted(p.name for p in talk.iterdir()) == ["slides", "slides.json", "talk.json", "talk.md", "transcript.json"]
    assert sorted(p.name for p in (talk / "slides").iterdir()) == ["a.webp", "b.webp", "c.webp"]
    assert (talk / "slides" / "a.webp").read_bytes() == b"already here"  # copied, not fetched again
    assert not any(url.endswith("a.webp") for url in site.calls)
    assert {p.name for p in (dest / "index").iterdir()} >= {"talks.json", "talks.csv", "chapters.csv", "status.json", "data.md"}
    assert stats["slides"] == 2 and stats["slides_copied"] == 1 and stats["markdown"] == 1 and stats["failed"] == 0
    assert (dest / "README.md").read_text().startswith("# AI Engineer official talk library")

    again = FakeSite()
    archive_library(dest, library=Library(fetch=again, sleep=lambda s: None, api_interval=0, asset_interval=0),
                    artifacts_root=artifacts_root)
    assert again.calls == ["https://ai.engineer/api/data/status"]  # resumable: only the status is refreshed
