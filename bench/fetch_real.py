"""Build the real-podcast benchmark samples from human transcripts + YouTube audio.

    uv run python bench/fetch_real.py ~/AIContent/bench/samples

Each source has a human-made transcript where every turn carries a speaker
name and a start time on the YouTube video's clock. We download that video's
audio (no browser cookies), keep audio up to the first turn boundary at/after
EXCERPT_SEC, and write the turns inside it. Turn end = next turn's start (timing: turn_start).
"""

from __future__ import annotations

import html
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

EXCERPT_SEC = 900.0

SOURCES = [
    {
        "id": "real-en-lex503",
        "language": "en",
        "transcript": "https://lexfridman.com/math-ai-crisis-transcript/",
        "youtube": "ZAzWPE9CItg",
        "format": "lex",
        "note": "Lex Fridman Podcast #503, 4 speakers. Human transcript, lightly cleaned of fillers.",
    },
    {
        "id": "real-en-lex503-full",
        "language": "en",
        "transcript": "https://lexfridman.com/math-ai-crisis-transcript/",
        "youtube": "ZAzWPE9CItg",
        "format": "lex",
        "full": True,
        "note": "Lex Fridman Podcast #503, whole episode (~3.7 h), for long-audio chunking tests.",
    },
    {
        "id": "real-zh-bmb232",
        "language": "zh",
        "transcript": "https://bumingbai.net/2026/09/09/ep-232-txt/",
        "youtube": "UISJGnJ1LpA",
        "format": "bumingbai",
        "note": "不明白播客 EP-232, 2 speakers. Human-edited transcript (文字版), not strictly verbatim.",
    },
]


def parse_lex(page: str) -> list[dict]:
    turns = []
    pattern = re.compile(
        r'<span class="ts-name">(.*?)</span>.*?\((\d+):(\d+):(\d+)\)</a>.*?<span class="ts-text">(.*?)</span>',
        re.S,
    )
    for name, h, m, s, text in pattern.findall(page):
        turns.append({"start": int(h) * 3600 + int(m) * 60 + int(s), "speaker": _clean(name), "text": _clean(text)})
    return turns


def parse_bumingbai(page: str) -> list[dict]:
    turns: list[dict] = []
    for block in re.findall(r'<p class="wp-block-paragraph">(.*?)</p>', page, re.S):
        text = _clean(block)
        match = re.match(r"\[(\d+):(\d+):(\d+)\]\s*([^：:]{1,12})[：:]\s*(.*)", text, re.S)
        if match:
            h, m, s, name, rest = match.groups()
            turns.append({"start": int(h) * 3600 + int(m) * 60 + int(s), "speaker": name.strip(), "text": rest.strip()})
        elif turns and text and not text.startswith("时间轴"):
            # Unlabelled paragraphs continue the previous speaker's turn.
            turns[-1]["text"] += "\n" + text
    return turns


def _clean(fragment: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", fragment)
    text = re.sub(r"<[^>]+>", "", text)
    return " ".join(html.unescape(text).split())


def _duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (benchmark fetch)"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        page = fetch(source["transcript"])
        turns = parse_lex(page) if source["format"] == "lex" else parse_bumingbai(page)
        if not turns:
            raise SystemExit(f"{source['id']}: no turns parsed")
        # Cut on a turn boundary: the first turn starting at/after EXCERPT_SEC. Cutting mid-turn
        # would leave reference text that the audio never reaches and inflate every system's ER.
        full = out_dir / f"{source['youtube']}.full.m4a"
        if not full.exists():
            subprocess.run(
                [sys.executable, "-m", "yt_dlp", "-q", "--no-playlist", "-f", "ba[ext=m4a]/ba",
                 "-o", str(full), f"https://www.youtube.com/watch?v={source['youtube']}"],
                check=True,
            )
        if source.get("full"):
            cut = _duration(full)
        else:
            cut = next((t["start"] for t in turns if t["start"] >= EXCERPT_SEC), turns[-1]["start"])
        kept = [turn for turn in turns if turn["start"] < cut]
        for turn, following in zip(kept, [t["start"] for t in turns[1 : len(kept)]] + [cut]):
            turn["end"] = float(following)
            turn["start"] = float(turn["start"])
        wav = out_dir / f"{source['id']}.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(full), "-t", str(cut),
             "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
            check=True,
        )
        names = sorted({turn["speaker"] for turn in kept})
        truth = {
            "id": source["id"],
            "kind": "real",
            "language": source["language"],
            "duration": float(cut),
            "timing": "turn_start",
            "source": {"transcript": source["transcript"], "youtube": source["youtube"], "note": source["note"]},
            "speakers": {name: name for name in names},
            "segments": [{"start": t["start"], "end": t["end"], "speaker": t["speaker"], "text": t["text"]} for t in kept],
        }
        (out_dir / f"{source['id']}.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{source['id']}: {len(kept)} turns, speakers={names}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser())
