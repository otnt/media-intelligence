"""Score benchmark hypotheses against ground truth.

    uv run --with jiwer --with scipy python bench/score.py ~/AIContent/bench

Metrics (per system x sample, averaged over runs):

* ER     token error rate. Tokens are CJK characters and Latin words, so this is
         CER on Chinese, WER on English and MER on code-switched audio.
         Lower-cased, punctuation removed, before comparison.
* cpER   speaker-attributed ER: concatenate each speaker's words, map
         hypothesis speakers to reference speakers with the Hungarian algorithm
         (minimising errors), sum the errors. Wrong speaker = wrong words.
* spk    hypothesis speaker count / reference speaker count.
* attrib share of hypothesis speech time whose speaker is wrong under the best
         time-overlap mapping (works for exact and turn-start timing).
* DER    diarization error rate with a 0.25 s collar (synthetic only: needs
         exact reference segment ends).
* chg    speaker-change points: median |error| in seconds and share of
         reference changes matched within 1 s.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import jiwer
import numpy as np
from scipy.optimize import linear_sum_assignment

_TOKEN = re.compile(r"[㐀-鿿豈-﫿]|[a-z0-9]+(?:'[a-z]+)?")
FRAME = 0.01


def tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    return _TOKEN.findall(text)


def error_count(ref: list[str], hyp: list[str]) -> int:
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    out = jiwer.process_words(" ".join(ref), " ".join(hyp))
    return out.substitutions + out.deletions + out.insertions


def er(ref_segments: list[dict], hyp_segments: list[dict]) -> float:
    ref = [t for s in ref_segments for t in tokens(s["text"])]
    hyp = [t for s in hyp_segments for t in tokens(s["text"])]
    return error_count(ref, hyp) / max(1, len(ref))


def cp_er(ref_segments: list[dict], hyp_segments: list[dict]) -> float | None:
    if any(s.get("speaker") in (None, "") for s in hyp_segments):
        return None
    ref_by: dict[str, list[str]] = defaultdict(list)
    hyp_by: dict[str, list[str]] = defaultdict(list)
    for s in ref_segments:
        ref_by[s["speaker"]].extend(tokens(s["text"]))
    for s in hyp_segments:
        hyp_by[s["speaker"]].extend(tokens(s["text"]))
    refs, hyps = list(ref_by), list(hyp_by)
    size = max(len(refs), len(hyps))
    cost = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            r = ref_by[refs[i]] if i < len(refs) else []
            h = hyp_by[hyps[j]] if j < len(hyps) else []
            cost[i, j] = error_count(r, h)
    rows, cols = linear_sum_assignment(cost)
    total_ref = sum(len(v) for v in ref_by.values())
    return float(cost[rows, cols].sum()) / max(1, total_ref)


def _labels(segments: list[dict], duration: float) -> list[str | None]:
    frames: list[str | None] = [None] * int(duration / FRAME + 1)
    for s in segments:
        for k in range(int(s["start"] / FRAME), min(len(frames), int(s["end"] / FRAME))):
            frames[k] = s.get("speaker") or "?"
    return frames


def _mapping(ref: list, hyp: list) -> dict[str, str]:
    ref_ids = sorted({r for r in ref if r})
    hyp_ids = sorted({h for h in hyp if h})
    if not ref_ids or not hyp_ids:
        return {}
    overlap = np.zeros((len(hyp_ids), len(ref_ids)))
    hi = {h: i for i, h in enumerate(hyp_ids)}
    ri = {r: i for i, r in enumerate(ref_ids)}
    for r, h in zip(ref, hyp):
        if r and h:
            overlap[hi[h], ri[r]] += 1
    rows, cols = linear_sum_assignment(-overlap)
    return {hyp_ids[a]: ref_ids[b] for a, b in zip(rows, cols)}


def attribution_error(ref_segments, hyp_segments, duration) -> float | None:
    if any(s.get("speaker") in (None, "") for s in hyp_segments):
        return None
    ref = _labels(ref_segments, duration)
    hyp = _labels(hyp_segments, duration)
    mapping = _mapping(ref, hyp)
    both = [(r, h) for r, h in zip(ref, hyp) if r and h]
    if not both:
        return None
    wrong = sum(1 for r, h in both if mapping.get(h) != r)
    return wrong / len(both)


def der(ref_segments, hyp_segments, duration, collar: float = 0.25) -> float | None:
    if any(s.get("speaker") in (None, "") for s in hyp_segments):
        return None
    ref = _labels(ref_segments, duration)
    hyp = _labels(hyp_segments, duration)
    scored = [True] * len(ref)
    for s in ref_segments:
        for edge in (s["start"], s["end"]):
            for k in range(max(0, int((edge - collar) / FRAME)), min(len(ref), int((edge + collar) / FRAME))):
                scored[k] = False
    mapping = _mapping(ref, hyp)
    speech = miss = fa = conf = 0
    for r, h, keep in zip(ref, hyp, scored):
        if not keep:
            continue
        if r:
            speech += 1
            if not h:
                miss += 1
            elif mapping.get(h) != r:
                conf += 1
        elif h:
            fa += 1
    return (miss + fa + conf) / max(1, speech)


def change_points(segments: list[dict], mapping: dict[str, str] | None = None) -> list[float]:
    points = []
    previous = None
    for s in sorted(segments, key=lambda x: x["start"]):
        label = (mapping or {}).get(s.get("speaker"), s.get("speaker"))
        if previous is not None and label != previous:
            points.append(s["start"])
        previous = label
    return points


def change_error(ref_segments, hyp_segments, duration) -> tuple[float | None, float | None]:
    if any(s.get("speaker") in (None, "") for s in hyp_segments):
        return None, None
    mapping = _mapping(_labels(ref_segments, duration), _labels(hyp_segments, duration))
    ref_pts = change_points(ref_segments)
    hyp_pts = change_points(hyp_segments, mapping)
    if not ref_pts or not hyp_pts:
        return None, 0.0
    errors = [min(abs(p - q) for q in hyp_pts) for p in ref_pts]
    return statistics.median(errors), sum(e <= 1.0 for e in errors) / len(errors)


def seam_er(ref_segments: list[dict], hyp_segments: list[dict], windows: list, reach: float = 120.0) -> float | None:
    """ER restricted to reference turns within ±reach seconds of a window seam."""
    seams = [w[0] for w in windows[1:]]
    errors = total = 0
    for seam in seams:
        refs = [s for s in ref_segments if s["end"] > seam - reach and s["start"] < seam + reach]
        if not refs:
            continue
        lo, hi = refs[0]["start"], refs[-1]["end"]
        hyps = [h for h in hyp_segments if lo <= (h["start"] + h["end"]) / 2 < hi]
        ref_tokens = [t for s in refs for t in tokens(s["text"])]
        errors += error_count(ref_tokens, [t for h in hyps for t in tokens(h["text"])])
        total += len(ref_tokens)
    return errors / total if total else None


def score(bench: Path) -> list[dict]:
    rows = []
    for truth_path in sorted((bench / "samples").glob("*.json")):
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        ref = truth["segments"]
        for system_dir in sorted((bench / "results").glob("*")):
            runs = sorted(system_dir.glob(f"{truth['id']}.run*.json"))
            for run_path in runs:
                hyp_doc = json.loads(run_path.read_text(encoding="utf-8"))
                hyp = [s for s in hyp_doc["segments"] if s.get("text", "").strip()]
                med, within = change_error(ref, hyp, truth["duration"])
                rows.append({
                    "sample": truth["id"],
                    "system": system_dir.name,
                    "run": hyp_doc.get("run"),
                    "elapsed": hyp_doc.get("elapsed_sec"),
                    "er": er(ref, hyp),
                    "cp_er": cp_er(ref, hyp),
                    "spk_hyp": len({s.get("speaker") for s in hyp if s.get("speaker")}) or None,
                    "spk_ref": len({s["speaker"] for s in ref}),
                    "attrib": attribution_error(ref, hyp, truth["duration"]),
                    "der": der(ref, hyp, truth["duration"]) if truth["timing"] == "exact" else None,
                    "chg_med": med,
                    "chg_1s": within,
                    "seam_er": seam_er(ref, hyp, hyp_doc["windows"]) if hyp_doc.get("windows") else None,
                })
    return rows


def _fmt(value, pct: bool = True) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%" if pct else f"{value:.2f}"


def report(rows: list[dict]) -> str:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["sample"], row["system"])].append(row)
    lines = ["| sample | system | runs | ER | cpER | spk | attrib err | DER | chg med (s) | chg ≤1s | seam ER | time (s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def mean(items, key):
        values = [r[key] for r in items if r[key] is not None]
        return statistics.mean(values) if values else None

    def spread(items, key):
        values = [r[key] for r in items if r[key] is not None]
        if len(values) < 2:
            return ""
        return f" ±{(max(values) - min(values)) * 50:.1f}"

    for (sample, system), items in sorted(grouped.items()):
        spk = "/".join(str(r["spk_hyp"] or "—") for r in items) + f" (ref {items[0]['spk_ref']})"
        lines.append(
            f"| {sample} | {system} | {len(items)} | {_fmt(mean(items, 'er'))}{spread(items, 'er')} | "
            f"{_fmt(mean(items, 'cp_er'))}{spread(items, 'cp_er')} | {spk} | {_fmt(mean(items, 'attrib'))} | "
            f"{_fmt(mean(items, 'der'))} | {_fmt(mean(items, 'chg_med'), pct=False)} | {_fmt(mean(items, 'chg_1s'))} | "
            f"{_fmt(mean(items, 'seam_er'))} | {_fmt(mean(items, 'elapsed'), pct=False)} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    bench_dir = Path(sys.argv[1]).expanduser()
    scored = score(bench_dir)
    (bench_dir / "scores.json").write_text(json.dumps(scored, ensure_ascii=False, indent=1), encoding="utf-8")
    print(report(scored))
