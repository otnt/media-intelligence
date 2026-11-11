"""Cut long audio into overlapping windows and stitch per-window transcripts back together.

Shared by the agy (Gemini) and DashScope (Fun-ASR) providers. Each window labels
speakers on its own; the overlap is transcribed twice and votes on which local
label is which global speaker.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass
class Piece:
    start: float
    end: float
    text: str
    speaker: str


def plan_windows(duration: float, window: float = 600.0, overlap: float = 30.0) -> list[tuple[float, float]]:
    if duration <= window * 1.25:
        return [(0.0, duration)]
    windows: list[tuple[float, float]] = []
    start = 0.0
    while True:
        end = min(duration, start + window)
        if duration - end < window * 0.25:
            end = duration
        windows.append((start, end))
        if end >= duration:
            return windows
        start = end - overlap


def stitch_windows(chunks: list[tuple[float, float, list[Piece]]], shared_labels: bool = False) -> list[Piece]:
    """Join windows and map window-local speaker labels to global SPEAKER_NN ids."""
    if not chunks:
        return []
    global_of: dict[str, str] = {}
    counter = {"n": 0}

    def fresh() -> str:
        label = f"SPEAKER_{counter['n']:02d}"
        counter["n"] += 1
        return label

    def relabel(pieces: list[Piece], mapping: dict[str, str]) -> list[Piece]:
        out = []
        for piece in pieces:
            if piece.speaker not in mapping:
                mapping[piece.speaker] = global_of.setdefault(piece.speaker, fresh())
            out.append(Piece(piece.start, piece.end, piece.text, mapping[piece.speaker]))
        return out

    _, prev_end, first = chunks[0]
    first_mapping: dict[str, str] = {}
    previous = relabel(first, first_mapping)
    # With a shared roster, window N's "SPEAKER_01" is meant to be window 1's "SPEAKER_01".
    by_base = {local.split(":", 1)[-1]: global_id for local, global_id in first_mapping.items()}
    stitched: list[Piece] = []
    for start, end, pieces in chunks[1:]:
        lo, hi = start, prev_end
        cut = (lo + hi) / 2
        mapping = _vote(
            [p for p in previous if p.end > lo and p.start < hi],
            [p for p in pieces if p.end > lo and p.start < hi],
        )
        if shared_labels:
            # The roster label is the stronger signal; overlap votes only place labels the roster doesn't know.
            voted = mapping
            mapping = {}
            for piece in pieces:
                target = by_base.get(piece.speaker.split(":", 1)[-1])
                if target:
                    mapping[piece.speaker] = target
            used = set(mapping.values())
            for local, target in voted.items():
                if local not in mapping and target not in used:
                    mapping[local] = target
                    used.add(target)
        stitched.extend(p for p in previous if p.start < cut)
        previous = relabel([p for p in pieces if p.start >= cut], mapping)
        prev_end = end
    stitched.extend(previous)
    return stitched


def _vote(prev: list[Piece], nxt: list[Piece]) -> dict[str, str]:
    votes: Counter[tuple[str, str]] = Counter()
    for piece in nxt:
        for other in prev:
            shared = min(piece.end, other.end) - max(piece.start, other.start)
            if shared > 0:
                votes[(piece.speaker, other.speaker)] += shared
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for (local, global_id), _weight in votes.most_common():
        if local in mapping or global_id in used:
            continue
        mapping[local] = global_id
        used.add(global_id)
    return mapping
