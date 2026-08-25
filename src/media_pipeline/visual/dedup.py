from __future__ import annotations

from pathlib import Path

from media_pipeline.visual.hashing import combined_similarity, dhash64, hash_similarity, histogram16, histogram_similarity
from media_pipeline.visual.models import CandidateFrame, DedupInfo, Keyframe
from media_pipeline.visual.timestamps import SOURCE_MANUAL, SOURCE_OCR, keyframe_path_name

KEEP = "keep"
DROP = "drop"


def apply_dedup(
    candidates: list[CandidateFrame],
    artifact_root: Path,
    *,
    similarity_threshold: float,
    overrides: dict[str, str],
    ocr_keep: set[str] | None = None,
    lookback_sec: float = 30.0,
) -> list[tuple[CandidateFrame, DedupInfo]]:
    """Drop near-duplicates, biased toward keeping borderline frames.

    Manual overrides are never replaced by the automatic decision.
    """
    ocr_keep = ocr_keep or set()
    kept_index: list[tuple[CandidateFrame, int, list[int]]] = []
    results: list[tuple[CandidateFrame, DedupInfo]] = []
    for candidate in candidates:
        image = artifact_root / candidate.path
        name = Path(candidate.path).name
        override = overrides.get(name) or overrides.get(candidate.path)
        if not image.exists():
            info = DedupInfo(kept=False, decision="auto", reason="missing_file")
            if override == KEEP:
                info = DedupInfo(kept=True, decision=SOURCE_MANUAL, reason="override_keep_missing")
            results.append((candidate, info))
            continue
        digest = dhash64(image)
        hist = histogram16(image)
        nearest = ""
        best = 0.0
        for previous, prev_hash, prev_hist in reversed(kept_index):
            if candidate.timestamp - previous.timestamp > lookback_sec:
                break
            sim = combined_similarity(
                hash_similarity(digest, prev_hash),
                histogram_similarity(hist, prev_hist),
            )
            if sim > best:
                best = sim
                nearest = Path(previous.path).name
        auto_drop = best >= similarity_threshold and nearest != ""
        if override == KEEP:
            info = DedupInfo(kept=True, nearest_frame=nearest, similarity=best, decision=SOURCE_MANUAL, reason="override_keep")
        elif override == DROP:
            info = DedupInfo(kept=False, nearest_frame=nearest, similarity=best, decision=SOURCE_MANUAL, reason="override_drop")
        elif name in ocr_keep and auto_drop:
            info = DedupInfo(kept=True, nearest_frame=nearest, similarity=best, decision="auto", reason="ocr_change")
            if SOURCE_OCR not in candidate.sources:
                candidate.sources.append(SOURCE_OCR)
        elif auto_drop:
            info = DedupInfo(kept=False, nearest_frame=nearest, similarity=best, decision="auto", reason="similar")
        else:
            info = DedupInfo(kept=True, nearest_frame=nearest, similarity=best, decision="auto", reason="distinct")
        results.append((candidate, info))
        if info.kept:
            kept_index.append((candidate, digest, hist))
    return results


def keyframes_from_dedup(
    results: list[tuple[CandidateFrame, DedupInfo]],
    artifact_root: Path,
) -> list[Keyframe]:
    frames: list[Keyframe] = []
    key_dir = artifact_root / "keyframes"
    key_dir.mkdir(parents=True, exist_ok=True)
    for candidate, info in results:
        if not info.kept:
            continue
        source = artifact_root / candidate.path
        dest_name = keyframe_path_name(candidate.timestamp)
        dest = artifact_root / dest_name
        if source.exists():
            dest.write_bytes(source.read_bytes())
        frames.append(
            Keyframe(
                timestamp=candidate.timestamp,
                image_path=dest_name,
                sources=list(candidate.sources),
                candidate_sources=list(candidate.sources),
                dedup=info,
            )
        )
    return frames
