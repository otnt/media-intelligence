from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class OcrEngine:
    name = "none"

    def text_for(self, image_path: Path) -> str:
        return ""


class TesseractOcr(OcrEngine):
    name = "tesseract"

    def text_for(self, image_path: Path) -> str:
        binary = shutil.which("tesseract")
        if not binary or not image_path.exists():
            return ""
        completed = subprocess.run(
            [binary, str(image_path), "stdout", "--psm", "6"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return ""
        return " ".join((completed.stdout or "").split())


def build_ocr_engine() -> OcrEngine:
    if shutil.which("tesseract"):
        return TesseractOcr()
    return OcrEngine()


def text_change_ratio(previous: str, current: str) -> float:
    prev_tokens = set(_tokens(previous))
    curr_tokens = set(_tokens(current))
    if not prev_tokens and not curr_tokens:
        return 0.0
    if not prev_tokens or not curr_tokens:
        return 1.0 if (prev_tokens or curr_tokens) else 0.0
    union = prev_tokens | curr_tokens
    if not union:
        return 0.0
    return 1.0 - (len(prev_tokens & curr_tokens) / len(union))


def ocr_keep_names(
    candidates: list,
    artifact_root: Path,
    engine: OcrEngine,
    *,
    threshold: float,
) -> set[str]:
    """Return candidate filenames whose on-screen text changed enough to keep."""
    if engine.name == "none" or threshold <= 0:
        return set()
    keep: set[str] = set()
    previous = ""
    for candidate in candidates:
        path = artifact_root / candidate.path
        current = engine.text_for(path)
        if previous and current and text_change_ratio(previous, current) >= threshold:
            keep.add(Path(candidate.path).name)
        if current:
            previous = current
    return keep


def _tokens(text: str) -> list[str]:
    return [part.lower() for part in text.replace("\n", " ").split() if part]
