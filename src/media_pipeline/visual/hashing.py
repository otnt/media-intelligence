from __future__ import annotations

import math
from pathlib import Path

from PIL import Image


def dhash64(path: Path) -> int:
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(gray.tobytes())
    value = 0
    for row in range(8):
        base = row * 9
        for col in range(8):
            if pixels[base + col] > pixels[base + col + 1]:
                value |= 1 << (row * 8 + col)
    return value


def histogram16(path: Path) -> list[int]:
    with Image.open(path) as image:
        gray = image.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
        pixels = list(gray.tobytes())
    bins = [0] * 16
    for pixel in pixels:
        bins[min(15, pixel >> 4)] += 1
    return bins


def hash_similarity(left: int, right: int) -> float:
    return 1.0 - (left ^ right).bit_count() / 64.0


def histogram_similarity(left: list[int], right: list[int]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    norm_left = math.sqrt(sum(value * value for value in left))
    norm_right = math.sqrt(sum(value * value for value in right))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_left * norm_right)))


def combined_similarity(hash_sim: float, hist_sim: float) -> float:
    return max(hash_sim, hist_sim)
