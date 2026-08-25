#!/usr/bin/env python3
"""Generate simple PNG icons for the Chrome extension."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path


def png(width: int, height: int, rgba: list[tuple[int, int, int, int]]) -> bytes:
    raw = b""
    for row in range(height):
        raw += b"\x00"
        for col in range(width):
            raw += bytes(rgba[row * width + col])

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    idat = chunk(b"IDAT", zlib.compress(raw, 9))
    iend = chunk(b"IEND", b"")
    return header + ihdr + idat + iend


def draw(size: int) -> list[tuple[int, int, int, int]]:
    pixels = []
    for y in range(size):
        for x in range(size):
            nx = (x + 0.5) / size
            ny = (y + 0.5) / size
            inset = 0.08
            in_square = inset <= nx <= 1 - inset and inset <= ny <= 1 - inset
            if not in_square:
                pixels.append((0, 0, 0, 0))
                continue
            color = (79, 70, 229, 255)
            cx, cy = 0.50, 0.46
            dx, dy = nx - cx, ny - cy
            star = abs(dx) * 3.4 + abs(dy) * 3.4
            if star < 0.55 or (abs(dx) < 0.07 and abs(dy) < 0.28) or (abs(dy) < 0.07 and abs(dx) < 0.28):
                color = (255, 255, 255, 255)
            pixels.append(color)
    return pixels


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "extension" / "icons"
    root.mkdir(parents=True, exist_ok=True)
    for size in (16, 48, 128):
        (root / f"icon{size}.png").write_bytes(png(size, size, draw(size)))


if __name__ == "__main__":
    main()
