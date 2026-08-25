from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 16000
FRAME_MS = 20
PAD_SEC = 0.25
MERGE_GAP_SEC = 0.4
PREFERRED_MIN_SEC = 15.0
PREFERRED_MAX_SEC = 30.0
HARD_MAX_SEC = 45.0
MIN_SPEECH_SEC = 0.12
MIN_REGION_SEC = 0.08


@dataclass(frozen=True)
class SpeechRegion:
    """Contiguous speech on the source-audio timeline, before ASR packing."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class AsrChunk:
    """One ASR inference window.

    This is not a final presentation segment. Later stages may split or merge
    transcript text using speaker turns, pauses, or scene boundaries while
    keeping these source timestamps as the coarse alignment grid.
    """

    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def build_asr_chunks(
    audio_path: Path,
    *,
    duration: float | None = None,
    pad_sec: float = PAD_SEC,
    merge_gap_sec: float = MERGE_GAP_SEC,
    preferred_min_sec: float = PREFERRED_MIN_SEC,
    preferred_max_sec: float = PREFERRED_MAX_SEC,
    hard_max_sec: float = HARD_MAX_SEC,
) -> list[AsrChunk]:
    """VAD → merge/split → padded ASR windows with absolute timestamps."""
    audio_duration = duration if duration is not None else wav_duration(audio_path)
    regions = detect_speech_regions(audio_path)
    return pack_asr_chunks(
        regions,
        audio_duration,
        pad_sec=pad_sec,
        merge_gap_sec=merge_gap_sec,
        preferred_min_sec=preferred_min_sec,
        preferred_max_sec=preferred_max_sec,
        hard_max_sec=hard_max_sec,
    )


def detect_speech_regions(audio_path: Path, frame_ms: int = FRAME_MS) -> list[SpeechRegion]:
    energies, _sample_rate = frame_energies(audio_path, frame_ms=frame_ms)
    if not energies:
        return []
    frame_sec = frame_ms / 1000.0
    mask = speech_mask(energies)
    return regions_from_mask(mask, frame_sec, min_speech_sec=MIN_SPEECH_SEC)


def pack_asr_chunks(
    regions: list[SpeechRegion],
    duration: float,
    *,
    pad_sec: float = PAD_SEC,
    merge_gap_sec: float = MERGE_GAP_SEC,
    preferred_min_sec: float = PREFERRED_MIN_SEC,
    preferred_max_sec: float = PREFERRED_MAX_SEC,
    hard_max_sec: float = HARD_MAX_SEC,
) -> list[AsrChunk]:
    merged = merge_regions(regions, merge_gap_sec)
    bounded = split_long_regions(merged, preferred_max_sec=preferred_max_sec, hard_max_sec=hard_max_sec)
    packed = pack_regions(
        bounded,
        preferred_min_sec=preferred_min_sec,
        preferred_max_sec=preferred_max_sec,
        hard_max_sec=hard_max_sec,
    )
    chunks: list[AsrChunk] = []
    for region in packed:
        start = max(0.0, region.start - pad_sec)
        end = min(duration, region.end + pad_sec) if duration > 0 else region.end + pad_sec
        if end - start < MIN_REGION_SEC:
            continue
        chunks.append(AsrChunk(start=start, end=end))
    return chunks


def merge_regions(regions: list[SpeechRegion], merge_gap_sec: float) -> list[SpeechRegion]:
    ordered = [item for item in sorted(regions, key=lambda item: item.start) if item.end > item.start]
    if not ordered:
        return []
    merged = [ordered[0]]
    for region in ordered[1:]:
        previous = merged[-1]
        if region.start - previous.end <= merge_gap_sec:
            merged[-1] = SpeechRegion(previous.start, max(previous.end, region.end))
        else:
            merged.append(region)
    return merged


def split_long_regions(
    regions: list[SpeechRegion],
    *,
    preferred_max_sec: float,
    hard_max_sec: float,
) -> list[SpeechRegion]:
    split: list[SpeechRegion] = []
    for region in regions:
        if region.duration <= hard_max_sec:
            split.append(region)
            continue
        cursor = region.start
        while cursor < region.end:
            remaining = region.end - cursor
            length = remaining if remaining <= hard_max_sec else preferred_max_sec
            nxt = min(region.end, cursor + length)
            split.append(SpeechRegion(cursor, nxt))
            cursor = nxt
    return split


def pack_regions(
    regions: list[SpeechRegion],
    *,
    preferred_min_sec: float,
    preferred_max_sec: float,
    hard_max_sec: float,
) -> list[SpeechRegion]:
    packed: list[SpeechRegion] = []
    current: SpeechRegion | None = None
    for region in regions:
        if current is None:
            current = region
            continue
        combined = SpeechRegion(current.start, region.end)
        if combined.duration <= preferred_max_sec:
            current = combined
            continue
        if current.duration < preferred_min_sec and combined.duration <= hard_max_sec:
            current = combined
            continue
        packed.append(current)
        current = region
    if current is not None:
        packed.append(current)
    return packed


def speech_mask(energies: list[float]) -> list[bool]:
    if not energies:
        return []
    ranked = sorted(energies)
    noise = ranked[max(0, int(len(ranked) * 0.2) - 1)]
    threshold = max(noise * 4.0, 0.012)
    return [energy >= threshold for energy in energies]


def regions_from_mask(
    mask: list[bool],
    frame_sec: float,
    *,
    min_speech_sec: float = MIN_SPEECH_SEC,
) -> list[SpeechRegion]:
    regions: list[SpeechRegion] = []
    start: int | None = None
    for index, spoken in enumerate(mask):
        if spoken and start is None:
            start = index
        elif not spoken and start is not None:
            regions.append(SpeechRegion(start * frame_sec, index * frame_sec))
            start = None
    if start is not None:
        regions.append(SpeechRegion(start * frame_sec, len(mask) * frame_sec))
    return [region for region in regions if region.duration >= min_speech_sec]


def frame_energies(audio_path: Path, frame_ms: int = FRAME_MS) -> tuple[list[float], int]:
    energies: list[float] = []
    with wave.open(str(audio_path), "rb") as handle:
        sample_rate = handle.getframerate() or SAMPLE_RATE
        channels = max(1, handle.getnchannels())
        width = handle.getsampwidth()
        frame_samples = max(1, int(sample_rate * frame_ms / 1000))
        while True:
            raw = handle.readframes(frame_samples)
            if not raw:
                break
            samples = _pcm_to_mono(raw, width=width, channels=channels)
            if not samples:
                continue
            mean_square = sum(sample * sample for sample in samples) / len(samples)
            energies.append(math.sqrt(mean_square))
    return energies, sample_rate


def read_wav_slice(audio_path: Path, start: float, end: float) -> tuple[list[float], int]:
    """Read one ASR window without loading the rest of the file."""
    start = max(0.0, start)
    end = max(start, end)
    with wave.open(str(audio_path), "rb") as handle:
        sample_rate = handle.getframerate() or SAMPLE_RATE
        channels = max(1, handle.getnchannels())
        width = handle.getsampwidth()
        total = handle.getnframes()
        start_frame = min(total, int(start * sample_rate))
        end_frame = min(total, int(math.ceil(end * sample_rate)))
        handle.setpos(start_frame)
        raw = handle.readframes(max(0, end_frame - start_frame))
    samples = _pcm_to_mono(raw, width=width, channels=channels)
    return samples, sample_rate


def wav_duration(audio_path: Path) -> float:
    with wave.open(str(audio_path), "rb") as handle:
        sample_rate = handle.getframerate() or SAMPLE_RATE
        frames = handle.getnframes()
    if sample_rate <= 0:
        return 0.0
    return frames / sample_rate


def _pcm_to_mono(raw: bytes, *, width: int, channels: int) -> list[float]:
    if not raw:
        return []
    if width == 2:
        import array

        pcm = array.array("h")
        pcm.frombytes(raw[: len(raw) - (len(raw) % 2)])
        values = [sample / 32768.0 for sample in pcm]
    elif width == 1:
        values = [(sample - 128) / 128.0 for sample in raw]
    else:
        return []
    if channels <= 1:
        return values
    mono: list[float] = []
    for index in range(0, len(values) - channels + 1, channels):
        mono.append(sum(values[index : index + channels]) / channels)
    return mono
