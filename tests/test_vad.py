import math
import wave
from pathlib import Path

from media_pipeline.vad import (
    AsrChunk,
    SpeechRegion,
    apply_hangover,
    close_small_gaps,
    fill_short_holes,
    merge_regions,
    pack_asr_chunks,
    pack_regions,
    regions_from_mask,
    speech_mask,
    split_long_regions,
    detect_speech_regions,
    read_wav_slice,
)


def test_merges_short_pauses_and_keeps_longer_gaps():
    regions = [
        SpeechRegion(1.0, 2.0),
        SpeechRegion(2.3, 3.0),
        SpeechRegion(5.0, 6.0),
    ]
    merged = merge_regions(regions, merge_gap_sec=0.4)
    assert [(item.start, item.end) for item in merged] == [(1.0, 3.0), (5.0, 6.0)]


def test_splits_continuous_speech_at_hard_max():
    split = split_long_regions(
        [SpeechRegion(0.0, 100.0)],
        preferred_max_sec=30.0,
        hard_max_sec=45.0,
    )
    assert all(item.duration <= 45.0 for item in split)
    assert split[0].start == 0.0
    assert split[-1].end == 100.0
    assert len(split) >= 3


def test_packs_short_regions_toward_preferred_duration():
    regions = [SpeechRegion(float(index * 6), float(index * 6 + 5)) for index in range(8)]
    packed = pack_regions(
        regions,
        preferred_min_sec=15.0,
        preferred_max_sec=30.0,
        hard_max_sec=45.0,
    )
    assert all(item.duration <= 45.0 for item in packed)
    assert any(item.duration >= 15.0 for item in packed)
    assert packed[0].start == 0.0
    assert packed[-1].end == regions[-1].end


def test_chunk_timestamps_include_padding_and_stay_absolute():
    chunks = pack_asr_chunks(
        [SpeechRegion(10.0, 20.0), SpeechRegion(20.2, 28.0)],
        duration=60.0,
        pad_sec=0.25,
        merge_gap_sec=0.4,
    )
    assert len(chunks) == 1
    assert math.isclose(chunks[0].start, 9.75)
    assert math.isclose(chunks[0].end, 28.25)


def test_speech_mask_keeps_moderate_speech_on_speech_heavy_audio():
    energies = [0.006] * 10 + [0.03] * 90
    mask = speech_mask(energies)
    assert sum(mask) >= 80


def test_short_internal_holes_are_filled():
    mask = [True] * 10 + [False] * 10 + [True] * 10
    filled = fill_short_holes(mask, max_hole_frames=15)
    assert all(filled[10:20])
    leftover = fill_short_holes(mask, max_hole_frames=4)
    assert not any(leftover[10:20])


def test_hangover_extends_speech_forward():
    held = apply_hangover([True, False, False, False], hangover_frames=2)
    assert held == [True, True, True, False]


def test_close_small_gaps_covers_boundary_holes():
    closed = close_small_gaps(
        [AsrChunk(0.0, 10.0), AsrChunk(10.3, 20.0), AsrChunk(22.0, 30.0)],
        max_gap_sec=0.5,
    )
    assert math.isclose(closed[0].end, 10.3)
    assert math.isclose(closed[1].start, 10.3)
    assert math.isclose(closed[1].end, 20.0)
    assert math.isclose(closed[2].start, 22.0)


def test_split_prefers_quiet_point_near_preferred_max():
    frame_sec = 0.02
    energies = [0.05] * int(100 / frame_sec)
    dip_at = int(27.2 / frame_sec)
    for index in range(dip_at, dip_at + 12):
        energies[index] = 0.001
    split = split_long_regions(
        [SpeechRegion(0.0, 100.0)],
        preferred_max_sec=30.0,
        hard_max_sec=45.0,
        energies=energies,
        frame_sec=frame_sec,
    )
    assert 26.0 < split[0].end < 29.0
    assert split[-1].end == 100.0


def test_vad_finds_tone_and_slice_does_not_need_full_file(tmp_path: Path):
    path = tmp_path / "speech.wav"
    _write_pcm16_wav(path, _silence(1.0) + _tone(1.5) + _silence(1.0))
    regions = detect_speech_regions(path)
    assert len(regions) == 1
    assert regions[0].start >= 0.7
    assert regions[0].end <= 2.8
    samples, sample_rate = read_wav_slice(path, 1.0, 1.2)
    assert sample_rate == 16000
    assert 16000 * 0.15 < len(samples) < 16000 * 0.3
    energy = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    assert energy > 0.05


def _write_pcm16_wav(path: Path, samples: list[float], sample_rate: int = 16000) -> None:
    import array

    pcm = array.array("h", [max(-32767, min(32767, int(sample * 32767))) for sample in samples])
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _silence(seconds: float, sample_rate: int = 16000) -> list[float]:
    return [0.0] * int(seconds * sample_rate)


def _tone(seconds: float, sample_rate: int = 16000, freq: float = 220.0) -> list[float]:
    total = int(seconds * sample_rate)
    return [0.35 * math.sin(2 * math.pi * freq * index / sample_rate) for index in range(total)]
