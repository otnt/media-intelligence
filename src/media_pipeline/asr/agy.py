"""ASR + speaker tags through the Antigravity CLI (``agy``) and the user's own Gemini login.

No API key: ``agy`` is already signed in to Google. Each audio window is copied
alone into a temporary workspace and ``agy -p`` is asked to ``read_file`` it and
return JSON segments. The agent runs sandboxed, cannot approve shell commands,
and cannot read outside that workspace; any tool other than ``read_file`` on
the window fails the run.

Long audio is cut into overlapping windows. Gemini labels speakers per window,
so the overlap (transcribed twice) votes on which local label is which global
speaker. Gemini's timestamps drift by a second or two, so segment edges are
finally snapped to local VAD speech boundaries.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from media_pipeline.asr.base import ASRNotAvailableError, ASRProvider
from media_pipeline.models import ASROptions, Transcript, TranscriptSegment

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.8-flash-medium"
SEGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number", "description": "seconds from the start of this file"},
                    "end": {"type": "number", "description": "seconds from the start of this file"},
                    "speaker": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["start", "end", "speaker", "text"],
            },
        }
    },
    "required": ["segments"],
}
# agy hands structured (--json-schema) answers back through this tool; it touches nothing.
_ANSWER_TOOLS = {"finish"}
_LANGUAGE_NAMES = {"zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean", "yue": "Cantonese"}


class AgyError(RuntimeError):
    pass


@dataclass
class Piece:
    start: float
    end: float
    text: str
    speaker: str


@dataclass
class AgySettings:
    binary: str = "agy"
    model: str = DEFAULT_MODEL
    window_sec: float = 180.0
    overlap_sec: float = 45.0
    concurrency: int = 3
    timeout_sec: int = 900


def agy_settings(config: Any) -> AgySettings:
    raw = getattr(config, "agy", None)
    if raw is None:
        return AgySettings()
    return AgySettings(
        binary=str(getattr(raw, "binary", "") or "agy"),
        model=str(getattr(raw, "asr_model", "") or DEFAULT_MODEL),
        window_sec=float(getattr(raw, "window_sec", 0) or 180),
        overlap_sec=float(getattr(raw, "overlap_sec", 0) or 45),
        concurrency=max(1, int(getattr(raw, "concurrency", 0) or 3)),
        timeout_sec=int(getattr(raw, "timeout_sec", 0) or 900),
    )


def run_agy(
    prompt: str,
    *,
    workspace: Path,
    allowed_reads: set[Path],
    settings: AgySettings,
    schema: dict[str, Any] | None = None,
    model: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> str:
    """Run one sandboxed ``agy -p`` turn and return the final response text."""
    binary = shutil.which(settings.binary) or settings.binary
    command = [
        binary,
        "-p",
        prompt,
        "--model",
        model or settings.model,
        "--mode",
        "accept-edits",
        "--sandbox",
        "--disable-slash-commands",
        "--settings",
        json.dumps({"allowNonWorkspaceAccess": False}),
        "--output-format",
        "stream-json",
        "--print-timeout",
        f"{settings.timeout_sec}s",
    ]
    if schema is not None:
        schema_path = workspace / ".schema.json"
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        command += ["--json-schema", str(schema_path)]
    try:
        completed = runner(
            command,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=settings.timeout_sec + 60,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ASRNotAvailableError(f"{settings.binary} is not installed") from exc
    except subprocess.TimeoutExpired as exc:
        raise AgyError(f"agy did not finish within {settings.timeout_sec}s") from exc
    allowed = {str(path.resolve()) for path in allowed_reads}
    result: dict[str, Any] | None = None
    for line in completed.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        step = event.get("step_update") or {}
        if step.get("step_type") == "tool" and step.get("state") == "ACTIVE":
            info = step.get("tool_info") or {}
            name = str(info.get("name") or "")
            target = str((info.get("parameters") or {}).get("Path") or "")
            if name in _ANSWER_TOOLS:
                continue
            if name != "read_file" or (target and str(Path(target).resolve()) not in allowed):
                raise AgyError(f"agy tried an unexpected tool call: {name} {target}".strip())
        if event.get("event") == "result":
            result = event.get("result") or {}
    if result is None:
        tail = (completed.stderr or completed.stdout or "").strip()[-400:]
        raise AgyError(f"agy exited {completed.returncode} without a result: {tail}")
    if str(result.get("status") or "").upper() != "SUCCESS":
        raise AgyError(f"agy {result.get('status')}: {result.get('error') or result.get('finished_reason')}")
    return str(result.get("response") or "")


class AgyASRProvider(ASRProvider):
    runtime = "Antigravity CLI (Gemini)"
    display_name = "Gemini via agy"
    local = False

    def __init__(self, model: str = "agy-gemini", config: Any = None, runner: Callable[..., Any] | None = None) -> None:
        self.id = model
        self.config = config
        self._runner = runner or subprocess.run

    def bind_config(self, config: Any) -> None:
        self.config = config

    def transcribe(self, audio_path: Path, options: ASROptions | None = None) -> Transcript:
        options = options or ASROptions()
        settings = agy_settings(self.config)
        if not shutil.which(settings.binary) and self._runner is subprocess.run:
            raise ASRNotAvailableError(f"{settings.binary} is not installed or not on PATH")
        duration = _probe_duration(audio_path) or 0.0
        windows = plan_windows(duration, settings.window_sec, settings.overlap_sec)
        logger.info("agy transcribing %.0fs of audio in %s window(s) with %s", duration, len(windows), settings.model)

        def work(item: tuple[int, tuple[float, float]]) -> list[Piece]:
            index, (start, end) = item
            return self._transcribe_window(audio_path, index, start, end, len(windows) > 1, options, settings)

        with ThreadPoolExecutor(max_workers=min(settings.concurrency, len(windows))) as pool:
            results = list(pool.map(work, enumerate(windows)))
        pieces = stitch_windows([(start, end, found) for (start, end), found in zip(windows, results)])
        pieces = snap_to_speech(pieces, audio_path)
        segments = [
            TranscriptSegment(start=p.start, end=p.end, text=p.text, speaker=p.speaker)
            for p in pieces
            if p.text.strip()
        ]
        logger.info("agy returned %s segments, %s speakers", len(segments), len({s.speaker for s in segments}))
        return Transcript(language=options.language, segments=segments, provider="agy", model=settings.model)

    def _transcribe_window(
        self,
        audio_path: Path,
        index: int,
        start: float,
        end: float,
        trim: bool,
        options: ASROptions,
        settings: AgySettings,
    ) -> list[Piece]:
        with tempfile.TemporaryDirectory(prefix="agy-asr-") as tmp:
            workspace = Path(tmp)
            clip = workspace / "audio.wav"
            _cut_wav(audio_path, clip, start, end, trim)
            prompt = build_prompt(clip, options)
            last_error: Exception | None = None
            for attempt in range(2):
                try:
                    text = run_agy(
                        prompt,
                        workspace=workspace,
                        allowed_reads={clip},
                        settings=settings,
                        schema=SEGMENT_SCHEMA,
                        runner=self._runner,
                    )
                    pieces = parse_segments(text)
                    break
                except (AgyError, ValueError) as exc:
                    last_error = exc
                    logger.warning("agy window %s attempt %s failed: %s", index + 1, attempt + 1, exc)
            else:
                raise AgyError(f"agy window {index + 1} failed: {last_error}")
        length = end - start if end > start else float("inf")
        return [
            Piece(
                start=start + max(0.0, min(p.start, length)),
                end=start + max(0.0, min(max(p.end, p.start), length)),
                text=p.text,
                speaker=f"w{index}:{p.speaker}",
            )
            for p in pieces
        ]


def build_prompt(clip: Path, options: ASROptions) -> str:
    language = ""
    if options.language:
        language = f" The audio is mostly {_LANGUAGE_NAMES.get(options.language, options.language)}."
    context = ""
    if options.context:
        context = (
            "\nContext about the recording (for spelling names and terms only; never copy it into the transcript):\n"
            f"{options.context.strip()[:1500]}\n"
        )
    return (
        f"Call read_file on {clip} exactly once, then transcribe the audio you hear. "
        "Do not run any commands and do not read any other file. "
        "Transcribe verbatim, keeping each utterance in the language it was spoken "
        "(Chinese stays Chinese in simplified characters, English stays English, code-switching stays mixed). "
        "Start a new segment at every change of speaker and at sentence ends; keep segments under about 20 seconds. "
        "Label voices SPEAKER_00, SPEAKER_01, ... in order of first appearance; the same voice keeps the same label. "
        "start and end are seconds from the beginning of this file. Skip music and silence; "
        f"never invent speech you cannot hear.{language}{context}"
    )


def parse_segments(text: str) -> list[Piece]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("{") :]
    try:
        data = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"agy did not return JSON: {text[:200]}") from exc
    pieces: list[Piece] = []
    for item in data.get("segments") or []:
        try:
            start, end = float(item.get("start")), float(item.get("end"))
        except (TypeError, ValueError):
            continue
        content = " ".join(str(item.get("text") or "").split())
        if not content:
            continue
        pieces.append(Piece(start=start, end=max(end, start), text=content, speaker=str(item.get("speaker") or "?")))
    pieces.sort(key=lambda p: (p.start, p.end))
    return pieces


# ------------------------------------------------------------------ windows & stitching


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


def stitch_windows(chunks: list[tuple[float, float, list[Piece]]]) -> list[Piece]:
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
    previous = relabel(first, {})
    stitched: list[Piece] = []
    for start, end, pieces in chunks[1:]:
        lo, hi = start, prev_end
        cut = (lo + hi) / 2
        mapping = _vote(
            [p for p in previous if p.end > lo and p.start < hi],
            [p for p in pieces if p.end > lo and p.start < hi],
        )
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


def snap_to_speech(pieces: list[Piece], audio_path: Path, reach: float = 2.0) -> list[Piece]:
    """Move segment edges onto nearby VAD speech boundaries; Gemini timestamps drift a little."""
    try:
        from media_pipeline.vad import detect_speech_regions

        regions = detect_speech_regions(audio_path)
    except Exception as exc:
        logger.info("Skipping VAD snap: %s", exc)
        return pieces
    if not regions:
        return pieces
    starts = [r.start for r in regions]
    ends = [r.end for r in regions]

    def nearest(value: float, candidates: list[float], limit: float = reach) -> float:
        best = min(candidates, key=lambda c: abs(c - value))
        return best if abs(best - value) <= limit else value

    snapped: list[Piece] = []
    floor = 0.0
    previous_speaker = None
    for piece in pieces:
        # Never start before the previous segment ended: pick the first speech onset at/after it.
        # A change of speaker nearly always sits on a pause, so look further for one.
        limit = reach * 2 if previous_speaker not in (None, piece.speaker) else reach
        previous_speaker = piece.speaker
        later = [c for c in starts if c >= floor - 0.05]
        start = nearest(max(piece.start, floor), later, limit) if later else max(piece.start, floor)
        start = max(start, floor)
        end = nearest(piece.end, [c for c in ends if c > start] or [piece.end])
        if end <= start:
            end = max(piece.end, start + 0.3)
        snapped.append(Piece(round(start, 2), round(end, 2), piece.text, piece.speaker))
        floor = end
    return snapped


# ------------------------------------------------------------------------ audio io


def _probe_duration(path: Path) -> float | None:
    from media_pipeline.media import probe_duration

    try:
        return probe_duration(path)
    except Exception:
        return None


def _cut_wav(source: Path, target: Path, start: float, end: float, trim: bool) -> None:
    if not trim:
        shutil.copyfile(source, target)
        return
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise AgyError("ffmpeg is required to split long audio for agy")
    command = [
        ffmpeg, "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(source),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0 or not target.exists():
        raise AgyError(f"ffmpeg could not cut audio: {completed.stderr.strip()[-300:]}")
