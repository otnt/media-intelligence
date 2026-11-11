from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from media_pipeline.asr.base import ASRNotAvailableError, ASRProvider
from media_pipeline.models import ASROptions, Transcript, TranscriptSegment, WordSpan

logger = logging.getLogger(__name__)


class DashScopeASRProvider(ASRProvider):
    """Alibaba Cloud Fun-ASR / Paraformer file transcription.

    Unlike the local providers this one also answers "who spoke when": with
    diarization enabled every sentence carries a ``speaker_id``, which the
    pipeline uses instead of running pyannote.
    """

    runtime = "DashScope cloud"
    display_name = "Fun-ASR"
    local = False

    def __init__(self, model: str = "fun-asr", config: Any = None, client: Any = None) -> None:
        self.id = model
        self.model = model
        self.config = config
        self._client = client

    def bind_config(self, config: Any) -> None:
        self.config = config
        dash = getattr(config, "dashscope", None)
        configured = str(getattr(dash, "asr_model", "") or "").strip()
        if configured and self.model == "fun-asr":
            self.model = configured

    def transcribe(self, audio_path: Path, options: ASROptions | None = None) -> Transcript:
        client = self._ensure_client()
        options = options or ASROptions()
        dash = getattr(self.config, "dashscope", None)
        chunk = float(getattr(dash, "chunk_sec", 0) or 0)
        overlap = float(getattr(dash, "chunk_overlap_sec", 0) or 0)
        duration = _probe_duration(audio_path) or 0.0
        if chunk > 0 and duration > chunk * 1.25:
            transcript = self._transcribe_chunked(client, audio_path, duration, chunk, overlap, options)
        else:
            with tempfile.TemporaryDirectory(prefix="dashscope-asr-") as tmp:
                upload = _compress_for_upload(audio_path, Path(tmp))
                file_url = client.upload_file(upload, self.model)
            result = client.transcribe(self.model, file_url, self._parameters(options))
            transcript = parse_transcription(result, provider="dashscope", model=self.model)
        if options.language:
            transcript.language = options.language
        logger.info(
            "DashScope %s returned %s segments, %s speakers",
            self.model,
            len(transcript.segments),
            len({segment.speaker for segment in transcript.segments if segment.speaker}),
        )
        return transcript

    def _transcribe_chunked(
        self,
        client,
        audio_path: Path,
        duration: float,
        chunk: float,
        overlap: float,
        options: ASROptions,
    ) -> Transcript:
        """Diarization is only reliable up to ~2 h per request, so long audio goes in overlapping windows."""
        from concurrent.futures import ThreadPoolExecutor

        from media_pipeline.asr.windows import Piece, plan_windows, stitch_windows

        windows = plan_windows(duration, chunk, overlap)
        logger.info("DashScope %s: %.0fs of audio in %s windows (overlap %.0fs)", self.model, duration, len(windows), overlap)
        parameters = self._parameters(options)

        def work(item: tuple[int, tuple[float, float]]) -> list[Piece]:
            index, (start, end) = item
            with tempfile.TemporaryDirectory(prefix="dashscope-asr-") as tmp:
                clip = _compress_for_upload(audio_path, Path(tmp), start=start, end=end)
                file_url = client.upload_file(clip, self.model)
            result = client.transcribe(self.model, file_url, parameters)
            part = parse_transcription(result, provider="dashscope", model=self.model)
            return [
                Piece(start + s.start, start + s.end, s.text, f"w{index}:{s.speaker or 'SPEAKER_00'}")
                for s in part.segments
            ]

        with ThreadPoolExecutor(max_workers=min(3, len(windows))) as pool:
            results = list(pool.map(work, enumerate(windows)))
        pieces = stitch_windows([(s, e, found) for (s, e), found in zip(windows, results)])
        return Transcript(
            language=None,
            segments=[TranscriptSegment(start=p.start, end=p.end, text=p.text, speaker=p.speaker) for p in pieces],
            provider="dashscope",
            model=self.model,
        )

    def _parameters(self, options: ASROptions) -> dict[str, Any]:
        params: dict[str, Any] = {"channel_id": [0], "diarization_enabled": True}
        dash = getattr(self.config, "dashscope", None)
        count = int(getattr(dash, "speaker_count", 0) or 0)
        if 2 <= count <= 100:
            params["speaker_count"] = count
        if options.language:
            params["language_hints"] = [options.language]
        return params

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        from media_pipeline.dashscope import DashScopeError, client_from_config

        config = self.config
        if config is None:
            from media_pipeline.config import load_config

            config = load_config()
        try:
            self._client = client_from_config(config)
        except DashScopeError as exc:
            raise ASRNotAvailableError(str(exc)) from exc
        return self._client


def parse_transcription(result: dict[str, Any], *, provider: str, model: str) -> Transcript:
    """Turn a DashScope transcription result JSON into a Transcript with optional speakers."""
    transcripts = result.get("transcripts") or []
    channel = transcripts[0] if transcripts else {}
    speaker_map: dict[Any, str] = {}
    segments: list[TranscriptSegment] = []
    for sentence in channel.get("sentences") or []:
        text = str(sentence.get("text") or "").strip()
        if not text:
            continue
        speaker = None
        raw_speaker = sentence.get("speaker_id")
        if raw_speaker is not None:
            if raw_speaker not in speaker_map:
                speaker_map[raw_speaker] = f"SPEAKER_{len(speaker_map):02d}"
            speaker = speaker_map[raw_speaker]
        words = [
            WordSpan(
                start=_seconds(word.get("begin_time")),
                end=_seconds(word.get("end_time")),
                text=f"{word.get('text') or ''}{word.get('punctuation') or ''}",
            )
            for word in sentence.get("words") or []
            if str(word.get("text") or "").strip()
        ]
        segments.append(
            TranscriptSegment(
                start=_seconds(sentence.get("begin_time")),
                end=_seconds(sentence.get("end_time")),
                text=text,
                words=words or None,
                speaker=speaker,
            )
        )
    segments.sort(key=lambda item: (item.start, item.end))
    return Transcript(language=None, segments=segments, provider=provider, model=model)


def _compress_for_upload(audio_path: Path, tmp_dir: Path, start: float | None = None, end: float | None = None) -> Path:
    """Shrink 16 kHz PCM WAV (≈115 MB/hour) to mono AAC (≈20 MB/hour) before uploading; optionally cut a window."""
    ffmpeg = shutil.which("ffmpeg")
    trimming = start is not None and end is not None
    if not ffmpeg:
        if trimming:
            raise ASRNotAvailableError("ffmpeg is required to split long audio for Fun-ASR")
        return audio_path
    suffix = f"-{int(start)}" if trimming else ""
    target = tmp_dir / f"{audio_path.stem}{suffix}.m4a"
    command = [ffmpeg, "-y", "-loglevel", "error"]
    if trimming:
        command += ["-ss", f"{start:.3f}", "-to", f"{end:.3f}"]
    command += ["-i", str(audio_path), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "48k", str(target)]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0 or not target.exists() or target.stat().st_size == 0:
        if trimming:
            raise ASRNotAvailableError(f"ffmpeg could not cut audio: {completed.stderr.strip()[-300:]}")
        logger.warning("Audio compression failed, uploading original: %s", completed.stderr.strip()[-300:])
        return audio_path
    return target


def _probe_duration(path: Path) -> float | None:
    from media_pipeline.media import probe_duration

    try:
        return probe_duration(path)
    except Exception:
        return None


def _seconds(value: object) -> float:
    try:
        return round(float(value) / 1000.0, 3)
    except (TypeError, ValueError):
        return 0.0
