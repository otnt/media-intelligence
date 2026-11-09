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


def _compress_for_upload(audio_path: Path, tmp_dir: Path) -> Path:
    """Shrink 16 kHz PCM WAV (≈115 MB/hour) to mono AAC (≈20 MB/hour) before uploading."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return audio_path
    target = tmp_dir / f"{audio_path.stem}.m4a"
    command = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(audio_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "48k", str(target),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0 or not target.exists() or target.stat().st_size == 0:
        logger.warning("Audio compression failed, uploading original: %s", completed.stderr.strip()[-300:])
        return audio_path
    return target


def _seconds(value: object) -> float:
    try:
        return round(float(value) / 1000.0, 3)
    except (TypeError, ValueError):
        return 0.0
