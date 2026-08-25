from __future__ import annotations

import logging
from pathlib import Path

from media_pipeline.align import align_transcript
from media_pipeline.artifacts import ArtifactStore
from media_pipeline.asr.base import ASRNotAvailableError
from media_pipeline.asr.registry import require_provider
from media_pipeline.config import AppConfig
from media_pipeline.diarization import DiarizationProvider, NullDiarizationProvider, build_diarization_provider
from media_pipeline.media import (
    MediaError,
    UnsupportedURLError,
    download_video,
    extract_audio,
    fetch_metadata,
    find_media_file,
    parse_video_ref,
)
from media_pipeline.models import ASROptions, Task, TaskStatus, VideoMetadata, asr_label
from media_pipeline.notes import NoteWriter
from media_pipeline.speakers import name_speakers
from media_pipeline.store import TaskStore
from media_pipeline.transcript import clean_transcript

logger = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    def __init__(self, stage: TaskStatus | str, message: str) -> None:
        super().__init__(message)
        self.stage = stage.value if isinstance(stage, TaskStatus) else stage


class Pipeline:
    def __init__(
        self,
        config: AppConfig,
        store: TaskStore,
        diarization: DiarizationProvider | None = None,
    ) -> None:
        self.config = config
        self.store = store
        notes_dir = config.notes_dir()
        if notes_dir is None:
            raise PipelineError(TaskStatus.fetching_metadata, "Obsidian vault is not configured")
        self.notes = NoteWriter(notes_dir)
        self.diarization = diarization or build_diarization_provider(
            config.diarization.provider,
            config.diarization.model,
            config.diarization.hf_token,
        )

    def run(self, task: Task) -> Task:
        try:
            return self._run(task)
        except PipelineError as exc:
            return self._fail(task, TaskStatus(exc.stage) if _is_status(exc.stage) else task.status, str(exc))
        except MediaError as exc:
            stage = TaskStatus(exc.stage) if _is_status(exc.stage) else task.status
            return self._fail(task, stage, str(exc))
        except ASRNotAvailableError as exc:
            return self._fail(task, TaskStatus.transcribing, str(exc))
        except Exception as exc:
            logger.exception("Task %s failed unexpectedly", task.id)
            stage = task.status if task.status != TaskStatus.queued else TaskStatus.fetching_metadata
            return self._fail(task, stage, f"{type(exc).__name__}: {exc}")

    def _run(self, task: Task) -> Task:
        metadata = self._fetch_metadata(task)
        artifacts = ArtifactStore(self.config.paths.artifacts, metadata.video_id)
        artifacts.save_metadata(metadata)
        self._create_note(task, metadata, artifacts)

        video_path = self._download(task, metadata)
        audio_path = self._extract_audio(task, metadata, video_path)
        transcript = self._transcribe(task, metadata, artifacts, audio_path)
        diarization = self._diarize(task, artifacts, audio_path)
        aligned = self._align(task, artifacts, transcript, diarization)
        named = name_speakers(aligned, metadata)
        artifacts.save_named(named)

        task.status = TaskStatus.completed
        task.error = ""
        task.error_stage = ""
        self.store.update(task)
        if task.note_path:
            self.notes.update_progress(Path(task.note_path), task, metadata, named)
        logger.info("Task %s completed with %s", task.id, asr_label(task.asr_model))
        return task

    def _fetch_metadata(self, task: Task) -> VideoMetadata:
        self._set_status(task, TaskStatus.fetching_metadata)
        video_id = task.video_id
        if not video_id:
            try:
                _, video_id = parse_video_ref(task.url)
            except UnsupportedURLError:
                video_id = ""
        if video_id:
            cached = ArtifactStore(self.config.paths.artifacts, video_id).load_metadata()
            if cached is not None:
                cached.asr_model = task.asr_model
                task.video_id = cached.video_id
                task.platform = cached.platform
                task.title = cached.title
                if not task.video_path:
                    task.video_path = str(self.config.paths.videos / f"{cached.video_id}.mp4")
                self.store.update(task)
                return cached
        metadata = fetch_metadata(task.url, task.asr_model, self.config)
        task.video_id = metadata.video_id
        task.platform = metadata.platform
        task.title = metadata.title
        task.video_path = str(self.config.paths.videos / f"{metadata.video_id}.mp4")
        self.store.update(task)
        return metadata

    def _create_note(self, task: Task, metadata: VideoMetadata, artifacts: ArtifactStore) -> None:
        task.status = TaskStatus.downloading
        note_path = self.notes.create_or_open(metadata, task)
        task.note_path = str(note_path)
        artifacts.remember_note(note_path)
        self.store.update(task)
        self.notes.update_progress(note_path, task, metadata)

    def _download(self, task: Task, metadata: VideoMetadata) -> Path:
        existing = find_media_file(self.config.paths.videos, metadata.video_id)
        if existing:
            task.video_path = str(existing)
            self._set_status(task, TaskStatus.extracting_audio, metadata)
            return existing
        self._set_status(task, TaskStatus.downloading, metadata)
        path = download_video(metadata.url, metadata.video_id, self.config.paths.videos, self.config)
        task.video_path = str(path)
        self._set_status(task, TaskStatus.extracting_audio, metadata)
        return path

    def _extract_audio(self, task: Task, metadata: VideoMetadata, video_path: Path) -> Path:
        audio_path = self.config.paths.audio / f"{metadata.video_id}.wav"
        if audio_path.exists() and audio_path.stat().st_size > 0:
            task.audio_path = str(audio_path)
            self._set_status(task, TaskStatus.transcribing, metadata)
            return audio_path
        self._set_status(task, TaskStatus.extracting_audio, metadata)
        path = extract_audio(video_path, audio_path)
        task.audio_path = str(path)
        self._set_status(task, TaskStatus.transcribing, metadata)
        return path

    def _transcribe(self, task: Task, metadata: VideoMetadata, artifacts: ArtifactStore, audio_path: Path):
        existing = artifacts.load_transcript(task.asr_model)
        if existing and existing.segments:
            return existing
        self._set_status(task, TaskStatus.transcribing, metadata)
        provider = require_provider(task.asr_model)
        context = _asr_context(metadata)
        transcript = provider.transcribe(audio_path, ASROptions(context=context))
        transcript = clean_transcript(transcript)
        artifacts.save_transcript(task.asr_model, transcript)
        return transcript

    def _diarize(self, task: Task, artifacts: ArtifactStore, audio_path: Path):
        existing = artifacts.load_diarization()
        if existing is not None:
            return existing
        self._set_status(task, TaskStatus.diarizing)
        try:
            result = self.diarization.diarize(audio_path)
        except Exception as exc:
            logger.warning("Diarization failed for task %s: %s", task.id, exc)
            fallback = NullDiarizationProvider()
            result = fallback.diarize(audio_path)
            result.provider = f"fallback:{exc}"
        artifacts.save_diarization(result)
        return result

    def _align(self, task: Task, artifacts: ArtifactStore, transcript, diarization):
        self._set_status(task, TaskStatus.aligning)
        aligned = align_transcript(transcript, diarization.segments)
        artifacts.save_aligned(aligned)
        return aligned

    def _set_status(self, task: Task, status: TaskStatus, metadata: VideoMetadata | None = None) -> None:
        task.status = status
        self.store.update(task)
        if task.note_path:
            self.notes.update_progress(Path(task.note_path), task, metadata)

    def _fail(self, task: Task, stage: TaskStatus, message: str) -> Task:
        task.status = TaskStatus.failed
        task.error_stage = stage.value
        task.error = message
        if not task.note_path:
            self._write_stub_failure_note(task)
        self.store.update(task)
        if task.note_path:
            try:
                self.notes.update_progress(Path(task.note_path), task)
            except OSError:
                logger.exception("Could not write failure status to note %s", task.note_path)
        logger.error("Task %s failed at %s: %s", task.id, stage.value, message)
        return task

    def _write_stub_failure_note(self, task: Task) -> None:
        video_id = task.video_id
        if not video_id:
            try:
                _, video_id = parse_video_ref(task.url)
                task.video_id = video_id
            except UnsupportedURLError:
                return
        metadata = VideoMetadata(
            url=task.url,
            title=task.title or video_id,
            platform=task.platform,
            author="",
            video_id=video_id,
            duration=None,
            published="",
            description="",
            thumbnail_url="",
            asr_model=task.asr_model,
        )
        try:
            path = self.notes.create_or_open(metadata, task)
        except OSError:
            logger.exception("Could not create a failure note for task %s", task.id)
            return
        task.note_path = str(path)


def _asr_context(metadata: VideoMetadata) -> str:
    parts = [metadata.title, metadata.author]
    if metadata.description:
        parts.append(metadata.description[:400])
    return "\n".join(part for part in parts if part)


def _is_status(value: str) -> bool:
    try:
        TaskStatus(value)
        return True
    except ValueError:
        return False
