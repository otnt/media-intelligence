from __future__ import annotations

import logging
import shutil
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from typing import Iterator

from media_pipeline.align import align_transcript
from media_pipeline.artifacts import ArtifactStore
from media_pipeline.asr.base import ASRNotAvailableError
from media_pipeline.asr.language import format_detected_languages, resolve_provider_language
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
from media_pipeline.models import ASR_MODELS, ASROptions, NamedSegment, Task, TaskStatus, VideoMetadata, asr_label
from media_pipeline.notes import NoteWriter
from media_pipeline.speakers import name_speakers
from media_pipeline.stage_timing import begin_stage, clear_invalidated_timings, finish_stage
from media_pipeline.store import TaskStore
from media_pipeline.transcript import clean_transcript, render_transcript
from media_pipeline.visual.extract import VisualExtractor, copy_keyframes_to_vault
from media_pipeline.visual.filtering import caption_for
from media_pipeline.visual.vlm import VisionProvider, build_vision_provider

logger = logging.getLogger(__name__)

RERUN_STAGES = frozenset(
    {
        "transcribing",
        "diarizing",
        "aligning_transcript",
        "detecting_scenes",
        "sampling_frames",
        "deduplicating_frames",
        "filtering_frames",
        "aligning_multimodal",
        "writing_outputs",
        "all",
    }
)
VISUAL_RERUN_STAGES = frozenset(
    {
        "detecting_scenes",
        "sampling_frames",
        "deduplicating_frames",
        "filtering_frames",
        "aligning_multimodal",
    }
)


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
        visual: VisualExtractor | None = None,
        vision: VisionProvider | None = None,
        model_lock: AbstractContextManager[object] | None = None,
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
        self.vision = vision if vision is not None else build_vision_provider(config)
        self.visual = visual if visual is not None else VisualExtractor(vision=self.vision, model_lock=model_lock)
        self._model_lock = model_lock

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
        from_stage = str(task.extra.pop("rerun_stage", "") or "")
        metadata = self._fetch_metadata(task)
        artifacts = ArtifactStore(self.config.paths.artifacts, metadata.video_id)
        artifacts.save_metadata(metadata)
        if from_stage:
            artifacts.invalidate_from(from_stage)
            clear_invalidated_timings(task.extra, from_stage)
        self._persist_finished_timings(task, artifacts)
        self._create_note(task, metadata, artifacts)

        video_path = self._download(task, metadata, artifacts)
        task.extra["media_kind"] = metadata.media_kind
        self.store.update(task)
        if metadata.media_kind == "image":
            self._write_image_post(task, metadata, artifacts, video_path)
            task.status = TaskStatus.completed
            task.error = ""
            task.error_stage = ""
            task.extra.pop("rerun_stage", None)
            self.store.update(task)
            if task.note_path:
                self.notes.update_progress(Path(task.note_path), task, metadata)
            logger.info("Task %s completed Xiaohongshu image post %s", task.id, metadata.video_id)
            return task
        audio_path = self._extract_audio(task, metadata, video_path, artifacts)
        transcript = self._transcribe(task, metadata, artifacts, audio_path)
        diarization = self._diarize(task, artifacts, audio_path)
        named = self._align(task, artifacts, transcript, diarization, metadata)
        task.extra["segment_count"] = len(named)
        self.store.update(task)

        visual_result: dict = {}
        if _wants_keyframes(task, from_stage):
            visual_result = self._extract_visual(task, metadata, artifacts, video_path, named, from_stage)
        else:
            task.extra.setdefault("extract_keyframes", False)
            task.extra["candidate_count"] = 0
            task.extra["keyframe_count"] = 0
            task.extra["selected_count"] = 0
            self.store.update(task)
        self._write_outputs(task, metadata, artifacts, named, visual_result)
        task.status = TaskStatus.completed
        task.error = ""
        task.error_stage = ""
        task.extra.pop("rerun_stage", None)
        self.store.update(task)
        if task.note_path:
            self.notes.update_progress(Path(task.note_path), task, metadata)
        logger.info("Task %s completed with %s", task.id, asr_label(task.asr_model))
        return task

    def close(self) -> None:
        closer = getattr(self.vision, "close", None)
        if callable(closer):
            closer()

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
                task.extra["media_kind"] = cached.media_kind
                if not task.video_path:
                    task.video_path = str(_media_destination(self.config, cached))
                self.store.update(task)
                return cached
        with self._stage(task, TaskStatus.fetching_metadata):
            metadata = fetch_metadata(task.url, task.asr_model, self.config)
        task.video_id = metadata.video_id
        task.platform = metadata.platform
        task.title = metadata.title
        task.extra["media_kind"] = metadata.media_kind
        task.video_path = str(_media_destination(self.config, metadata))
        self.store.update(task)
        return metadata

    def _create_note(self, task: Task, metadata: VideoMetadata, artifacts: ArtifactStore) -> None:
        task.status = TaskStatus.downloading
        note_path = self.notes.create_or_open(metadata, task)
        task.note_path = str(note_path)
        artifacts.remember_note(note_path)
        self.store.update(task)
        self.notes.update_progress(note_path, task, metadata)

    def _download(self, task: Task, metadata: VideoMetadata, artifacts: ArtifactStore) -> Path:
        existing = find_media_file(self.config.paths.videos, metadata.video_id)
        if existing:
            task.video_path = str(existing)
            self.store.update(task)
            return existing
        with self._stage(task, TaskStatus.downloading, metadata, artifacts):
            path = download_video(metadata.url, metadata.video_id, self.config.paths.videos, self.config)
            task.video_path = str(path)
            return path

    def _extract_audio(
        self, task: Task, metadata: VideoMetadata, video_path: Path, artifacts: ArtifactStore
    ) -> Path:
        audio_path = self.config.paths.audio / f"{metadata.video_id}.wav"
        if audio_path.exists() and audio_path.stat().st_size > 0:
            task.audio_path = str(audio_path)
            self.store.update(task)
            return audio_path
        with self._stage(task, TaskStatus.extracting_audio, metadata, artifacts):
            path = extract_audio(video_path, audio_path)
            task.audio_path = str(path)
            return path

    def _transcribe(self, task: Task, metadata: VideoMetadata, artifacts: ArtifactStore, audio_path: Path):
        existing = artifacts.load_transcript(task.asr_model)
        if existing and existing.segments:
            self._record_languages(task, existing.language)
            return existing
        with self._stage(task, TaskStatus.transcribing, metadata, artifacts):
            provider = require_provider(task.asr_model)
            context = _asr_context(metadata)
            provider_name = ASR_MODELS[task.asr_model]["provider"]
            requested_language = str(task.extra.get("language") or self.config.asr.language or "auto")
            task.extra["language"] = requested_language
            with self._model_slot():
                transcript = provider.transcribe(
                    audio_path,
                    ASROptions(
                        context=context,
                        language=resolve_provider_language(provider_name, requested_language),
                    ),
                )
            transcript = clean_transcript(transcript)
            self._record_languages(task, transcript.language)
            artifacts.save_transcript(task.asr_model, transcript)
            return transcript

    def _diarize(self, task: Task, artifacts: ArtifactStore, audio_path: Path):
        existing = artifacts.load_diarization()
        if existing is not None:
            return existing
        with self._stage(task, TaskStatus.diarizing, artifacts=artifacts):
            try:
                with self._model_slot():
                    result = self.diarization.diarize(audio_path)
            except Exception as exc:
                logger.warning("Diarization failed for task %s: %s", task.id, exc)
                fallback = NullDiarizationProvider()
                result = fallback.diarize(audio_path)
                result.provider = f"fallback:{exc}"
            artifacts.save_diarization(result)
            return result

    def _align(
        self,
        task: Task,
        artifacts: ArtifactStore,
        transcript,
        diarization,
        metadata: VideoMetadata,
    ):
        existing = artifacts.load_named()
        if existing is not None:
            return existing
        with self._stage(task, TaskStatus.aligning_transcript, metadata, artifacts):
            aligned = align_transcript(transcript, diarization.segments)
            artifacts.save_aligned(aligned)
            named = name_speakers(aligned, metadata)
            artifacts.save_named(named)
            return named

    def _extract_visual(
        self,
        task: Task,
        metadata: VideoMetadata,
        artifacts: ArtifactStore,
        video_path: Path,
        named: list[NamedSegment],
        from_stage: str,
    ) -> dict:
        settings = self.config.visual.merged(task.extra.get("visual"))
        task.extra["visual"] = settings
        self.store.update(task)
        open_stage: list[str] = []

        def progress(status: str, extra: dict) -> None:
            payload = dict(extra or {})
            event = str(payload.pop("_event", "start") or "start")
            for key, value in payload.items():
                task.extra[key] = value
            if event == "skip":
                self.store.update(task)
                return
            if event == "done":
                self._complete_stage(task, status, artifacts, succeeded=True)
                if open_stage and open_stage[0] == status:
                    open_stage.clear()
                return
            if open_stage:
                self._complete_stage(task, open_stage[0], artifacts, succeeded=True)
                open_stage.clear()
            begin_stage(task.extra, status)
            open_stage.append(status)
            self._set_status(task, TaskStatus(status), metadata)

        visual_from = from_stage if from_stage in RERUN_STAGES else ""
        result = self.visual.run(
            video_path,
            artifacts,
            metadata,
            named,
            settings,
            from_stage=visual_from,
            progress=progress,
        )
        if open_stage:
            self._complete_stage(task, open_stage[0], artifacts, succeeded=True)
        task.extra["candidate_count"] = len(result.get("candidates") or [])
        task.extra["keyframe_count"] = len(result.get("keyframes") or [])
        task.extra["selected_count"] = len(result.get("selected") or result.get("keyframes") or [])
        self.store.update(task)
        return result

    def _write_outputs(
        self,
        task: Task,
        metadata: VideoMetadata,
        artifacts: ArtifactStore,
        named: list[NamedSegment],
        visual_result: dict,
    ) -> None:
        with self._stage(task, TaskStatus.writing_outputs, metadata, artifacts):
            keyframes = visual_result.get("selected") or visual_result.get("keyframes") or []
            analysis = visual_result.get("analysis") or []
            notes_dir = self.config.notes_dir()
            if notes_dir is not None:
                attachment_dir = notes_dir / "attachments" / metadata.video_id
                copy_keyframes_to_vault(keyframes, artifacts.root, attachment_dir)
            frames = [
                (frame.timestamp, frame.image_path, caption_for(frame, analysis))
                for frame in keyframes
            ]
            body = render_transcript(named, video_id=metadata.video_id, frames=frames)
            if task.note_path:
                self.notes.update_progress(Path(task.note_path), task, metadata, named, body=body)

    def _write_image_post(
        self,
        task: Task,
        metadata: VideoMetadata,
        artifacts: ArtifactStore,
        media_path: Path,
    ) -> None:
        from media_pipeline.xhs import find_xhs_images, render_image_post

        with self._stage(task, TaskStatus.writing_outputs, metadata, artifacts):
            images = find_xhs_images(media_path)
            if not images:
                raise PipelineError(TaskStatus.writing_outputs, "Xiaohongshu image files were not found")
            names = [path.name for path in images]
            notes_dir = self.config.notes_dir()
            if notes_dir is not None:
                attachment_dir = notes_dir / "attachments" / metadata.video_id
                attachment_dir.mkdir(parents=True, exist_ok=True)
                for image in images:
                    shutil.copy2(image, attachment_dir / image.name)
            body = render_image_post(metadata.video_id, metadata.description, names)
            if task.note_path:
                self.notes.update_progress(Path(task.note_path), task, metadata, body=body)
            task.extra["image_count"] = len(images)
            self.store.update(task)

    def _set_status(self, task: Task, status: TaskStatus, metadata: VideoMetadata | None = None) -> None:
        task.status = status
        self.store.update(task)
        if task.note_path:
            self.notes.update_progress(Path(task.note_path), task, metadata)

    @contextmanager
    def _stage(
        self,
        task: Task,
        status: TaskStatus,
        metadata: VideoMetadata | None = None,
        artifacts: ArtifactStore | None = None,
    ) -> Iterator[None]:
        begin_stage(task.extra, status.value)
        self._set_status(task, status, metadata)
        try:
            yield
        except Exception:
            self._complete_stage(task, status.value, artifacts, succeeded=False)
            raise
        else:
            self._complete_stage(task, status.value, artifacts, succeeded=True)

    def _complete_stage(
        self,
        task: Task,
        key: str,
        artifacts: ArtifactStore | None,
        *,
        succeeded: bool,
    ) -> None:
        entry = finish_stage(task.extra, key, succeeded=succeeded)
        if succeeded and artifacts is not None and entry:
            artifacts.save_stage_timing(key, entry)
        self.store.update(task)

    def _persist_finished_timings(self, task: Task, artifacts: ArtifactStore) -> None:
        timings = task.extra.get("stage_timings")
        if not isinstance(timings, dict):
            return
        for key, entry in timings.items():
            if isinstance(entry, dict) and entry.get("status") == "succeeded":
                artifacts.save_stage_timing(key, entry)

    def _fail(self, task: Task, stage: TaskStatus, message: str) -> Task:
        finish_stage(task.extra, stage.value, succeeded=False)
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

    def _model_slot(self) -> AbstractContextManager[object]:
        return self._model_lock if self._model_lock is not None else nullcontext()

    def _record_languages(self, task: Task, detected: object | None) -> None:
        task.extra.setdefault("language", self.config.asr.language or "auto")
        label = format_detected_languages(detected)
        if label:
            task.extra["detected_languages"] = label
        self.store.update(task)


def _media_destination(config: AppConfig, metadata: VideoMetadata) -> Path:
    if metadata.media_kind == "image":
        return config.paths.videos / metadata.video_id
    return config.paths.videos / f"{metadata.video_id}.mp4"


def _wants_keyframes(task: Task, from_stage: str = "") -> bool:
    if from_stage in VISUAL_RERUN_STAGES:
        task.extra["extract_keyframes"] = True
        return True
    return bool(task.extra.get("extract_keyframes"))


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
