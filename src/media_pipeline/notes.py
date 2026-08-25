from __future__ import annotations

import re
from pathlib import Path

from media_pipeline.models import (
    NamedSegment,
    Task,
    TaskStatus,
    VideoMetadata,
    asr_label,
    format_duration,
    now_stamp,
)
from media_pipeline.transcript import render_transcript
from media_pipeline.summary import extract_summary_section, strip_summary_section

_INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_HEADER_END = re.compile(r"\n---\s*\n", re.MULTILINE)
_FIELD = re.compile(r"^(URL|Platform|Kind|Author|Duration|Published|Thumbnail|Description|ASR Model|Language Mode|Languages|Status|Video Path|Audio Path|Error Stage|Error|Created|Updated):\s*(.*)$")


class NoteWriter:
    def __init__(self, notes_dir: Path) -> None:
        self.notes_dir = notes_dir
        self.notes_dir.mkdir(parents=True, exist_ok=True)

    def create_or_open(self, metadata: VideoMetadata, task: Task) -> Path:
        path = self._path_for(metadata)
        if path.exists():
            existing = NoteDocument.parse(path.read_text(encoding="utf-8"))
            if existing.fields.get("URL") and existing.fields.get("URL") != metadata.url:
                path = self._path_for(metadata, disambiguate=True)
        created = now_stamp()
        if path.exists():
            document = NoteDocument.parse(path.read_text(encoding="utf-8"))
            created = document.fields.get("Created") or created
        document = NoteDocument.from_metadata(metadata, task, created=created)
        path.write_text(document.render(), encoding="utf-8")
        return path

    def update_progress(
        self,
        path: Path,
        task: Task,
        metadata: VideoMetadata | None = None,
        transcript: list[NamedSegment] | None = None,
        body: str | None = None,
    ) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        document = NoteDocument.parse(path.read_text(encoding="utf-8"))
        document.apply_task(task, metadata)
        if body is not None:
            _apply_body(document, body)
        elif transcript is not None:
            _apply_body(document, render_transcript(transcript))
        path.write_text(document.render(), encoding="utf-8")

    def update_summary(self, path: Path, markdown: str) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        document = NoteDocument.parse(path.read_text(encoding="utf-8"))
        document.summary_markdown = (markdown or "").strip()
        document.transcript_markdown = strip_summary_section(document.transcript_markdown)
        path.write_text(document.render(), encoding="utf-8")

    def _path_for(self, metadata: VideoMetadata, disambiguate: bool = False) -> Path:
        title = sanitize_filename(metadata.title) or metadata.video_id or "untitled"
        if disambiguate and metadata.video_id:
            title = f"{title} [{metadata.video_id}]"
        return self.notes_dir / f"{title}.md"


def load_note(path: Path, *, rewrite_layout: bool = False) -> NoteDocument:
    raw = path.read_text(encoding="utf-8")
    document = NoteDocument.parse(raw)
    if rewrite_layout and _summary_is_below_divider(raw, document):
        path.write_text(document.render(), encoding="utf-8")
    return document


def sanitize_filename(title: str, max_length: int = 120) -> str:
    cleaned = _INVALID_FILENAME.sub("", title or "")
    cleaned = cleaned.strip(" .")
    cleaned = re.sub(r"\s+", " ", cleaned)
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip(" .")
    return cleaned


class NoteDocument:
    def __init__(
        self,
        title: str,
        fields: dict[str, str],
        transcript_markdown: str = "",
        summary_markdown: str = "",
    ) -> None:
        self.title = title
        self.fields = fields
        self.transcript_markdown = transcript_markdown
        self.summary_markdown = summary_markdown

    @classmethod
    def from_metadata(cls, metadata: VideoMetadata, task: Task, created: str | None = None) -> NoteDocument:
        stamp = created or now_stamp()
        fields = {
            "URL": metadata.url,
            "Platform": metadata.platform,
            "Kind": _kind_label(metadata, task),
            "Author": metadata.author,
            "Duration": format_duration(metadata.duration),
            "Published": metadata.published,
            "Thumbnail": metadata.thumbnail_url,
            "Description": _one_line(metadata.description, 500),
            "Status": task.status.value if task.status != TaskStatus.queued else TaskStatus.downloading.value,
            "Video Path": task.video_path,
            "Created": stamp,
            "Updated": now_stamp(),
        }
        if not _is_image_post(metadata, task):
            fields["ASR Model"] = asr_label(task.asr_model)
            fields["Language Mode"] = str(task.extra.get("language") or "auto")
            fields["Languages"] = str(task.extra.get("detected_languages") or "")
            fields["Audio Path"] = task.audio_path
        return cls(title=metadata.title, fields=fields)

    def apply_task(self, task: Task, metadata: VideoMetadata | None = None) -> None:
        if metadata is not None:
            self.title = metadata.title or self.title
            self.fields["URL"] = metadata.url
            self.fields["Platform"] = metadata.platform
            self.fields["Kind"] = _kind_label(metadata, task)
            self.fields["Author"] = metadata.author
            self.fields["Duration"] = format_duration(metadata.duration)
            self.fields["Published"] = metadata.published
            self.fields["Thumbnail"] = metadata.thumbnail_url
            self.fields["Description"] = _one_line(metadata.description, 500)
        if _is_image_post(metadata, task):
            self.fields.pop("ASR Model", None)
            self.fields.pop("Language Mode", None)
            self.fields.pop("Languages", None)
            self.fields.pop("Audio Path", None)
        else:
            self.fields["ASR Model"] = asr_label(task.asr_model)
            self.fields["Language Mode"] = str(task.extra.get("language") or "auto")
            detected = str(task.extra.get("detected_languages") or "")
            if detected:
                self.fields["Languages"] = detected
            else:
                self.fields.pop("Languages", None)
            self.fields["Audio Path"] = task.audio_path
        self.fields["Status"] = task.status.value
        self.fields["Video Path"] = task.video_path
        self.fields["Updated"] = now_stamp()
        if task.status == TaskStatus.failed:
            self.fields["Error Stage"] = task.error_stage
            self.fields["Error"] = task.error
        else:
            self.fields.pop("Error Stage", None)
            self.fields.pop("Error", None)

    def render(self) -> str:
        lines = [f"# {self.title}", ""]
        if self.summary_markdown.strip():
            lines.append("## Summary")
            lines.append("")
            lines.append(self.summary_markdown.strip())
            lines.append("")
        order = [
            "URL",
            "Platform",
            "Kind",
            "Author",
            "Duration",
            "Published",
            "Thumbnail",
            "Description",
            "ASR Model",
            "Language Mode",
            "Languages",
            "Status",
            "Video Path",
            "Audio Path",
            "Error Stage",
            "Error",
            "Created",
            "Updated",
        ]
        for key in order:
            if key not in self.fields:
                continue
            value = self.fields.get(key, "")
            if key in {"Thumbnail", "Description", "Languages", "Duration", "Audio Path"} and not value:
                continue
            lines.append(f"{key}: {value}")
        if self.transcript_markdown.strip():
            lines.append("")
            lines.append("---")
            lines.append("")
            body = self.transcript_markdown.strip()
            lines.append(body)
        lines.append("")
        return "\n".join(lines)

    @classmethod
    def parse(cls, text: str) -> NoteDocument:
        header, transcript = _split_note(text)
        lines = header.splitlines()
        title = "Untitled"
        fields: dict[str, str] = {}
        if lines and lines[0].startswith("# "):
            title = lines[0][2:].strip()
            lines = lines[1:]
        summary, lines = _split_header_summary(lines)
        for line in lines:
            match = _FIELD.match(line)
            if match:
                fields[match.group(1)] = match.group(2)
        if not summary:
            summary = extract_summary_section(transcript)
            if summary:
                transcript = strip_summary_section(transcript)
        return cls(title=title, fields=fields, transcript_markdown=transcript, summary_markdown=summary)


def _split_note(text: str) -> tuple[str, str]:
    match = _HEADER_END.search(text)
    if not match:
        return text.strip(), ""
    return text[: match.start()].strip(), text[match.end() :].strip()


def _one_line(value: str, limit: int) -> str:
    collapsed = re.sub(r"\s+", " ", value or "").strip()
    if len(collapsed) > limit:
        return collapsed[: limit - 1].rstrip() + "…"
    return collapsed


def _is_image_post(metadata: VideoMetadata | None, task: Task) -> bool:
    if metadata is not None and metadata.media_kind == "image":
        return True
    return str(task.extra.get("media_kind") or "") == "image"


def _kind_label(metadata: VideoMetadata | None, task: Task) -> str:
    return "Image" if _is_image_post(metadata, task) else "Video"


def _apply_body(document: NoteDocument, incoming: str) -> None:
    extracted = extract_summary_section(incoming)
    if extracted:
        document.summary_markdown = extracted
        document.transcript_markdown = strip_summary_section(incoming)
        return
    document.transcript_markdown = incoming


def _summary_is_below_divider(raw: str, document: NoteDocument) -> bool:
    if not document.summary_markdown.strip():
        return False
    header, _body = _split_note(raw)
    return not any(line.strip() == "## Summary" for line in header.splitlines())


def _split_header_summary(lines: list[str]) -> tuple[str, list[str]]:
    rest = list(lines)
    while rest and not rest[0].strip():
        rest.pop(0)
    if not rest or rest[0].strip() != "## Summary":
        return "", rest
    rest.pop(0)
    while rest and not rest[0].strip():
        rest.pop(0)
    summary_lines: list[str] = []
    while rest:
        if _FIELD.match(rest[0]):
            break
        summary_lines.append(rest.pop(0))
    while summary_lines and not summary_lines[-1].strip():
        summary_lines.pop()
    return "\n".join(summary_lines).strip(), rest
