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

_INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_HEADER_END = re.compile(r"\n---\s*\n", re.MULTILINE)
_FIELD = re.compile(r"^(URL|Platform|Author|Duration|Published|Thumbnail|Description|ASR Model|Language Mode|Languages|Status|Video Path|Audio Path|Error Stage|Error|Created|Updated):\s*(.*)$")


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
            document.transcript_markdown = body
        elif transcript is not None:
            document.transcript_markdown = render_transcript(transcript)
        path.write_text(document.render(), encoding="utf-8")

    def _path_for(self, metadata: VideoMetadata, disambiguate: bool = False) -> Path:
        title = sanitize_filename(metadata.title) or metadata.video_id or "untitled"
        if disambiguate and metadata.video_id:
            title = f"{title} [{metadata.video_id}]"
        return self.notes_dir / f"{title}.md"


def sanitize_filename(title: str, max_length: int = 120) -> str:
    cleaned = _INVALID_FILENAME.sub("", title or "")
    cleaned = cleaned.strip(" .")
    cleaned = re.sub(r"\s+", " ", cleaned)
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip(" .")
    return cleaned


class NoteDocument:
    def __init__(self, title: str, fields: dict[str, str], transcript_markdown: str = "") -> None:
        self.title = title
        self.fields = fields
        self.transcript_markdown = transcript_markdown

    @classmethod
    def from_metadata(cls, metadata: VideoMetadata, task: Task, created: str | None = None) -> NoteDocument:
        stamp = created or now_stamp()
        fields = {
            "URL": metadata.url,
            "Platform": metadata.platform,
            "Author": metadata.author,
            "Duration": format_duration(metadata.duration),
            "Published": metadata.published,
            "Thumbnail": metadata.thumbnail_url,
            "Description": _one_line(metadata.description, 500),
            "ASR Model": asr_label(task.asr_model),
            "Language Mode": str(task.extra.get("language") or "auto"),
            "Languages": str(task.extra.get("detected_languages") or ""),
            "Status": task.status.value if task.status != TaskStatus.queued else TaskStatus.downloading.value,
            "Video Path": task.video_path,
            "Audio Path": task.audio_path,
            "Created": stamp,
            "Updated": now_stamp(),
        }
        return cls(title=metadata.title, fields=fields)

    def apply_task(self, task: Task, metadata: VideoMetadata | None = None) -> None:
        if metadata is not None:
            self.title = metadata.title or self.title
            self.fields["URL"] = metadata.url
            self.fields["Platform"] = metadata.platform
            self.fields["Author"] = metadata.author
            self.fields["Duration"] = format_duration(metadata.duration)
            self.fields["Published"] = metadata.published
            self.fields["Thumbnail"] = metadata.thumbnail_url
            self.fields["Description"] = _one_line(metadata.description, 500)
        self.fields["ASR Model"] = asr_label(task.asr_model)
        self.fields["Language Mode"] = str(task.extra.get("language") or "auto")
        detected = str(task.extra.get("detected_languages") or "")
        if detected:
            self.fields["Languages"] = detected
        else:
            self.fields.pop("Languages", None)
        self.fields["Status"] = task.status.value
        self.fields["Video Path"] = task.video_path
        self.fields["Audio Path"] = task.audio_path
        self.fields["Updated"] = now_stamp()
        if task.status == TaskStatus.failed:
            self.fields["Error Stage"] = task.error_stage
            self.fields["Error"] = task.error
        else:
            self.fields.pop("Error Stage", None)
            self.fields.pop("Error", None)

    def render(self) -> str:
        lines = [f"# {self.title}", ""]
        order = [
            "URL",
            "Platform",
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
            if key in {"Thumbnail", "Description", "Languages"} and not value:
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
        for line in lines:
            match = _FIELD.match(line)
            if match:
                fields[match.group(1)] = match.group(2)
        return cls(title=title, fields=fields, transcript_markdown=transcript)


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
