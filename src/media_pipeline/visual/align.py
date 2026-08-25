from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from media_pipeline.models import NamedSegment, VideoMetadata
from media_pipeline.visual.models import Keyframe


@dataclass
class TranscriptContext:
    start: float
    end: float
    segments: list[NamedSegment]

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "segments": [
                {
                    "start": item.start,
                    "end": item.end,
                    "speaker": item.speaker_label,
                    "speaker_id": item.speaker_id,
                    "text": item.text,
                }
                for item in self.segments
            ],
        }


@dataclass
class TimelineItem:
    timestamp: float
    frame: str
    transcript_context: TranscriptContext

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "frame": self.frame,
            "transcript_context": self.transcript_context.to_dict(),
        }


def align_keyframes(
    keyframes: list[Keyframe],
    segments: list[NamedSegment],
    *,
    before_sec: float,
    after_sec: float,
) -> list[TimelineItem]:
    timeline: list[TimelineItem] = []
    for frame in keyframes:
        start = max(0.0, frame.timestamp - before_sec)
        end = frame.timestamp + after_sec
        nearby = [
            item
            for item in segments
            if item.end >= start and item.start <= end
        ]
        timeline.append(
            TimelineItem(
                timestamp=frame.timestamp,
                frame=frame.image_path,
                transcript_context=TranscriptContext(start=start, end=end, segments=nearby),
            )
        )
    return timeline


def build_multimodal_document(
    metadata: VideoMetadata,
    segments: list[NamedSegment],
    keyframes: list[Keyframe],
    timeline: list[TimelineItem],
    analysis: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = {
        "metadata": {
            "title": metadata.title,
            "url": metadata.url,
            "platform": metadata.platform,
            "duration": metadata.duration,
            "video_id": metadata.video_id,
            "author": metadata.author,
        },
        "transcript": [
            {
                "start": item.start,
                "end": item.end,
                "speaker": item.speaker_label,
                "speaker_id": item.speaker_id,
                "text": item.text,
            }
            for item in segments
        ],
        "keyframes": [frame.to_dict() for frame in keyframes],
        "timeline": [item.to_dict() for item in timeline],
    }
    if analysis is not None:
        payload["frame_analysis"] = analysis
    return payload


