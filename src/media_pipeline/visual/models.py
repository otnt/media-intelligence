from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SceneSpan:
    start: float
    end: float

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneSpan:
        return cls(start=float(data["start"]), end=float(data["end"]))


@dataclass
class CandidateFrame:
    timestamp: float
    path: str
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"timestamp": self.timestamp, "path": self.path, "sources": list(self.sources)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateFrame:
        sources = [str(item) for item in data.get("sources") or [] if str(item)]
        return cls(timestamp=float(data["timestamp"]), path=str(data.get("path") or ""), sources=sources)


@dataclass
class DedupInfo:
    kept: bool
    nearest_frame: str = ""
    similarity: float = 0.0
    decision: str = "auto"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DedupInfo:
        return cls(
            kept=bool(data.get("kept")),
            nearest_frame=str(data.get("nearest_frame") or ""),
            similarity=float(data.get("similarity") or 0.0),
            decision=str(data.get("decision") or "auto"),
            reason=str(data.get("reason") or ""),
        )


@dataclass
class Keyframe:
    timestamp: float
    image_path: str
    sources: list[str] = field(default_factory=list)
    candidate_sources: list[str] = field(default_factory=list)
    dedup: DedupInfo | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "timestamp": self.timestamp,
            "image_path": self.image_path,
            "sources": list(self.sources or self.candidate_sources),
            "candidate_sources": list(self.candidate_sources or self.sources),
        }
        if self.dedup is not None:
            payload["dedup"] = self.dedup.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Keyframe:
        dedup_raw = data.get("dedup")
        sources = [str(item) for item in data.get("sources") or [] if str(item)]
        candidate_sources = [str(item) for item in data.get("candidate_sources") or sources if str(item)]
        return cls(
            timestamp=float(data["timestamp"]),
            image_path=str(data.get("image_path") or ""),
            sources=sources or candidate_sources,
            candidate_sources=candidate_sources or sources,
            dedup=DedupInfo.from_dict(dedup_raw) if isinstance(dedup_raw, dict) else None,
        )
