from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from src.contract import Detection


@dataclass
class RawTrack:
    """Pre-ReID / pre-split track from a single-camera MOT backend."""

    local_id: str
    detections: list[Detection] = field(default_factory=list)


class MotBackend(Protocol):
    def track(self, detections: list[Detection]) -> list[RawTrack]:
        """Run single-camera MOT. Input must be one camera only."""
        ...
