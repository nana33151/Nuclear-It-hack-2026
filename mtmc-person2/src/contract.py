"""Shared data contracts with Person 1 (input) and Person 3 (output)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Detection:
    """One silhouette/detection row from Person 1."""

    camera_id: str
    frame_idx: int
    timestamp: float
    bbox: list[float]  # x1, y1, x2, y2
    score: float
    usable_for_reid: bool
    mask: Any = None
    keypoints: Any = None
    crop_path: str | None = None
    color_top: list[float] | None = None
    color_bottom: list[float] | None = None
    det_id: str | int | None = None

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Detection:
        required = ("camera_id", "frame_idx", "timestamp", "bbox", "score", "usable_for_reid")
        missing = [k for k in required if k not in row]
        if missing:
            raise ValueError(f"Detection missing fields: {missing}")
        bbox = [float(x) for x in row["bbox"]]
        if len(bbox) != 4:
            raise ValueError(f"bbox must have 4 values, got {len(bbox)}")
        return cls(
            camera_id=str(row["camera_id"]),
            frame_idx=int(row["frame_idx"]),
            timestamp=float(row["timestamp"]),
            bbox=bbox,
            score=float(row["score"]),
            usable_for_reid=bool(row["usable_for_reid"]),
            mask=row.get("mask"),
            keypoints=row.get("keypoints"),
            crop_path=row.get("crop_path"),
            color_top=_optional_float_list(row.get("color_top")),
            color_bottom=_optional_float_list(row.get("color_bottom")),
            det_id=row.get("det_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BoxObservation:
    frame_idx: int
    timestamp: float
    bbox: list[float]
    score: float
    usable_for_reid: bool = False


@dataclass
class Tracklet:
    """Local tracklet handed to Person 3."""

    camera_id: str
    local_id: str
    t_start: float
    t_end: float
    boxes: list[BoxObservation]
    reid_emb: list[float] | None
    color_top: list[float] | None
    color_bottom: list[float] | None
    representative_frame: int
    split_reason: str | None = None
    quality: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "local_id": self.local_id,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "boxes": [asdict(b) for b in self.boxes],
            "reid_emb": self.reid_emb,
            "color_top": self.color_top,
            "color_bottom": self.color_bottom,
            "representative_frame": self.representative_frame,
            "split_reason": self.split_reason,
            "quality": self.quality,
        }


def _optional_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    return [float(x) for x in value]
