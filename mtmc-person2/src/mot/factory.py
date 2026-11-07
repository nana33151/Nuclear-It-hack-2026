from __future__ import annotations

from typing import Any

from src.mot.bytetrack import ByteTrackMot
from src.mot.stub import StubMot


def build_mot(backend: str, **kwargs: Any):
    backend = backend.lower()
    if backend == "stub":
        return StubMot(
            iou_thresh=float(kwargs.get("iou_thresh", 0.3)),
            max_gap=int(kwargs.get("track_buffer", 30)),
        )
    if backend == "bytetrack":
        return ByteTrackMot(
            track_thresh=float(kwargs.get("track_thresh", 0.5)),
            track_buffer=int(kwargs.get("track_buffer", 30)),
            match_thresh=float(kwargs.get("match_thresh", 0.8)),
            frame_rate=int(kwargs.get("frame_rate", 25)),
            reid_weight=float(kwargs.get("reid_weight", 0.45)),
        )
    raise ValueError(f"Unknown MOT backend: {backend}")
