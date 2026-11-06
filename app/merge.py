"""Cross-camera merge via person3 GlobalTracker."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.local_track import LocalTracklet
from app.paths import load_global_tracker_cls


def to_global_tracks(
    tracklets: list[LocalTracklet],
    *,
    T: float = 8.0,
    min_sim: float = 0.45,
    max_transition_time: float = 20.0,
    method: str = "greedy",
    topology: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    GlobalTracker, Tracklet, _ = load_global_tracker_cls()
    gt_tracklets = [
        Tracklet(
            camera_id=t.camera_id,
            local_id=t.local_id,
            t_start=t.t_start,
            t_end=t.t_end,
            reid=np.asarray(t.reid, dtype=np.float64),
            color_upper=np.asarray(t.color_upper, dtype=np.float64),
            color_lower=np.asarray(t.color_lower, dtype=np.float64),
            boxes=t.boxes,
        )
        for t in tracklets
    ]
    tracker = GlobalTracker(
        T=T,
        min_sim=min_sim,
        max_transition_time=max_transition_time,
        topology=topology or {},
        method=method,
    )
    tracks = tracker.track(gt_tracklets)

    # Map (camera_id, local_id) → boxes for overlay index
    box_index: dict[tuple[str, int], list[dict]] = {
        (t.camera_id, t.local_id): t.boxes for t in tracklets
    }

    result: list[dict[str, Any]] = []
    for tr in tracks:
        segments = []
        for seg in tr.segments:
            key = (seg["camera_id"], int(seg["local_id"]))
            segments.append(
                {
                    **seg,
                    "boxes": box_index.get(key, []),
                }
            )
        result.append(
            {
                "track_id": int(tr.track_id),
                "t_start": float(tr.t_start),
                "t_end": float(tr.t_end),
                "cameras": list(tr.cameras),
                "segments": segments,
                "close_reason": tr.close_reason,
            }
        )
    return result


def build_overlays(tracks: list[dict[str, Any]]) -> dict[str, dict[str, list[dict]]]:
    """camera_id -> frame_idx(str) -> [{track_id, bbox, score}]."""
    overlays: dict[str, dict[str, list[dict]]] = {}
    for tr in tracks:
        tid = tr["track_id"]
        for seg in tr["segments"]:
            cam = seg["camera_id"]
            cam_map = overlays.setdefault(cam, {})
            for box in seg.get("boxes") or []:
                key = str(int(box["frame_idx"]))
                cam_map.setdefault(key, []).append(
                    {
                        "track_id": tid,
                        "bbox": box["bbox"],
                        "score": box.get("score", 1.0),
                        "timestamp": box.get("timestamp"),
                    }
                )
    return overlays
