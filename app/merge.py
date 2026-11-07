"""Cross-camera merge via person3 GlobalTracker."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.local_track import LocalTracklet
from app.paths import ensure_import_paths, load_global_tracker_cls

from app.projectors_loader import load_projectors        # NEW
import json, os          

def parse_transitions_from_cfg(
    merge_cfg: dict[str, Any] | None,
    *,
    default_max_time: float = 12.0,
) -> dict[tuple[str, str], Any]:
    """YAML transitions + legacy topology → directed transition edges."""
    ensure_import_paths()
    from GlobalTracker import TransitionEdge  # type: ignore

    cfg = merge_cfg or {}
    raw = cfg.get("transitions")
    out: dict[tuple[str, str], TransitionEdge] = {}

    if isinstance(raw, dict) and raw:
        for src, dst_map in raw.items():
            if not isinstance(dst_map, dict):
                continue
            for dst, spec in dst_map.items():
                if isinstance(spec, dict):
                    out[(str(src), str(dst))] = TransitionEdge(
                        min_time=float(spec.get("min_time", 0.0)),
                        max_time=float(spec.get("max_time", default_max_time)),
                        weight=float(spec.get("weight", 1.0)),
                    )
                else:
                    out[(str(src), str(dst))] = TransitionEdge(
                        max_time=float(default_max_time),
                    )
        return out

    topo = cfg.get("topology") or {}
    if isinstance(topo, dict):
        for src, dsts in topo.items():
            for dst in dsts or []:
                out[(str(src), str(dst))] = TransitionEdge(
                    min_time=0.0,
                    max_time=float(default_max_time),
                    weight=1.0,
                )
    return out


def to_global_tracks(
    tracklets: list[LocalTracklet],
    *,
    T: float = 8.0,
    min_sim: float = 0.45,
    max_transition_time: float = 20.0,
    method: str = "lifecycle",
    transitions: dict[tuple[str, str], Any] | None = None,
    merge_cfg: dict[str, Any] | None = None,
    min_hits: int = 2,
    min_confirm_sec: float = 0.4,
    lost_timeout_sec: float | None = None,
    enable_bev: bool = True,                             # NEW
    debug: bool = False,                                 # NEW
    debug_path: str | None = None,                       # NEW
) -> list[dict[str, Any]]:
    GlobalTracker, Tracklet, _ = load_global_tracker_cls()
    if transitions is None and merge_cfg is not None:
        transitions = parse_transitions_from_cfg(
            merge_cfg, default_max_time=max_transition_time
        )

    projectors: dict = {}
    if enable_bev:
        cams = {t.camera_id for t in tracklets}
        projectors = load_projectors(cams)               # {} если камеры не мапятся

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
        transitions=transitions or {},
        method=method,
        min_hits=min_hits,
        min_confirm_sec=min_confirm_sec,
        lost_timeout_sec=lost_timeout_sec,
        projectors=projectors,                           # NEW
        debug=debug,                                     # NEW
    )
    tracks = tracker.track(gt_tracklets)

    if debug and debug_path and getattr(tracker, "decision_log", None):
        try:
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(tracker.decision_log, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[warn] decision_log write failed: {exc}")

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
                "state": getattr(tr, "state", "confident"),
            }
        )
    return result


def build_overlays(tracks: list[dict[str, Any]]) -> dict[str, dict[str, list[dict]]]:
    """camera_id -> frame_idx(str) -> [{track_id, local_id, bbox, score}]."""
    overlays: dict[str, dict[str, list[dict]]] = {}
    for tr in tracks:
        tid = tr["track_id"]
        for seg in tr["segments"]:
            cam = seg["camera_id"]
            local_id = seg["local_id"]
            cam_map = overlays.setdefault(cam, {})
            for box in seg.get("boxes") or []:
                key = str(int(box["frame_idx"]))
                cam_map.setdefault(key, []).append({
                    "track_id": tid,
                    "local_id": local_id,            # NEW
                    "camera_id": cam,                # NEW
                    "bbox": box["bbox"],
                    "score": box.get("score", 1.0),
                    "timestamp": box.get("timestamp"),
                })
    return overlays
