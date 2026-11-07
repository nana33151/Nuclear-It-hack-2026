"""Pure data logic for BEV UI tab.

overlays.json + tracks.json + projectors  →  list[BBoxEvent] / list[BEVEvent]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from app.projectors_loader import project_bbox


@dataclass
class BBoxEvent:
    t: float
    cam: str
    frame_idx: int
    local_id: int
    global_id: Optional[int]
    bbox: tuple[float, float, float, float]
    score: float = 0.0


@dataclass
class BEVEvent:
    t: float
    cam: str
    global_id: int
    x: float
    y: float
    local_id: int


def build_local_to_global(tracks_doc: dict) -> dict[tuple[str, int], int]:
    mapping: dict[tuple[str, int], int] = {}
    for tr in tracks_doc.get("tracks", []):
        gid = int(tr["track_id"])
        for seg in tr.get("segments", []):
            mapping[(str(seg["camera_id"]), int(seg["local_id"]))] = gid
    return mapping


def iter_bbox_events(
    overlays_doc: dict,
    tracks_doc: dict,
    keep_unmapped: bool = False,
) -> Iterable[BBoxEvent]:
    mapping = build_local_to_global(tracks_doc)
    overlays = overlays_doc.get("overlays", {}) or {}
    for cam, by_frame in overlays.items():
        for frame_key, items in by_frame.items():
            try:
                frame_idx = int(frame_key)
            except (TypeError, ValueError):
                continue
            for item in items or []:
                local_id = item.get("local_id")
                if local_id is None:
                    local_id = item.get("track_id", -1)
                local_id = int(local_id)
                if local_id < 0:
                    continue
                gid = mapping.get((cam, local_id))
                if gid is None and not keep_unmapped:
                    continue
                bbox = item.get("bbox") or item.get("box")
                if not bbox or len(bbox) != 4:
                    continue
                yield BBoxEvent(
                    t=float(item.get("timestamp", 0.0)),
                    cam=cam,
                    frame_idx=frame_idx,
                    local_id=local_id,
                    global_id=gid,
                    bbox=tuple(float(v) for v in bbox),
                    score=float(item.get("score", 0.0)),
                )


def project_events(
    bbox_events: Iterable[BBoxEvent],
    projectors: dict,
) -> list[BEVEvent]:
    out: list[BEVEvent] = []
    for ev in bbox_events:
        if ev.global_id is None:
            continue
        proj = projectors.get(ev.cam)
        if proj is None:
            continue
        xy = project_bbox(proj, ev.bbox)
        if xy is None:
            continue
        out.append(BEVEvent(
            t=ev.t, cam=ev.cam, global_id=ev.global_id,
            x=xy[0], y=xy[1], local_id=ev.local_id,
        ))
    return out


def footprint_per_camera(
    events: Sequence[BEVEvent],
    max_points_per_cam: int = 400,
) -> dict[str, list[tuple[float, float]]]:
    """Convex hull of every BEV point observed per camera (approximate FOV)."""
    from scipy.spatial import ConvexHull

    by_cam: dict[str, list[tuple[float, float]]] = {}
    for ev in events:
        by_cam.setdefault(ev.cam, []).append((ev.x, ev.y))

    out: dict[str, list[tuple[float, float]]] = {}
    for cam, pts in by_cam.items():
        if len(pts) < 3:
            continue
        if len(pts) > max_points_per_cam:
            step = len(pts) // max_points_per_cam
            pts = pts[::step]
        arr = np.asarray(pts, dtype=float)
        try:
            hull = ConvexHull(arr)
        except Exception:
            continue
        out[cam] = [tuple(arr[i]) for i in hull.vertices]
    return out


def active_at(events: Sequence[BEVEvent], t: float, tol_sec: float) -> list[BEVEvent]:
    """One BEVEvent per (cam, gid) closest to t within tol_sec."""
    best: dict[tuple[str, int], BEVEvent] = {}
    for ev in events:
        if abs(ev.t - t) > tol_sec:
            continue
        key = (ev.cam, ev.global_id)
        cur = best.get(key)
        if cur is None or abs(ev.t - t) < abs(cur.t - t):
            best[key] = ev
    return list(best.values())


def trail_for(
    events: Sequence[BEVEvent],
    global_id: int,
    t: float,
    tail_sec: float,
) -> list[BEVEvent]:
    out = [e for e in events if e.global_id == global_id and t - tail_sec <= e.t <= t]
    out.sort(key=lambda e: e.t)
    return out


def bboxes_for_cam_at(
    overlays_doc: dict,
    cam: str,
    t: float,
    tol_sec: float,
    tracks_doc: dict,
) -> list[dict]:
    """Camera overlay items closest to t, each item gets local_id/global_id resolved."""
    by_frame = (overlays_doc.get("overlays") or {}).get(cam, {})
    if not by_frame:
        return []

    frame_ts: list[tuple[float, str]] = []
    for fk, items in by_frame.items():
        if items and "timestamp" in items[0]:
            frame_ts.append((abs(float(items[0]["timestamp"]) - t), fk))
    if not frame_ts:
        return []
    frame_ts.sort()
    best_dt, best_fk = frame_ts[0]
    if best_dt > tol_sec:
        return []

    best_items = by_frame[best_fk]
    mapping = build_local_to_global(tracks_doc)
    out: list[dict] = []
    for it in best_items:
        # После патча 1 поле local_id есть прямо в overlay.
        local_id = it.get("local_id")
        if local_id is None:
            # Backward-compat для старых overlays.json: без local_id маппинг
            # по track_id верен только когда global == local (совпадает у части треков).
            local_id = it.get("track_id", -1)
        local_id = int(local_id)
        gid = mapping.get((cam, local_id))
        out.append({**it, "local_id": local_id, "global_id": gid})
    return out