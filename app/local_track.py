"""Single-camera ByteTrack + ReID (SOLIDER or color fallback) → tracklets."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.detect import FrameDet
from app.paths import load_bytetrack_cls


@dataclass
class LocalTracklet:
    camera_id: str
    local_id: int
    t_start: float
    t_end: float
    reid: np.ndarray
    color_upper: np.ndarray
    color_lower: np.ndarray
    boxes: list[dict[str, Any]] = field(default_factory=list)


def _l2(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec)) or 1.0
    return vec / n


def _color_emb(top: list[float], bottom: list[float], dim: int = 32) -> np.ndarray:
    base = list(top) + list(bottom)
    while len(base) < dim:
        i = len(base)
        base.append(float(np.sin(sum(base) + i) * 0.01))
    return _l2(np.asarray(base[:dim], dtype=np.float64))


def _det_emb(det: FrameDet) -> np.ndarray:
    if det.reid is not None and len(det.reid) > 0:
        return _l2(np.asarray(det.reid, dtype=np.float64))
    return _color_emb(det.color_top, det.color_bottom)


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def run_mot(
    detections: list[FrameDet],
    *,
    track_thresh: float = 0.4,
    track_buffer: int = 30,
    match_thresh: float = 0.8,
    frame_rate: int = 25,
    min_frames: int = 2,
) -> list[LocalTracklet]:
    if not detections:
        return []
    BYTETracker = load_bytetrack_cls()
    by_frame: dict[int, list[FrameDet]] = defaultdict(list)
    for d in detections:
        by_frame[d.frame_idx].append(d)

    tracker = BYTETracker(
        track_thresh=track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        frame_rate=frame_rate,
    )
    histories: dict[int, list[FrameDet]] = defaultdict(list)
    fmin, fmax = min(by_frame), max(by_frame)
    for frame_idx in range(fmin, fmax + 1):
        frame_dets = by_frame.get(frame_idx, [])
        arr = [list(d.bbox) + [d.score] for d in frame_dets]
        online = tracker.update(arr)
        if not frame_dets or not online:
            continue
        used: set[int] = set()
        for tr in online:
            best_i, best_iou = -1, 0.0
            for i, det in enumerate(frame_dets):
                if i in used:
                    continue
                iou = _iou(tr.tlbr, det.bbox)
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_i < 0 or best_iou < 0.1:
                continue
            used.add(best_i)
            histories[tr.track_id].append(frame_dets[best_i])

    tracklets: list[LocalTracklet] = []
    for tid, dets in sorted(histories.items()):
        dets = sorted(dets, key=lambda d: d.frame_idx)
        if len(dets) < min_frames:
            continue
        embs = [
            _det_emb(d)
            for d in dets
            if d.usable_for_reid or d.reid is not None
        ]
        if not embs:
            embs = [_det_emb(d) for d in dets]
        # Prefer SOLIDER vectors when present; do not mix dims with color fallback.
        solider = [e for e in embs if e.shape[0] > 64]
        use = solider if solider else embs
        reid = _l2(np.mean(np.stack(use, axis=0), axis=0))
        color_upper = np.median(np.stack([d.color_top for d in dets]), axis=0)
        color_lower = np.median(np.stack([d.color_bottom for d in dets]), axis=0)
        boxes = [
            {
                "frame_idx": d.frame_idx,
                "timestamp": d.timestamp,
                "bbox": d.bbox,
                "score": d.score,
            }
            for d in dets
        ]
        tracklets.append(
            LocalTracklet(
                camera_id=dets[0].camera_id,
                local_id=int(tid),
                t_start=float(dets[0].timestamp),
                t_end=float(dets[-1].timestamp),
                reid=reid,
                color_upper=np.asarray(color_upper, dtype=np.float64),
                color_lower=np.asarray(color_lower, dtype=np.float64),
                boxes=boxes,
            )
        )
    return tracklets
