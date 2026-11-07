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


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(a, b), 0.0, 1.0))


def _split_by_reid(dets: list[FrameDet], threshold: float) -> list[list[FrameDet]]:
    if threshold <= 0 or len(dets) < 2:
        return [dets]
    pieces: list[list[FrameDet]] = []
    cur = [dets[0]]
    last = _det_emb(dets[0])
    for d in dets[1:]:
        emb = _det_emb(d)
        dist = 1.0 - _cosine_sim(last, emb)
        if dist > threshold and last.shape == emb.shape:
            pieces.append(cur)
            cur = [d]
        else:
            cur.append(d)
        last = emb
    pieces.append(cur)
    return pieces


def _from_dets(dets: list[FrameDet], local_id: int) -> LocalTracklet | None:
    dets = sorted(dets, key=lambda d: d.frame_idx)
    embs = [
        _det_emb(d)
        for d in dets
        if d.usable_for_reid or d.reid is not None
    ]
    if not embs:
        embs = [_det_emb(d) for d in dets]
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
    return LocalTracklet(
        camera_id=dets[0].camera_id,
        local_id=int(local_id),
        t_start=float(dets[0].timestamp),
        t_end=float(dets[-1].timestamp),
        reid=reid,
        color_upper=np.asarray(color_upper, dtype=np.float64),
        color_lower=np.asarray(color_lower, dtype=np.float64),
        boxes=boxes,
    )


def _stitch_same_camera(
    tracklets: list[LocalTracklet],
    *,
    min_sim: float,
    max_gap: float,
) -> list[LocalTracklet]:
    if min_sim <= 0 or not tracklets:
        return tracklets
    by_cam: dict[str, list[LocalTracklet]] = defaultdict(list)
    for t in tracklets:
        by_cam[t.camera_id].append(t)
    out: list[LocalTracklet] = []
    next_id = 1
    for tls in by_cam.values():
        merged: list[LocalTracklet] = []
        for t in sorted(tls, key=lambda x: x.t_start):
            placed = False
            for m in merged:
                gap = float(t.t_start) - float(m.t_end)
                if gap < 0.0 or gap > max_gap:
                    continue
                if _cosine_sim(_l2(m.reid), _l2(t.reid)) < min_sim:
                    continue
                m.t_end = max(m.t_end, t.t_end)
                m.boxes.extend(t.boxes)
                m.boxes.sort(key=lambda b: b["frame_idx"])
                n_m, n_t = max(1, len(m.boxes) - len(t.boxes)), len(t.boxes)
                n = n_m + n_t
                m.reid = _l2((m.reid * n_m + t.reid * n_t) / n)
                m.color_upper = (m.color_upper * n_m + t.color_upper * n_t) / n
                m.color_lower = (m.color_lower * n_m + t.color_lower * n_t) / n
                placed = True
                break
            if not placed:
                merged.append(t)
        for m in merged:
            m.local_id = next_id
            next_id += 1
            out.append(m)
    return out


def run_mot(
    detections: list[FrameDet],
    *,
    track_thresh: float = 0.4,
    track_buffer: int = 30,
    match_thresh: float = 0.8,
    frame_rate: int = 25,
    min_frames: int = 2,
    n_init: int = 3,
    reid_weight: float = 0.45,
    split_reid_thresh: float = 0.45,
    stitch_min_sim: float = 0.6,
    stitch_max_gap: float = 8.0,
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
        n_init=n_init,
        reid_weight=float(reid_weight),
    )
    histories: dict[int, list[FrameDet]] = defaultdict(list)
    idxs = sorted(by_frame.keys())
    step = min((b - a for a, b in zip(idxs, idxs[1:])), default=1)
    step = max(1, int(step))
    # Age lost tracks on empty *sampled* frames; do not pump unsampled video frames.
    for frame_idx in range(idxs[0], idxs[-1] + 1, step):
        frame_dets = by_frame.get(frame_idx, [])
        arr = [list(d.bbox) + [d.score] for d in frame_dets]
        feats = [_det_emb(d).tolist() for d in frame_dets]
        online = tracker.update(arr, feats)
        if not online or not frame_dets:
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
    next_id = 1
    for _tid, dets in sorted(histories.items()):
        dets = sorted(dets, key=lambda d: d.frame_idx)
        for piece in _split_by_reid(dets, split_reid_thresh):
            if len(piece) < min_frames:
                continue
            tl = _from_dets(piece, next_id)
            if tl is None:
                continue
            tracklets.append(tl)
            next_id += 1
    return _stitch_same_camera(
        tracklets, min_sim=stitch_min_sim, max_gap=stitch_max_gap
    )
