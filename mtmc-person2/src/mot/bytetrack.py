"""ByteTrack backend over Person 1 detection tables."""

from __future__ import annotations

from src.contract import Detection
from src.mot.base import RawTrack
from src.mot.byte_tracker_core import BYTETracker


class ByteTrackMot:
    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        n_init: int = 3,
        match_thresh: float = 0.8,
        frame_rate: int = 25,
        reid_weight: float = 0.45,
        **_: object,
    ) -> None:
        self.track_thresh = float(track_thresh)
        self.track_buffer = int(track_buffer)
        self.match_thresh = float(match_thresh)
        self.frame_rate = int(frame_rate)
        self.n_init = int(n_init)
        self.reid_weight = float(reid_weight)

    def track(self, detections: list[Detection]) -> list[RawTrack]:
        if not detections:
            return []
        cams = {d.camera_id for d in detections}
        if len(cams) != 1:
            raise ValueError(f"ByteTrackMot expects one camera, got {cams}")

        by_frame: dict[int, list[Detection]] = {}
        for d in detections:
            by_frame.setdefault(d.frame_idx, []).append(d)

        tracker = BYTETracker(
            track_thresh=self.track_thresh,
            track_buffer=self.track_buffer,
            match_thresh=self.match_thresh,
            frame_rate=self.frame_rate,
            n_init=self.n_init,
            reid_weight=float(getattr(self, "reid_weight", 0.45)),
        )

        # local_id -> list[Detection]
        histories: dict[int, list[Detection]] = {}
        frame_min, frame_max = min(by_frame), max(by_frame)

        for frame_idx in range(frame_min, frame_max + 1):
            frame_dets = by_frame.get(frame_idx, [])
            arr = [list(d.bbox) + [d.score] for d in frame_dets]
            feats = [list(d.color_top) + list(d.color_bottom) if d.color_top and d.color_bottom else None for d in frame_dets]
            online = tracker.update(arr, feats)
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
                histories.setdefault(tr.track_id, []).append(frame_dets[best_i])

        return [
            RawTrack(local_id=f"bt_{tid}", detections=dets)
            for tid, dets in sorted(histories.items())
            if dets
        ]


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
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0
