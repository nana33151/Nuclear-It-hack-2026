"""IoU greedy tracker for smoke tests until ByteTrack is wired."""

from __future__ import annotations

from src.contract import Detection
from src.mot.base import RawTrack


class StubMot:
    def __init__(self, iou_thresh: float = 0.3, max_gap: int = 30) -> None:
        self.iou_thresh = iou_thresh
        self.max_gap = max_gap

    def track(self, detections: list[Detection]) -> list[RawTrack]:
        if not detections:
            return []
        cams = {d.camera_id for d in detections}
        if len(cams) != 1:
            raise ValueError(f"StubMot expects one camera, got {cams}")

        by_frame: dict[int, list[Detection]] = {}
        for d in detections:
            by_frame.setdefault(d.frame_idx, []).append(d)

        active: list[dict] = []
        finished: list[RawTrack] = []
        next_id = 1

        for frame_idx in sorted(by_frame):
            dets = by_frame[frame_idx]
            used: set[int] = set()

            # Drop stale tracks
            still = []
            for tr in active:
                if frame_idx - tr["last_frame"] > self.max_gap:
                    finished.append(RawTrack(local_id=tr["local_id"], detections=tr["dets"]))
                else:
                    still.append(tr)
            active = still

            # Greedy IoU match
            pairs: list[tuple[float, int, int]] = []
            for ti, tr in enumerate(active):
                last = tr["dets"][-1]
                for di, det in enumerate(dets):
                    pairs.append((_iou(last.bbox, det.bbox), ti, di))
            pairs.sort(reverse=True)

            matched_t: set[int] = set()
            matched_d: set[int] = set()
            for iou, ti, di in pairs:
                if iou < self.iou_thresh or ti in matched_t or di in matched_d:
                    continue
                active[ti]["dets"].append(dets[di])
                active[ti]["last_frame"] = frame_idx
                matched_t.add(ti)
                matched_d.add(di)
                used.add(di)

            for di, det in enumerate(dets):
                if di in used:
                    continue
                active.append(
                    {
                        "local_id": f"stub_{next_id}",
                        "dets": [det],
                        "last_frame": frame_idx,
                    }
                )
                next_id += 1

        for tr in active:
            finished.append(RawTrack(local_id=tr["local_id"], detections=tr["dets"]))
        return finished


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
