"""Minimal ByteTrack (IoU + two-stage assoc + buffer). No cython_bbox / scipy.

Algorithm follows Zhang et al. BYTE; assignment is greedy by cost for fewer deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _STrack:
    track_id: int
    tlbr: list[float]
    score: float
    state: str = "tracked"  # tracked | lost | removed
    frame_id: int = 0
    start_frame: int = 0
    is_activated: bool = False
    time_since_update: int = 0
    hits: int = 0
    feat: list[float] | None = None


@dataclass
class BYTETracker:
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    frame_rate: int = 30
    n_init: int = 3
    reid_weight: float = 0.45
    reid_gate: float = 0.7
    frame_id: int = 0
    _next_id: int = 1
    tracked: list[_STrack] = field(default_factory=list)
    lost: list[_STrack] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.max_time_lost = max(1, int(self.frame_rate / 30.0 * self.track_buffer))
        self.det_thresh = self.track_thresh + 0.1

    def update(
        self,
        dets_xyxy_score: list[list[float]],
        feats: list[list[float] | None] | None = None,
    ) -> list[_STrack]:
        """dets: each [x1,y1,x2,y2,score]. Optional L2 ReID feats aligned with dets."""
        self.frame_id += 1
        scores = [d[4] for d in dets_xyxy_score]
        bboxes = [d[:4] for d in dets_xyxy_score]
        if feats is None or len(feats) != len(dets_xyxy_score):
            feats = [None] * len(dets_xyxy_score)

        high, low = [], []
        for bb, sc, feat in zip(bboxes, scores, feats):
            row = (bb, sc, _l2_list(feat))
            if sc >= self.track_thresh:
                high.append(row)
            elif sc > 0.1:
                low.append(row)

        detections = [_STrack(0, list(bb), sc, feat=feat) for bb, sc, feat in high]
        detections_low = [_STrack(0, list(bb), sc, feat=feat) for bb, sc, feat in low]

        tracked = [t for t in self.tracked if t.state == "tracked"]
        unconfirmed = [t for t in self.tracked if not t.is_activated]
        tracked = [t for t in tracked if t.is_activated]

        strack_pool = tracked + self.lost
        matches, u_tracks, u_dets = _associate(
            strack_pool,
            detections,
            self.match_thresh,
            reid_weight=self.reid_weight,
            reid_gate=self.reid_gate,
        )

        activated, refound, lost_now = [], [], []

        for it, idet in matches:
            track = strack_pool[it]
            det = detections[idet]
            if track.state == "tracked":
                _update_track(track, det, self.frame_id, n_init=self.n_init)
                activated.append(track)
            else:
                _reactivate(track, det, self.frame_id, n_init=self.n_init)
                refound.append(track)

        # Second association with low-score dets
        r_tracked = [strack_pool[i] for i in u_tracks if strack_pool[i].state == "tracked"]
        matches2, u_tracks2, _ = _associate(
            r_tracked,
            detections_low,
            0.5,
            reid_weight=self.reid_weight,
            reid_gate=self.reid_gate,
        )
        for it, idet in matches2:
            track = r_tracked[it]
            det = detections_low[idet]
            _update_track(track, det, self.frame_id, n_init=self.n_init)
            activated.append(track)
        for it in u_tracks2:
            track = r_tracked[it]
            if track.state != "lost":
                track.state = "lost"
                lost_now.append(track)

        # Unconfirmed
        detections_left = [detections[i] for i in u_dets]
        matches_u, u_unconf, u_det_left = _associate(
            unconfirmed,
            detections_left,
            0.7,
            reid_weight=self.reid_weight,
            reid_gate=self.reid_gate,
        )
        for it, idet in matches_u:
            tr = unconfirmed[it]
            _update_track(tr, detections_left[idet], self.frame_id, n_init=self.n_init)
            # Keep in self.tracked until activated; return from update() stays filtered.
            activated.append(tr)
        for it in u_unconf:
            unconfirmed[it].state = "removed"

        # New tracks
        for inew in u_det_left:
            det = detections_left[inew]
            if det.score < self.det_thresh:
                continue
            det.track_id = self._next_id
            self._next_id += 1
            det.state = "tracked"
            det.is_activated = False
            det.frame_id = self.frame_id
            det.start_frame = self.frame_id
            det.time_since_update = 0
            det.hits = 1
            det.feat = _l2_list(det.feat)
            activated.append(det)

        # Age lost tracks
        for track in self.lost:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.state = "removed"

        self.tracked = [t for t in activated + refound if t.state == "tracked"]
        # Keep previously tracked that were not updated this frame → already in lost_now
        self.lost = [t for t in self.lost + lost_now if t.state == "lost"]
        # Deduplicate by track_id preferring tracked
        self.lost = [t for t in self.lost if t.track_id not in {x.track_id for x in self.tracked}]

        for t in self.tracked:
            t.time_since_update = 0
        return [t for t in self.tracked if t.is_activated]


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


def _l2_list(vec: list[float] | None) -> list[float] | None:
    if not vec:
        return None
    n = sum(x * x for x in vec) ** 0.5
    if n <= 1e-12:
        return list(vec)
    return [x / n for x in vec]


def _cos_dist(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None or len(a) != len(b):
        return None
    return max(0.0, 1.0 - sum(x * y for x, y in zip(a, b)))


def _pair_cost(
    track: _STrack,
    det: _STrack,
    *,
    reid_weight: float,
    reid_gate: float,
) -> float:
    iou_cost = 1.0 - _iou(track.tlbr, det.tlbr)
    app = _cos_dist(track.feat, det.feat)
    if app is None or reid_weight <= 0.0:
        return iou_cost
    if app > reid_gate:
        return 1.0
    w = min(0.75, reid_weight + (0.2 if track.state == "lost" else 0.0))
    return (1.0 - w) * iou_cost + w * app


def _ema_feat(old: list[float] | None, new: list[float] | None, alpha: float = 0.9) -> list[float] | None:
    new_n = _l2_list(new)
    if new_n is None:
        return old
    if old is None or len(old) != len(new_n):
        return new_n
    mixed = [alpha * o + (1.0 - alpha) * n for o, n in zip(old, new_n)]
    return _l2_list(mixed)


def _associate(
    tracks: list[_STrack],
    dets: list[_STrack],
    thresh: float,
    *,
    reid_weight: float = 0.0,
    reid_gate: float = 0.7,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))

    pairs: list[tuple[float, int, int]] = []
    for ti, tr in enumerate(tracks):
        for di, det in enumerate(dets):
            cost = _pair_cost(tr, det, reid_weight=reid_weight, reid_gate=reid_gate)
            pairs.append((cost, ti, di))
    pairs.sort()

    matches: list[tuple[int, int]] = []
    used_t: set[int] = set()
    used_d: set[int] = set()
    # Official BYTE uses cost = 1 - IoU; accept when cost <= thresh.
    for cost, ti, di in pairs:
        if ti in used_t or di in used_d or cost > thresh:
            continue
        used_t.add(ti)
        used_d.add(di)
        matches.append((ti, di))

    u_tracks = [i for i in range(len(tracks)) if i not in used_t]
    u_dets = [i for i in range(len(dets)) if i not in used_d]
    return matches, u_tracks, u_dets


def _update_track(track: _STrack, det: _STrack, frame_id: int, *, n_init: int = 3) -> None:
    track.tlbr = list(det.tlbr)
    track.score = det.score
    track.state = "tracked"
    track.hits = int(track.hits) + 1
    track.feat = _ema_feat(track.feat, det.feat)
    if track.hits >= max(1, int(n_init)):
        track.is_activated = True
    track.frame_id = frame_id
    track.time_since_update = 0


def _reactivate(track: _STrack, det: _STrack, frame_id: int, *, n_init: int = 3) -> None:
    _update_track(track, det, frame_id, n_init=n_init)
