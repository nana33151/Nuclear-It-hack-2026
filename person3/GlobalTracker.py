# global_tracker.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional

import numpy as np
from scipy.spatial.distance import cosine


class TrackState(str, Enum):
    UNCONFIDENT = "unconfident"
    CONFIDENT = "confident"
    LOST = "lost"
    DIED = "died"


@dataclass
class Tracklet:
    camera_id: str
    local_id: int
    t_start: float
    t_end: float
    reid: np.ndarray  # L2-normalized
    color_upper: np.ndarray
    color_lower: np.ndarray
    boxes: Optional[List] = None


@dataclass
class GlobalTrack:
    track_id: int
    t_start: float
    t_end: float
    segments: List[Dict]
    cameras: List[str]
    close_reason: str = "gap_exceeded"
    state: str = "died"


@dataclass
class _LiveTrack:
    state: TrackState
    segments: List[Tracklet] = field(default_factory=list)
    reid: np.ndarray = field(default_factory=lambda: np.zeros(1))
    color_upper: np.ndarray = field(default_factory=lambda: np.zeros(1))
    color_lower: np.ndarray = field(default_factory=lambda: np.zeros(1))
    coverage_end: float = 0.0
    t_start: float = 0.0
    track_id: Optional[int] = None
    close_reason: str = ""

    @property
    def duration_sec(self) -> float:
        return max(0.0, float(self.coverage_end) - float(self.t_start))

    @property
    def n_segments(self) -> int:
        return len(self.segments)


class GlobalTracker:
    def __init__(
        self,
        T: float = 8.0,
        min_sim: float = 0.55,
        w_reid: float = 1.0,
        w_color: float = 0.3,
        w_time: float = 0.1,
        max_transition_time: float = 15.0,
        topology: Optional[Dict[str, List[str]]] = None,
        method: str = "lifecycle",  # lifecycle | legacy_greedy | legacy_average
        min_hits: int = 2,
        min_confirm_sec: float = 0.4,
        lost_priority_margin: float = 0.02,
    ):
        self.T = T
        self.min_sim = min_sim
        self.w_reid = w_reid
        self.w_color = w_color
        self.w_time = w_time
        self.max_transition_time = max_transition_time
        self.topology = topology or {}
        self.method = method
        self.min_hits = max(1, int(min_hits))
        self.min_confirm_sec = float(min_confirm_sec)
        self.lost_priority_margin = float(lost_priority_margin)
        self.next_id = 1

    def _color_dist(self, c1: np.ndarray, c2: np.ndarray) -> float:
        return float(np.linalg.norm(c1 - c2))

    @staticmethod
    def _l2(vec: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(vec)) or 1.0
        return vec / n

    @staticmethod
    def _temporal_gap(a: Tracklet, b: Tracklet) -> float:
        if a.t_end < b.t_start:
            return float(b.t_start - a.t_end)
        if b.t_end < a.t_start:
            return float(a.t_start - b.t_end)
        return 0.0

    def _similarity(self, a: Tracklet, b: Tracklet) -> float:
        reid_sim = 1.0 - cosine(a.reid, b.reid)
        reid_sim = max(0.0, min(1.0, reid_sim))

        color_d = 0.5 * (
            self._color_dist(a.color_upper, b.color_upper)
            + self._color_dist(a.color_lower, b.color_lower)
        )
        color_sim = np.exp(-color_d)

        gap = self._temporal_gap(a, b)
        if gap > self.max_transition_time:
            return 0.0
        time_sim = (
            np.exp(-gap / self.max_transition_time)
            if self.max_transition_time > 0
            else 1.0
        )

        if self.topology:
            allowed = self.topology.get(a.camera_id, [])
            if b.camera_id != a.camera_id and b.camera_id not in allowed:
                return 0.0

        sim = (
            self.w_reid * reid_sim
            + self.w_color * color_sim
            + self.w_time * time_sim
        ) / (self.w_reid + self.w_color + self.w_time)
        return float(sim)

    def _same_cam_overlap(self, a: Tracklet, b: Tracklet) -> bool:
        if a.camera_id != b.camera_id:
            return False
        return not (a.t_end < b.t_start or b.t_end < a.t_start)

    def _can_merge(self, a: Tracklet, b: Tracklet) -> bool:
        if self._same_cam_overlap(a, b):
            return False
        return True

    def track(self, tracklets: List[Tracklet]) -> List[GlobalTrack]:
        if not tracklets:
            return []
        tracklets = sorted(tracklets, key=lambda t: (t.t_start, t.t_end))
        if self.method in ("legacy_greedy", "legacy_average"):
            return self._track_legacy(tracklets)
        return self._track_lifecycle(tracklets)

    def _track_lifecycle(self, tracklets: List[Tracklet]) -> List[GlobalTrack]:
        live: List[_LiveTrack] = []
        closed: List[GlobalTrack] = []

        for seg in tracklets:
            now = float(seg.t_start)
            self._on_tick(live, closed, now)
            match = self._best_match(seg, live)
            if match is not None:
                self._attach_segment(match, seg)
            else:
                tr = self._spawn_unconfident(seg)
                live.append(tr)

        timeline_end = max(float(s.t_end) for s in tracklets)
        self._on_tick(live, closed, timeline_end + self.T + 1.0)
        for tr in list(live):
            if tr.state == TrackState.DIED:
                continue
            if tr.track_id is None:
                tr.state = TrackState.DIED
                tr.close_reason = "never_confirmed"
                continue
            if tr.state == TrackState.LOST:
                tr.close_reason = "gap_exceeded"
            else:
                tr.close_reason = "end_of_window"
            closed.append(self._to_global(tr))

        closed.sort(key=lambda t: t.t_start)
        for i, tr in enumerate(closed, 1):
            tr.track_id = i
        return closed

    def _on_tick(self, live: List[_LiveTrack], closed: List[GlobalTrack], now: float) -> None:
        for tr in live:
            if tr.state == TrackState.DIED:
                continue
            if tr.track_id is None and tr.state == TrackState.UNCONFIDENT:
                continue

            if tr.coverage_end > 0 and now > tr.coverage_end:
                if tr.state == TrackState.CONFIDENT:
                    tr.state = TrackState.LOST

            if tr.state == TrackState.LOST and (now - tr.coverage_end) > self.T:
                tr.state = TrackState.DIED
                tr.close_reason = "gap_exceeded"
                if tr.track_id is not None:
                    closed.append(self._to_global(tr))

    def _track_can_accept(self, tr: _LiveTrack, seg: Tracklet) -> bool:
        for existing in tr.segments:
            if self._same_cam_overlap(existing, seg):
                return False
        if tr.segments and not self._can_merge(tr.segments[-1], seg):
            return False
        return True

    def _prototype(self, tr: _LiveTrack) -> Tracklet:
        last = tr.segments[-1]
        return Tracklet(
            camera_id=last.camera_id,
            local_id=last.local_id,
            t_start=tr.t_start,
            t_end=tr.coverage_end,
            reid=tr.reid,
            color_upper=tr.color_upper,
            color_lower=tr.color_lower,
        )

    def _best_match(self, seg: Tracklet, live: List[_LiveTrack]) -> Optional[_LiveTrack]:
        best: Optional[_LiveTrack] = None
        best_sim = -1.0
        for tr in live:
            if tr.state not in (TrackState.CONFIDENT, TrackState.LOST):
                continue
            if tr.track_id is None:
                continue
            if not self._track_can_accept(tr, seg):
                continue
            sim = self._similarity(self._prototype(tr), seg)
            if sim < self.min_sim:
                continue
            prefer = (
                best is not None
                and tr.state == TrackState.LOST
                and best.state != TrackState.LOST
                and sim >= best_sim - self.lost_priority_margin
            )
            if sim > best_sim + 1e-9 or prefer:
                best_sim = sim
                best = tr
        return best

    def _spawn_unconfident(self, seg: Tracklet) -> _LiveTrack:
        tr = _LiveTrack(
            state=TrackState.UNCONFIDENT,
            segments=[seg],
            reid=self._l2(np.asarray(seg.reid, dtype=np.float64)),
            color_upper=np.asarray(seg.color_upper, dtype=np.float64),
            color_lower=np.asarray(seg.color_lower, dtype=np.float64),
            coverage_end=float(seg.t_end),
            t_start=float(seg.t_start),
        )
        self._try_confirm(tr)
        return tr

    def _attach_segment(self, tr: _LiveTrack, seg: Tracklet) -> None:
        tr.segments.append(seg)
        tr.coverage_end = max(tr.coverage_end, float(seg.t_end))
        n = tr.n_segments
        tr.reid = self._l2(
            (tr.reid * (n - 1) + self._l2(np.asarray(seg.reid, dtype=np.float64))) / n
        )
        tr.color_upper = (tr.color_upper * (n - 1) + seg.color_upper) / n
        tr.color_lower = (tr.color_lower * (n - 1) + seg.color_lower) / n
        if tr.state == TrackState.LOST:
            tr.state = TrackState.CONFIDENT
        self._try_confirm(tr)

    def _try_confirm(self, tr: _LiveTrack) -> None:
        if tr.state != TrackState.UNCONFIDENT:
            return
        if tr.n_segments < self.min_hits and tr.duration_sec < self.min_confirm_sec:
            return
        tr.state = TrackState.CONFIDENT
        if tr.track_id is None:
            tr.track_id = self.next_id
            self.next_id += 1

    def _to_global(self, tr: _LiveTrack) -> GlobalTrack:
        cams = sorted({s.camera_id for s in tr.segments})
        segs = [
            {
                "camera_id": s.camera_id,
                "local_id": s.local_id,
                "t_start": s.t_start,
                "t_end": s.t_end,
            }
            for s in tr.segments
        ]
        return GlobalTrack(
            track_id=int(tr.track_id or 0),
            t_start=float(tr.t_start),
            t_end=float(tr.coverage_end),
            segments=segs,
            cameras=cams,
            close_reason=tr.close_reason or "gap_exceeded",
            state=tr.state.value,
        )

    # --- legacy batch clustering (optional) ---

    def _track_legacy(self, tracklets: List[Tracklet]) -> List[GlobalTrack]:
        from collections import defaultdict
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        if self.method == "legacy_greedy":
            clusters = self._greedy_match(tracklets)
        else:
            clusters = self._average_linkage(tracklets)
        clusters = self._split_overlap_clusters(tracklets, clusters)

        global_tracks = []
        for cluster_idxs in clusters:
            segs = [tracklets[i] for i in cluster_idxs]
            segs.sort(key=lambda s: (s.t_start, -s.t_end))
            current_segments: List[Tracklet] = []
            coverage_end: Optional[float] = None
            for s in segs:
                if coverage_end is not None:
                    gap = max(0.0, float(s.t_start) - float(coverage_end))
                    if gap > self.T:
                        if current_segments:
                            global_tracks.append(self._make_track_legacy(current_segments))
                        current_segments = []
                        coverage_end = None
                current_segments.append(s)
                coverage_end = (
                    float(s.t_end)
                    if coverage_end is None
                    else max(coverage_end, float(s.t_end))
                )
            if current_segments:
                global_tracks.append(self._make_track_legacy(current_segments))

        global_tracks.sort(key=lambda t: t.t_start)
        for i, tr in enumerate(global_tracks, 1):
            tr.track_id = i
        return global_tracks

    def _greedy_match(self, tracklets: List[Tracklet]) -> List[List[int]]:
        from collections import defaultdict

        n = len(tracklets)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if not self._can_merge(tracklets[i], tracklets[j]):
                    continue
                sim = self._similarity(tracklets[i], tracklets[j])
                if sim >= self.min_sim:
                    pairs.append((sim, i, j))
        pairs.sort(reverse=True)

        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for _sim, i, j in pairs:
            pi, pj = find(i), find(j)
            if pi == pj:
                continue
            if self._union_conflicts(tracklets, parent, find, pi, pj):
                continue
            parent[pi] = pj

        clusters = defaultdict(list)
        for i in range(n):
            clusters[find(i)].append(i)
        return list(clusters.values())

    def _average_linkage(self, tracklets: List[Tracklet]) -> List[List[int]]:
        from collections import defaultdict
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        n = len(tracklets)
        if n == 0:
            return []
        dist = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if self._can_merge(tracklets[i], tracklets[j]):
                    sim = self._similarity(tracklets[i], tracklets[j])
                    d = 1.0 - sim
                else:
                    d = 1.0
                dist[i, j] = dist[j, i] = d
        condensed = squareform(dist, checks=False)
        z = linkage(condensed, method="average")
        labels = fcluster(z, t=1.0 - self.min_sim, criterion="distance")
        clusters = defaultdict(list)
        for i, lab in enumerate(labels):
            clusters[lab].append(i)
        return list(clusters.values())

    def _union_conflicts(self, tracklets, parent, find, pi, pj) -> bool:
        for ia, a in enumerate(tracklets):
            if find(ia) != pi:
                continue
            for ib, b in enumerate(tracklets):
                if find(ib) != pj:
                    continue
                if self._same_cam_overlap(a, b):
                    return True
        return False

    def _split_overlap_clusters(
        self, tracklets: List[Tracklet], clusters: List[List[int]]
    ) -> List[List[int]]:
        out: List[List[int]] = []
        for idxs in clusters:
            groups: List[List[int]] = []
            for i in sorted(idxs, key=lambda k: tracklets[k].t_start):
                placed = False
                for g in groups:
                    if not any(
                        self._same_cam_overlap(tracklets[i], tracklets[j]) for j in g
                    ):
                        g.append(i)
                        placed = True
                        break
                if not placed:
                    groups.append([i])
            out.extend(groups)
        return out

    def _make_track_legacy(self, segments: List[Tracklet]) -> GlobalTrack:
        t_start = min(s.t_start for s in segments)
        t_end = max(s.t_end for s in segments)
        cams = sorted(set(s.camera_id for s in segments))
        segs = [
            {
                "camera_id": s.camera_id,
                "local_id": s.local_id,
                "t_start": s.t_start,
                "t_end": s.t_end,
            }
            for s in segments
        ]
        tid = self.next_id
        self.next_id += 1
        return GlobalTrack(
            track_id=tid,
            t_start=t_start,
            t_end=t_end,
            segments=segs,
            cameras=cams,
            close_reason="gap_exceeded",
            state="confident",
        )
