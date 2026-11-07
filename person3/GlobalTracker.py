# global_tracker.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import groupby
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine

# BEV support. The Projector class lives in mtmc-person2/src/geometry/projectors.py.
# If that package is importable (e.g. mtmc-person2/src is on sys.path), the type
# is used directly; otherwise GlobalTracker still works, just without BEV terms.
try:  # pragma: no cover - import path depends on the deployment layout
    from geometry.projectors import Projector  # type: ignore
except Exception:  # pragma: no cover
    try:
        from mtmc_person2.src.geometry.projectors import Projector  # type: ignore
    except Exception:
        Projector = Any  # type: ignore


@dataclass(frozen=True)
class TransitionEdge:
    min_time: float = 0.0
    max_time: float = 12.0
    weight: float = 1.0


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
    # Optional pre-computed BEV position (plan meters). If None and a Projector
    # is registered for camera_id, the position is derived from boxes[-1]
    # (bottom-center) at tracking time.
    plan_pos: Optional[Tuple[float, float]] = None


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
    # Last known BEV position (plan meters) and the camera it was seen by.
    plan_pos: Optional[Tuple[float, float]] = None
    plan_cam: str = ""

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
        transitions: Optional[Dict[Tuple[str, str], TransitionEdge]] = None,
        topology: Optional[Dict[str, List[str]]] = None,
        method: str = "lifecycle",  # lifecycle | legacy_greedy | legacy_average
        min_hits: int = 2,
        min_confirm_sec: float = 0.4,
        lost_priority_margin: float = 0.02,
        unique_margin: float = 0.05,
        # --- BEV (bird's-eye view) options ---
        projectors: Optional[Dict[str, "Projector"]] = None,
        w_bev: float = 0.4,                # weight of the spatial (BEV) term
        bev_typical_speed: float = 1.2,    # m/s, used to scale the distance similarity
        bev_max_speed: float = 3.0,        # m/s, hard reject above this speed
        bev_sigma_scale: float = 2.0,      # sigma = scale * typical_speed * travel_time
        bev_min_travel: float = 0.25,      # s, floor for the travel-time window
        debug: bool = False,   
        lost_timeout_sec: Optional[float] = None,            # collect per-decision reasons
    ):
        self.T = T
        self.min_sim = min_sim
        self.w_reid = w_reid
        self.w_color = w_color
        self.w_time = w_time
        self.w_bev = float(w_bev)
        self.bev_typical_speed = float(bev_typical_speed)
        self.bev_max_speed = float(bev_max_speed)
        self.bev_sigma_scale = float(bev_sigma_scale)
        self.bev_min_travel = float(bev_min_travel)
        self.debug = bool(debug)
        self.decision_log: List[Dict] = []
        # camera_id -> Projector; empty dict disables BEV terms (old behavior).
        self.projectors: Dict[str, "Projector"] = dict(projectors or {})

        self.max_transition_time = max_transition_time
        self.transitions: Dict[Tuple[str, str], TransitionEdge] = dict(
            transitions or {}
        )
        if not self.transitions and topology:
            for src, dsts in topology.items():
                for dst in dsts:
                    self.transitions[(src, dst)] = TransitionEdge(
                        min_time=0.0,
                        max_time=float(max_transition_time),
                        weight=1.0,
                    )
        self.method = method
        self.min_hits = max(1, int(min_hits))
        self.min_confirm_sec = float(min_confirm_sec)
        self.lost_priority_margin = float(lost_priority_margin)
        self.unique_margin = float(unique_margin)
        self.lost_timeout_sec = (
            None if lost_timeout_sec is None else float(lost_timeout_sec)
        )
        self.next_id = 1

    # ------------------------------------------------------------------
    # BEV helpers
    # ------------------------------------------------------------------

    def _seg_plan_pos(self, seg: Tracklet) -> Optional[Tuple[float, float]]:
        """BEV position of a tracklet: explicit plan_pos or from its last bbox."""
        if seg.plan_pos is not None:
            return (float(seg.plan_pos[0]), float(seg.plan_pos[1]))
        proj = self.projectors.get(seg.camera_id)
        if proj is None or not seg.boxes:
            return None

        last = seg.boxes[-1]
        # app/local_track.py stores boxes as dicts; person2 contract uses lists.
        if isinstance(last, dict):
            last = last.get("bbox")
        if not last or len(last) < 4:
            return None

        x, y = proj.bbox_bottom_center_to_plan(last)
        return (float(x), float(y))

    def _bev_term(
        self,
        pa: Tuple[float, float],
        pb: Tuple[float, float],
        gap: float,
        edge: Optional[TransitionEdge],
    ) -> float:
        """Spatial similarity for a cross-camera move, plus a feasibility gate.

        Returns a value in [0, 1]; 0.0 means physically impossible
        (required speed above bev_max_speed).
        """
        dist = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))
        blind = float(edge.min_time) if edge is not None else 0.0
        travel = max(float(gap), blind, self.bev_min_travel)
        if dist > self.bev_max_speed * travel:
            return 0.0  # cannot cover dist metres in travel seconds
        sigma = self.bev_sigma_scale * self.bev_typical_speed * travel
        return float(np.exp(-dist / max(sigma, 1e-6)))

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
        cross_cam = a.camera_id != b.camera_id
        edge = None

        if cross_cam and self.transitions:
            edge = self.transitions.get((a.camera_id, b.camera_id))
            if edge is None:
                return 0.0
            # min_time>0 ⇒ cameras do not share a person at once (travel / blind zone).
            if gap > float(edge.max_time) or gap < float(edge.min_time):
                return 0.0
            max_t = max(float(edge.max_time), 1e-6)
            time_sim = np.exp(-gap / max_t)
        else:
            if gap > self.max_transition_time:
                return 0.0
            time_sim = (
                np.exp(-gap / self.max_transition_time)
                if self.max_transition_time > 0
                else 1.0
            )

        num = (
            self.w_reid * reid_sim
            + self.w_color * color_sim
            + self.w_time * time_sim
        )
        den = self.w_reid + self.w_color + self.w_time

        # --- BEV term: only meaningful for cross-camera moves where both
        # positions are known (either precomputed or via the projectors).
        if cross_cam and self.projectors:
            pa = getattr(a, "plan_pos", None)
            pb = getattr(b, "plan_pos", None)
            if pa is not None and pb is not None:
                bev_sim = self._bev_term(pa, pb, gap, edge)
                if bev_sim <= 0.0:
                    return 0.0  # spatially impossible -> reject the association
                num += self.w_bev * bev_sim
                den += self.w_bev

        appearance = num / den
        # Topology weight only ranks Hungarian cost; do not scale the min_sim gate.
        return float(max(0.0, min(1.0, appearance)))

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

    def _max_lost_timeout(self) -> float:
        edge_caps = [float(e.max_time) for e in self.transitions.values()]
        return max(
            [float(self.T), float(self.max_transition_time), *edge_caps],
            default=float(self.T),
        )

    def _lost_timeout(self, tr: _LiveTrack) -> float:
        if self.lost_timeout_sec is not None:
            return max(0.0, self.lost_timeout_sec)
        last_cam = tr.segments[-1].camera_id if tr.segments else ""
        outgoing = [
            float(e.max_time)
            for (src, _dst), e in self.transitions.items()
            if src == last_cam
        ]
        cross = max(outgoing) if outgoing else float(self.max_transition_time)
        return max(float(self.T), cross)

    def _track_lifecycle(self, tracklets: List[Tracklet]) -> List[GlobalTrack]:
        live: List[_LiveTrack] = []
        closed: List[GlobalTrack] = []

        for _t0, group in groupby(tracklets, key=lambda s: float(s.t_start)):
            segs = list(group)
            now = float(segs[0].t_start)
            self._on_tick(live, closed, now)
            self._assign_segments(segs, live)

        timeline_end = max(float(s.t_end) for s in tracklets)
        self._on_tick(live, closed, timeline_end + self._max_lost_timeout() + 1.0)
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
            if tr.state == TrackState.UNCONFIDENT:
                if tr.coverage_end > 0 and (now - tr.coverage_end) > self.T:
                    tr.state = TrackState.DIED
                    tr.close_reason = "never_confirmed"
                continue

            if tr.coverage_end > 0 and now > tr.coverage_end:
                if tr.state == TrackState.CONFIDENT:
                    tr.state = TrackState.LOST

            if tr.state == TrackState.LOST and (now - tr.coverage_end) > self._lost_timeout(tr):
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

    def _prototype(self, tr: _LiveTrack, camera_id: Optional[str] = None) -> Tracklet:
        last = tr.segments[-1]
        return Tracklet(
            camera_id=camera_id or last.camera_id,
            local_id=last.local_id,
            t_start=tr.t_start,
            t_end=tr.coverage_end,
            reid=tr.reid,
            color_upper=tr.color_upper,
            color_lower=tr.color_lower,
            plan_pos=tr.plan_pos,
        )

    def _match_score_explained(
        self, tr: _LiveTrack, seg: Tracklet
    ) -> Tuple[Optional[float], str]:
        """Score + human-readable reason. None means 'association rejected'.

        Decision chain for a cross-camera (handover) candidate:
          1. state / geometry gates (overlap, topology edge, min_time rule);
          2. spatial feasibility (BEV): reject if the plan distance between the
             track's last position and the new segment's entry position cannot
             be covered within the temporal gap at bev_max_speed;
          3. appearance (ReID + color + time [+ BEV distance similarity]) must
             pass the min_sim gate.
        """
        if tr.state == TrackState.DIED:
            return None, "track_dead"
        if tr.state not in (
            TrackState.UNCONFIDENT,
            TrackState.CONFIDENT,
            TrackState.LOST,
        ):
            return None, "bad_state"
        if not self._track_can_accept(tr, seg):
            return None, "same_cam_overlap"

        last_cam = tr.segments[-1].camera_id if tr.segments else ""
        cross_cam = bool(last_cam) and last_cam != seg.camera_id
        edge = None
        if cross_cam and self.transitions:
            edge = self.transitions.get((last_cam, seg.camera_id))
            if edge is None:
                return None, "no_topology_edge"
            if float(edge.min_time) > 0.0 and tr.state != TrackState.LOST:
                # Cameras do not share a person at once: only a LOST track may
                # reappear in a blind-zone camera.
                return None, "edge_min_time_requires_lost"

        # Explicit BEV feasibility gate (also enforced inside _similarity).
        if cross_cam and self.projectors and tr.plan_pos is not None:
            seg_pos = self._seg_plan_pos(seg)
            if seg_pos is not None:
                gap = max(0.0, float(seg.t_start) - tr.coverage_end)
                if self._bev_term(tr.plan_pos, seg_pos, gap, edge) <= 0.0:
                    return None, "bev_infeasible"

        best = -1.0
        seen = set()
        for s in reversed(tr.segments):
            if s.camera_id in seen:
                continue
            seen.add(s.camera_id)
            best = max(best, self._similarity(self._prototype(tr, s.camera_id), seg))
            if len(seen) >= 4:
                break
        if best < self.min_sim:
            return None, "below_min_sim"
        return float(best), "ok"

    def _match_score(self, tr: _LiveTrack, seg: Tracklet) -> Optional[float]:
        score, reason = self._match_score_explained(tr, seg)
        if self.debug:
            self.decision_log.append(
                {
                    "seg": (seg.camera_id, seg.local_id, float(seg.t_start)),
                    "track_state": tr.state.value,
                    "score": score,
                    "reason": reason,
                }
            )
        return score

    def _assign_segments(self, segs: List[Tracklet], live: List[_LiveTrack]) -> None:
        if not segs:
            return
        if not live:
            for seg in segs:
                live.append(self._spawn_unconfident(seg))
            return

        n, m = len(segs), len(live)
        cost = np.full((n, m), 1e3, dtype=np.float64)
        ok = np.zeros((n, m), dtype=bool)
        for i, seg in enumerate(segs):
            for j, tr in enumerate(live):
                sim = self._match_score(tr, seg)
                if sim is None:
                    continue
                c = 1.0 - sim
                src = tr.segments[-1].camera_id if tr.segments else ""
                if src != seg.camera_id and self.transitions:
                    edge = self.transitions.get((src, seg.camera_id))
                    w = float(edge.weight) if edge is not None else 1.0
                    c = c / max(w, 0.05)
                if tr.state == TrackState.LOST:
                    c = max(0.0, c - self.lost_priority_margin)
                cost[i, j] = c
                ok[i, j] = True

        for i in range(n):
            ranked = sorted(
                ((1.0 - float(cost[i, j]), j) for j in range(m) if ok[i, j]),
                reverse=True,
            )
            if len(ranked) >= 2 and (ranked[0][0] - ranked[1][0]) < self.unique_margin:
                for j in range(m):
                    ok[i, j] = False
                    cost[i, j] = 1e3

        rows, cols = linear_sum_assignment(cost)
        taken = set()
        for i, j in zip(rows, cols):
            if not ok[i, j]:
                continue
            self._attach_segment(live[j], segs[i])
            taken.add(int(i))
        for i, seg in enumerate(segs):
            if i not in taken:
                live.append(self._spawn_unconfident(seg))

    def _best_match(self, seg: Tracklet, live: List[_LiveTrack]) -> Optional[_LiveTrack]:
        best: Optional[_LiveTrack] = None
        best_sim = -1.0
        for tr in live:
            sim = self._match_score(tr, seg)
            if sim is None:
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
            plan_pos=self._seg_plan_pos(seg),
            plan_cam=seg.camera_id,
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

        # Update the BEV anchor: light smoothing when the same camera keeps
        # reporting (reduces homography jitter), direct takeover on handover.
        pos = self._seg_plan_pos(seg)
        if pos is not None:
            if tr.plan_pos is not None and tr.plan_cam == seg.camera_id:
                tr.plan_pos = (
                    0.5 * tr.plan_pos[0] + 0.5 * pos[0],
                    0.5 * tr.plan_pos[1] + 0.5 * pos[1],
                )
            else:
                tr.plan_pos = pos
            tr.plan_cam = seg.camera_id

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
                # BEV position of the segment, if it could be resolved:
                **(
                    {"plan_x": float(p[0]), "plan_y": float(p[1])}
                    if (p := self._seg_plan_pos(s)) is not None
                    else {}
                ),
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
                **(
                    {"plan_x": float(p[0]), "plan_y": float(p[1])}
                    if (p := self._seg_plan_pos(s)) is not None
                    else {}
                ),
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
