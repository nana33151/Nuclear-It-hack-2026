# global_tracker.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict

@dataclass
class Tracklet:
    camera_id: str
    local_id: int
    t_start: float
    t_end: float
    reid: np.ndarray          # L2-normalized
    color_upper: np.ndarray   # e.g. HSV hist or mean RGB
    color_lower: np.ndarray
    boxes: Optional[List] = None  # optional, for UI

@dataclass
class GlobalTrack:
    track_id: int
    t_start: float
    t_end: float
    segments: List[Dict]      # {camera_id, local_id, t_start, t_end}
    cameras: List[str]
    close_reason: str = "gap_exceeded"

class GlobalTracker:
    def __init__(
        self,
        T: float = 8.0,                 # сек. максимальный разрыв без наблюдений
        min_sim: float = 0.55,          # порог сходства (выше = строже)
        w_reid: float = 1.0,
        w_color: float = 0.3,
        w_time: float = 0.1,
        max_transition_time: float = 15.0,  # макс. время перехода между камерами
        topology: Optional[Dict[str, List[str]]] = None,  # camera -> allowed next cameras
        method: str = "greedy",         # "greedy" | "average_linkage"
    ):
        self.T = T
        self.min_sim = min_sim
        self.w_reid = w_reid
        self.w_color = w_color
        self.w_time = w_time
        self.max_transition_time = max_transition_time
        self.topology = topology or {}
        self.method = method
        self.next_id = 1

    def _color_dist(self, c1: np.ndarray, c2: np.ndarray) -> float:
        # простая L2 по гистограммам / средним цветам (можно заменить на chi2)
        return float(np.linalg.norm(c1 - c2))

    def _similarity(self, a: Tracklet, b: Tracklet) -> float:
        """Возвращает similarity ∈ [0, 1]. Чем выше — тем лучше."""
        # 1. ReID (cosine similarity после L2-norm)
        reid_sim = 1.0 - cosine(a.reid, b.reid)
        reid_sim = max(0.0, min(1.0, reid_sim))

        # 2. Цвет одежды
        color_d = 0.5 * (self._color_dist(a.color_upper, b.color_upper) +
                         self._color_dist(a.color_lower, b.color_lower))
        color_sim = np.exp(-color_d)          # ≈1 при близких цветах

        # 3. Временной разрыв (штраф за большой gap)
        gap = abs(a.t_start - b.t_end) if a.t_start > b.t_end else abs(b.t_start - a.t_end)
        if gap > self.max_transition_time:
            return 0.0
        time_sim = np.exp(-gap / self.max_transition_time)

        # 4. Топология (жёсткий фильтр)
        if self.topology:
            allowed = self.topology.get(a.camera_id, [])
            if b.camera_id != a.camera_id and b.camera_id not in allowed:
                return 0.0

        sim = (self.w_reid * reid_sim +
               self.w_color * color_sim +
               self.w_time * time_sim) / (self.w_reid + self.w_color + self.w_time)
        return float(sim)

    def _can_merge(self, a: Tracklet, b: Tracklet) -> bool:
        # Нельзя склеивать пересекающиеся по времени на одной камере
        if a.camera_id == b.camera_id:
            return False
        # gap не больше max_transition_time уже учтён в similarity
        return True

    def _greedy_match(self, tracklets: List[Tracklet]) -> List[List[int]]:
        """Жадный матчинг: сортируем пары по убыванию sim и берём, если оба свободны."""
        n = len(tracklets)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if not self._can_merge(tracklets[i], tracklets[j]):
                    continue
                sim = self._similarity(tracklets[i], tracklets[j])
                if sim >= self.min_sim:
                    pairs.append((sim, i, j))
        pairs.sort(reverse=True)  # лучшие сначала

        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for sim, i, j in pairs:
            pi, pj = find(i), find(j)
            if pi != pj:
                parent[pi] = pj

        clusters = defaultdict(list)
        for i in range(n):
            clusters[find(i)].append(i)
        return list(clusters.values())

    def _average_linkage(self, tracklets: List[Tracklet]) -> List[List[int]]:
        """Hierarchical clustering (average linkage) + порог по расстоянию."""
        n = len(tracklets)
        if n == 0:
            return []
        # Матрица расстояний (1 - sim)
        dist = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if self._can_merge(tracklets[i], tracklets[j]):
                    sim = self._similarity(tracklets[i], tracklets[j])
                    d = 1.0 - sim
                else:
                    d = 1.0
                dist[i, j] = dist[j, i] = d

        # condensed distance matrix
        from scipy.spatial.distance import squareform
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="average")
        # fcluster: t = 1 - min_sim (расстояние)
        labels = fcluster(Z, t=1.0 - self.min_sim, criterion="distance")
        clusters = defaultdict(list)
        for i, lab in enumerate(labels):
            clusters[lab].append(i)
        return list(clusters.values())

    def track(self, tracklets: List[Tracklet]) -> List[GlobalTrack]:
        if not tracklets:
            return []

        # Сортируем по времени начала
        tracklets = sorted(tracklets, key=lambda t: t.t_start)

        if self.method == "greedy":
            clusters = self._greedy_match(tracklets)
        else:
            clusters = self._average_linkage(tracklets)

        global_tracks = []
        for cluster_idxs in clusters:
            segs = [tracklets[i] for i in cluster_idxs]
            segs.sort(key=lambda s: s.t_start)

            # Проверяем, что внутри кластера нет слишком больших gap'ов
            # (можно дополнительно разрезать, если gap > T)
            current_segments = []
            prev_end = None
            for s in segs:
                if prev_end is not None and s.t_start - prev_end > self.T:
                    # слишком большой разрыв → закрываем предыдущий трек
                    if current_segments:
                        global_tracks.append(self._make_track(current_segments))
                    current_segments = []
                current_segments.append(s)
                prev_end = max(prev_end or 0, s.t_end)

            if current_segments:
                global_tracks.append(self._make_track(current_segments))

        # Сортируем по t_start и перенумеровываем ID
        global_tracks.sort(key=lambda t: t.t_start)
        for i, tr in enumerate(global_tracks, 1):
            tr.track_id = i
        return global_tracks

    def _make_track(self, segments: List[Tracklet]) -> GlobalTrack:
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
        return GlobalTrack(
            track_id=self.next_id,
            t_start=t_start,
            t_end=t_end,
            segments=segs,
            cameras=cams,
            close_reason="gap_exceeded",
        )