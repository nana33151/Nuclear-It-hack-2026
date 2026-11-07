"""Office floor plan model, cameras and mirror reflection filter.

Coordinates: X right, Y down (same as user's plan).
Units: meters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class Segment:
    """2D line segment."""
    start: np.ndarray  # shape (2,)
    end: np.ndarray    # shape (2,)

    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    def contains_point(self, point: np.ndarray, tol: float = 0.05) -> bool:
        """Check if point lies on the segment (with tolerance)."""
        p = np.asarray(point, dtype=float)
        v = self.end - self.start
        w = p - self.start
        c1 = np.dot(w, v)
        if c1 <= 0:
            return float(np.linalg.norm(p - self.start)) <= tol
        c2 = np.dot(v, v)
        if c2 <= c1:
            return float(np.linalg.norm(p - self.end)) <= tol
        b = c1 / c2
        pb = self.start + b * v
        return float(np.linalg.norm(p - pb)) <= tol


@dataclass
class Camera:
    camera_id: str
    position: tuple[float, float]
    sees_mirror: bool = False


class OfficeModel:
    """Non-convex office polygon + mirrors + cameras."""

    # User's points (X right, Y down)
    DEFAULT_POLYGON = [
        (0.0, 0.0),
        (6.06, 0.0),
        (6.06, 1.07),
        (10.86, 1.07),
        (10.86, 28.63),
        (8.6, 28.63),
        (8.6, 16.43),
        (0.2, 16.43),
        (0.2, 14.63),
        (8.6, 14.63),
        (8.6, 5.03),
        (0.0, 5.03),
        (0.0, 0.0),
    ]

    DEFAULT_MIRROR = ((4.46, 0.0), (5.26, 0.0))  # 0.8 m on top wall

    DEFAULT_CAMERAS = [
        Camera("cam1", (4.86, 0.0), sees_mirror=False),
        Camera("cam2", (0.0, 0.0), sees_mirror=True),   # only this one sees the mirror
        Camera("cam3", (10.0, 1.07), sees_mirror=False),
        Camera("cam4", (10.86, 15.63), sees_mirror=False),
        Camera("cam5", (9.3, 28.63), sees_mirror=False),
    ]

    def __init__(
        self,
        polygon: Sequence[tuple[float, float]] | None = None,
        mirrors: Sequence[tuple[tuple[float, float], tuple[float, float]]] | None = None,
        cameras: Sequence[Camera] | None = None,
        mirror_margin: float = 0.25,
        mirror_back_depth: float = 8.0,
    ):
        self.polygon = np.array(polygon or self.DEFAULT_POLYGON, dtype=float)
        self.mirrors = [
            Segment(np.array(s, dtype=float), np.array(e, dtype=float))
            for s, e in (mirrors or [self.DEFAULT_MIRROR])
        ]
        self.cameras = {c.camera_id: c for c in (cameras or self.DEFAULT_CAMERAS)}
        # Safety band behind the mirror plane where we DON'T drop (homography
        # jitter near the wall could otherwise eat real people standing at Y≈0).
        self.mirror_margin = mirror_margin
        # How deep behind the mirror plane a projection may still count as a
        # reflection. Physical reflections of someone Y meters in front of the
        # mirror land ≈ Y meters behind; 8 m covers the whole room.
        self.mirror_back_depth = mirror_back_depth

    def camera_sees_mirror(self, camera_id: str) -> bool:
        cam = self.cameras.get(camera_id)
        return bool(cam and cam.sees_mirror)

    # ------------------------------------------------------------------
    # geometry helpers
    # ------------------------------------------------------------------

    def _point_in_polygon(self, p: Sequence[float]) -> bool:
        """Ray-casting test (W. R. Franklin). Works with numpy 2D array or list."""
        x = float(p[0])
        y = float(p[1])
        poly = self.polygon
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi = float(poly[i][0])
            yi = float(poly[i][1])
            xj = float(poly[j][0])
            yj = float(poly[j][1])
            if ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / (yj - yi) + xi
            ):
                inside = not inside
            j = i
        return inside

    def is_inside_room(self, point_xy: Sequence[float]) -> bool:
        return self._point_in_polygon(point_xy)

    def is_on_mirror(
        self,
        point_xy: Sequence[float],
        margin: float | None = None,
        behind_depth: float | None = None,
    ) -> bool:
        """Return True if the projected floor point is a mirror reflection.

        The correct test is *which side of the mirror plane* the projection
        falls on, not Euclidean distance to the mirror segment:

          * inside the office polygon           -> real person, KEEP
          * within ``margin`` behind the plane   -> homography jitter, KEEP
                                                   (do not eat real people
                                                    standing right at the wall)
          * deeper than ``margin`` behind plane
            but not further than ``behind_depth``-> reflection, DROP

        Rationale: for a person Y meters in front of the mirror, the reflected
        foot point projects ≈ Y meters *behind* the mirror plane. The old
        distance-to-segment heuristic caught only the strip ±margin, so ghosts
        several meters behind the wall slipped through untouched.
        """
        margin = self.mirror_margin if margin is None else margin
        behind = self.mirror_back_depth if behind_depth is None else behind_depth
        p = np.asarray(point_xy, dtype=float)

        # Real person standing inside the room -> never filtered.
        if self._point_in_polygon(p):
            return False

        centroid = self.polygon.mean(axis=0)

        for seg in self.mirrors:
            v = seg.end - seg.start
            length = float(np.linalg.norm(v))
            if length < 1e-6:
                continue

            u = v / length                          # unit tangent
            normal = np.array([-u[1], u[0]])         # rotate tangent +90°
            # Orient the normal toward the room interior.
            if float(np.dot(centroid - seg.start, normal)) < 0.0:
                normal = -normal

            # Lateral coordinate along the mirror, normalised to [0, 1].
            t = float(np.dot(p - seg.start, u) / length)
            if not (-0.05 <= t <= 1.05):
                # Outside the mirror's lateral span -> not this mirror's ghost.
                continue

            # Signed distance from the mirror plane. Positive = room side.
            signed_perp = float(np.dot(p - seg.start, normal))

            # Behind the plane, deep enough that jitter cannot explain it,
            # but not absurdly deep (avoid catching projections of far-away
            # parts of the office from mis-calibrated homographies).
            if -behind <= signed_perp < -margin:
                return True

        return False