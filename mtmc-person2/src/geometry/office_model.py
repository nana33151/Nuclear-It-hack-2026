"""Office floor plan model, cameras and mirror reflection filter.

Coordinates: X right, Y down (same as user's plan).
Units: meters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    ):
        self.polygon = np.array(polygon or self.DEFAULT_POLYGON, dtype=float)
        self.mirrors = [
            Segment(np.array(s, dtype=float), np.array(e, dtype=float))
            for s, e in (mirrors or [self.DEFAULT_MIRROR])
        ]
        self.cameras = {c.camera_id: c for c in (cameras or self.DEFAULT_CAMERAS)}
        self.mirror_margin = mirror_margin  # meters of forbidden strip around mirror

    def camera_sees_mirror(self, camera_id: str) -> bool:
        cam = self.cameras.get(camera_id)
        return bool(cam and cam.sees_mirror)

    def is_on_mirror(self, point_xy: Sequence[float], margin: float | None = None) -> bool:
        """Return True if the projected floor point falls on/near any mirror."""
        margin = self.mirror_margin if margin is None else margin
        p = np.asarray(point_xy, dtype=float)
        for seg in self.mirrors:
            if seg.contains_point(p, tol=margin):
                return True
            v = seg.end - seg.start
            length = np.linalg.norm(v)
            if length < 1e-6:
                continue
            t = np.clip(np.dot(p - seg.start, v) / (length ** 2), 0.0, 1.0)
            closest = seg.start + t * v
            dist = float(np.linalg.norm(p - closest))
            if dist <= margin:
                return True
        return False
