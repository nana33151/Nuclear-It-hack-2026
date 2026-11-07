# mtmc-person2/src/geometry/projectors.py
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:  # YAML is optional; JSON works out of the box
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

PathLike = Union[str, Path]


class Projector:
    """Projects a pixel point from a camera image onto the floor plan (meters).

    Coordinates of the plan: X right, Y down.
    Each camera uses its own homography (image pixels -> plan meters).
    """

    # Homographies: image pixels -> plan meters
    HOMOGRAPHIES: Dict[str, np.ndarray] = {
        "cam1": np.array([
            [
                -0.004711056098462246,
                0.0038183223209354807,
                7.630554964186124
            ],
            [
                -0.0020339556394392663,
                -0.004232885921651842,
                10.287618879144562
            ],
            [
                -0.00023425111917469265,
                0.000591412460959166,
                1.0
            ]
        ], dtype=np.float64),

        "cam2": np.array([
            [
                0.0075397998705991395,
                0.008243862605486188,
                -8.748034532207841
            ],
            [
                0.003683968307514554,
                0.025147539545031107,
                -24.566278420155182
            ],
            [
                -0.0008970156960633939,
                0.003561428017205104,
                1.0
            ]
        ], dtype=np.float64),

        "cam3": np.array([
            [
                -0.17409943369842454,
                0.6262847755530632,
                206.94338721320236
            ],
            [
                0.0067954450224434314,
                0.032053262535120086,
                332.8828915567185
            ],
            [
                0.0035499070456282288,
                0.0646706993158012,
                1.0
            ]
        ], dtype=np.float64),

        "cam4": np.array([
            [
                -0.0007270654802091358,
                0.02481178397467071,
                -6.5856976106718825
            ],
            [
                -0.008023508584408416,
                0.021819392236994594,
                24.359106409849414
            ],
            [
                -7.692405262326722e-05,
                0.0015480554279222187,
                0.9999999999999999
            ]
        ], dtype=np.float64),

        "cam5": np.array([
            [
                -0.05871224026600864,
                -0.1458248584812886,
                67.22462791055347
            ],
            [
                -0.03733320799392591,
                -0.4759570382589316,
                91.30134748772583
            ],
            [
                -0.001413668270784155,
                -0.01580104783633913,
                1.0
            ]
        ], dtype=np.float64),
    }

    # Lazy per-camera cache (used by trackers via Projector.get(cam)).
    _cache: Dict[str, "Projector"] = {}

    def __init__(
        self,
        camera_id: str,
        homographies: Optional[Mapping[str, np.ndarray]] = None,
    ):
        """Create a projector for one camera.

        Args:
            camera_id: camera key, e.g. "cam1".
            homographies: optional mapping camera_id -> 3x3 matrix that overrides
                or extends the built-in HOMOGRAPHIES (e.g. loaded from config).
        """
        known = set(self.HOMOGRAPHIES)
        if homographies:
            known |= set(homographies)
        if camera_id not in known:
            raise ValueError(f"Unknown camera_id: {camera_id}")
        self.camera_id = camera_id
        if homographies:
            merged: Dict[str, np.ndarray] = dict(self.HOMOGRAPHIES)
            merged.update(
                {str(k): np.asarray(v, dtype=np.float64) for k, v in homographies.items()}
            )
            self.H = merged[camera_id]
        else:
            self.H = self.HOMOGRAPHIES[camera_id]
        self.H_inv = np.linalg.inv(self.H)

    # ------------------------------------------------------------------
    # factories
    # ------------------------------------------------------------------

    @classmethod
    def get(cls, camera_id: str) -> "Projector":
        """Cached per-camera projector (one instance per camera per process)."""
        proj = cls._cache.get(camera_id)
        if proj is None:
            proj = cls(camera_id)
            cls._cache[camera_id] = proj
        return proj

    @classmethod
    def from_config(cls, path: PathLike) -> Dict[str, "Projector"]:
        """Load per-camera homographies from a JSON or YAML file.

        File format: {camera_id: 3x3 matrix}, e.g.

            cam1:
              - [-0.009409, 0.006394, 7.459238]
              - [-0.003867, -0.008060, 9.328105]
              - [-0.000460, 0.000912, 1.0]

        Returns a dict camera_id -> Projector ready to be passed to GlobalTracker.
        """
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            if yaml is None:
                raise ImportError("PyYAML is required to read YAML configs")
            raw = yaml.safe_load(text)
        else:
            raw = json.loads(text)
        if not isinstance(raw, Mapping):
            raise ValueError(f"Homography config must be a mapping: {path}")
        homographies = {
            str(cam): np.asarray(mat, dtype=np.float64) for cam, mat in raw.items()
        }
        return {cam: cls(cam, homographies=homographies) for cam in homographies}

    # ------------------------------------------------------------------
    # projections
    # ------------------------------------------------------------------

    def image_to_plan(self, point: Sequence[float]) -> Tuple[float, float]:
        """Convert a pixel point (x, y) to floor-plan coordinates (meters).

        Args:
            point: (x, y) in image pixels (origin = top-left).

        Returns:
            (X, Y) in plan meters (X right, Y down).
        """
        x, y = float(point[0]), float(point[1])
        pt = np.array([x, y, 1.0])
        r = self.H @ pt
        w = float(r[2])
        if abs(w) < 1e-12:
            raise ValueError(f"Point {point} is on the horizon line (w~0)")
        return float(r[0] / w), float(r[1] / w)

    def plan_to_image(self, plan_point: Sequence[float]) -> Tuple[float, float]:
        """Convert floor-plan meters back to image pixels (inverse projection)."""
        X, Y = float(plan_point[0]), float(plan_point[1])
        pt = np.array([X, Y, 1.0])
        r = self.H_inv @ pt
        w = float(r[2])
        if abs(w) < 1e-12:
            raise ValueError(f"Plan point {plan_point} has no image preimage (w~0)")
        return float(r[0] / w), float(r[1] / w)

    def bbox_bottom_center_to_plan(self, bbox: Sequence[float]) -> Tuple[float, float]:
        """Convenience: take [x1, y1, x2, y2] and project the bottom-center."""
        x1, y1, x2, y2 = bbox
        cx = (float(x1) + float(x2)) / 2.0
        cy = float(y2)
        return self.image_to_plan((cx, cy))

    def bbox_to_plan(self, bbox: Sequence[float]) -> Tuple[float, float]:
        """Short alias for bbox_bottom_center_to_plan."""
        return self.bbox_bottom_center_to_plan(bbox)

    @staticmethod
    def plan_distance(p1: Sequence[float], p2: Sequence[float]) -> float:
        """Euclidean distance (meters) between two plan points."""
        return float(math.hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1])))


# ----------------------------------------------------------------------
# Detection-level adapter (used by mirror filter and any FrameDet consumer)
# ----------------------------------------------------------------------

def project_detection(
    projector: "Projector",
    det,
) -> Optional[Tuple[float, float]]:
    """Adapter: detection (dict or FrameDet-like object) -> plan (X, Y).

    Uses the bbox bottom-center as the floor contact point. Returns None on
    missing bbox / projection error so callers can default to "keep".
    """
    if isinstance(det, dict):
        bbox = det.get("bbox")
    else:
        bbox = getattr(det, "bbox", None)
    if bbox is None:
        return None
    try:
        return projector.bbox_bottom_center_to_plan(bbox)
    except Exception:
        return None


__all__ = ["Projector", "project_detection"]