"""Camera-to-floor projectors (class-based).

Each camera has its own homography.
The Projector class converts an image point (pixels) into a floor-plan point (meters).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class Projector:
    """Projects a pixel point from a camera image onto the floor plan (meters).

    Coordinates of the plan: X right, Y down.
    """

    # Homographies: image pixels -> plan meters
    HOMOGRAPHIES: dict[str, np.ndarray] = {
        "cam1": np.array([
            [-0.009409,  0.006394,  7.459238],
            [-0.003867, -0.008060,  9.328105],
            [-0.000460,  0.000912,  1.0],
        ], dtype=np.float64),

        "cam2": np.array([
            [0.001251, -0.019190,  5.919907],
            [0.000023,  0.000176, -0.073698],
            [0.000143, -0.003069,  1.0],
        ], dtype=np.float64),

        "cam3": np.array([
            [-0.019402,  0.024947,  8.600000],
            [-0.012324,  0.008965,  8.779557],
            [-0.002256,  0.002901,  1.0],
        ], dtype=np.float64),

        "cam4": np.array([
            [-0.005143,  0.069808, -6.592956],
            [-0.020875,  0.074192, 25.712872],
            [-0.000237,  0.005062,  1.0],
        ], dtype=np.float64),

        "cam5": np.array([
            [0.011874, -0.039955,  4.845265],
            [0.005065, -0.048761, 20.647310],
            [0.000373, -0.004324,  1.0],
        ], dtype=np.float64),
    }

    def __init__(self, camera_id: str):
        if camera_id not in self.HOMOGRAPHIES:
            raise ValueError(f"Unknown camera_id: {camera_id}")
        self.camera_id = camera_id
        self.H = self.HOMOGRAPHIES[camera_id]

    def image_to_plan(self, point: Sequence[float]) -> tuple[float, float]:
        """Convert a pixel point (x, y) to floor-plan coordinates (meters).

        Args:
            point: (x, y) in image pixels (origin = top-left).

        Returns:
            (X, Y) in plan meters (X right, Y down).
        """
        x, y = float(point[0]), float(point[1])
        pt = np.array([x, y, 1.0])
        r = self.H @ pt
        return float(r[0] / r[2]), float(r[1] / r[2])

    def bbox_bottom_center_to_plan(self, bbox: Sequence[float]) -> tuple[float, float]:
        """Convenience: take [x1, y1, x2, y2] and project the bottom-center."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = float(y2)
        return self.image_to_plan((cx, cy))


# ---------------------------------------------------------------------------
# Helper used by the detection filter
# ---------------------------------------------------------------------------

def project_detection(det: dict) -> tuple[float, float] | None:
    """Project the bottom-center of a detection bbox onto the floor plan.

    Returns None if the camera is unknown.
    """
    cam_id = det.get("camera_id")
    if cam_id not in Projector.HOMOGRAPHIES:
        return None
    projector = Projector(cam_id)
    return projector.bbox_bottom_center_to_plan(det["bbox"])
