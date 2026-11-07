"""Camera-to-floor projectors.

Each camera has its own homography (image pixels -> plan meters).
Only cameras that actually see the mirror need a projector for the filter,
but we keep the interface uniform.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Homographies (image pixels -> plan meters, X right, Y down)
# ---------------------------------------------------------------------------

# Camera 2 (sees the mirror)
H_CAM2 = np.array([
    [0.001251, -0.019190, 5.919907],
    [0.000023,  0.000176, -0.073698],
    [0.000143, -0.003069, 1.0],
], dtype=np.float64)

# Camera 1
H_CAM1 = np.array([
    [-0.009409,  0.006394,  7.459238],
    [-0.003867, -0.008060,  9.328105],
    [-0.000460,  0.000912,  1.0],
], dtype=np.float64)
# Camera 3
H_CAM3 = np.array([
    [-0.019402,  0.024947,  8.600000],
    [-0.012324,  0.008965,  8.779557],
    [-0.002256,  0.002901,  1.0],
], dtype=np.float64)
# Camera 4
H_CAM4 = np.array([
    [-0.005143,  0.069808, -6.592956],
    [-0.020875,  0.074192, 25.712872],
    [-0.000237,  0.005062,  1.0],
], dtype=np.float64)
# Camera 5
H_CAM5 = np.array([
    [0.011874, -0.039955,  4.845265],
    [0.005065, -0.048761, 20.647310],
    [0.000373, -0.004324,  1.0],
], dtype=np.float64)


def _project_with_H(H: np.ndarray | None, det: dict) -> tuple[float, float] | None:
    if H is None:
        return None
    x1, y1, x2, y2 = det["bbox"]
    cx = (x1 + x2) / 2.0
    cy = float(y2)          # bottom of the box
    pt = np.array([cx, cy, 1.0])
    r = H @ pt
    return float(r[0] / r[2]), float(r[1] / r[2])


def project_cam1(det: dict) -> tuple[float, float] | None:
    return _project_with_H(H_CAM1, det)


def project_cam2(det: dict) -> tuple[float, float] | None:
    return _project_with_H(H_CAM2, det)


def project_cam3(det: dict) -> tuple[float, float] | None:
    return _project_with_H(H_CAM3, det)


def project_cam4(det: dict) -> tuple[float, float] | None:
    return _project_with_H(H_CAM4, det)


def project_cam5(det: dict) -> tuple[float, float] | None:
    return _project_with_H(H_CAM5, det)


# Unified dispatcher used by the filter
PROJECTORS: dict[str, Callable[[dict], tuple[float, float] | None]] = {
    "cam1": project_cam1,
    "cam2": project_cam2,
    "cam3": project_cam3,
    "cam4": project_cam4,
    "cam5": project_cam5,
}


def project_detection(det: dict) -> tuple[float, float] | None:
    """Dispatch to the correct camera projector. Returns None if unknown or no H."""
    cam_id = det.get("camera_id", "")
    fn = PROJECTORS.get(cam_id)
    if fn is None:
        return None
    return fn(det)
