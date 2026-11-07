"""Mirror reflection filter for detections.

Only cameras that actually see the mirror are filtered.
"""

from __future__ import annotations

from typing import Callable, Sequence

from src.geometry.office_model import OfficeModel


# Type of a projector: takes detection and returns floor (x, y) in meters or None
Projector = Callable[[dict], tuple[float, float] | None]


def filter_detections_by_mirror(
    detections: Sequence[dict],
    office: OfficeModel,
    projector: Projector,
    camera_id_field: str = "camera_id",
) -> list[dict]:
    """Drop detections whose projected floor point falls on a mirror zone.

    Filtering is applied only for cameras that have sees_mirror=True.
    For all other cameras the detections are kept as-is.
    """
    kept = []
    for det in detections:
        cam_id = det.get(camera_id_field, "")
        # Skip filter entirely if this camera does not see any mirror
        if not office.camera_sees_mirror(cam_id):
            kept.append(det)
            continue

        floor_xy = projector(det)
        if floor_xy is None:
            # Cannot project → keep (safe default)
            kept.append(det)
            continue
        if office.is_on_mirror(floor_xy):
            # Reflection → drop
            continue
        kept.append(det)
    return kept


def make_identity_projector_for_tests() -> Projector:
    """For synthetic tests: assume bbox bottom-center is already in floor meters."""
    def projector(det: dict) -> tuple[float, float]:
        x1, y1, x2, y2 = det["bbox"]
        cx = (x1 + x2) / 2.0
        return float(cx), float(y2)
    return projector
