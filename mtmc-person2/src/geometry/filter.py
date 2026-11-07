"""Mirror reflection filter for detections.

Only cameras that actually see the mirror are filtered. Accepts either raw
dicts or FrameDet-like objects; projector is looked up per (aliased) camera so
we don't need to convert detections to dicts or rebind signatures.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

from src.geometry.office_model import OfficeModel


# projector_for_camera(office_camera_id, bbox) -> (X, Y) in meters or None
CameraBBoxProjector = Callable[[str, Sequence[float]], "tuple[float, float] | None"]


def _get_bbox(det) -> Sequence[float] | None:
    if isinstance(det, dict):
        return det.get("bbox")
    return getattr(det, "bbox", None)


def _get_camera_id(det, field: str) -> str:
    if isinstance(det, dict):
        return str(det.get(field, ""))
    return str(getattr(det, field, ""))


def filter_detections_by_mirror(
    detections: Sequence,
    office: OfficeModel,
    projector_for_camera: CameraBBoxProjector,
    camera_id_field: str = "camera_id",
    camera_aliases: Mapping[str, str] | None = None,
    debug_first_n: int = 5,
) -> list:
    """Drop detections whose projected floor point falls on a mirror zone.

    Args:
        detections: sequence of FrameDet or dict-like detections.
        office: OfficeModel with mirror + polygon + camera sees_mirror flags.
        projector_for_camera: callable (office_camera_id, bbox) -> (X, Y)|None.
        camera_id_field: attribute/key holding the camera id on each detection.
        camera_aliases: optional map {video_stem -> office.camera_id}
            (e.g. {"camera_new_2": "cam2"}).
        debug_first_n: print first N KEEP/DROP decisions on mirror cameras
            so you can confirm the projection lands where you expect.

    Only cameras with sees_mirror=True are filtered; other cameras pass through.
    """
    kept: list = []
    aliases = dict(camera_aliases or {})
    debug_shown = 0

    for det in detections:
        cam_raw = _get_camera_id(det, camera_id_field)
        cam = aliases.get(cam_raw, cam_raw)

        if not office.camera_sees_mirror(cam):
            kept.append(det)
            continue

        bbox = _get_bbox(det)
        if bbox is None:
            kept.append(det)
            continue

        try:
            floor_xy = projector_for_camera(cam, bbox)
        except Exception as exc:
            if debug_shown < debug_first_n:
                print(f"[mirror] {cam_raw}->{cam} projector error: {exc}")
                debug_shown += 1
            kept.append(det)
            continue

        if floor_xy is None:
            # Cannot project -> keep (safe default)
            kept.append(det)
            continue

        drop = office.is_on_mirror(floor_xy)
        if debug_shown < debug_first_n:
            tag = "DROP" if drop else "KEEP"
            print(
                f"[mirror] {cam_raw}->{cam} bbox_bc="
                f"{[round(float(v), 1) for v in bbox]} -> "
                f"(X={floor_xy[0]:.2f}, Y={floor_xy[1]:.2f}) {tag}"
            )
            debug_shown += 1

        if not drop:
            kept.append(det)

    return kept


def make_identity_projector_for_tests() -> Callable[[Sequence[float]], tuple[float, float]]:
    """For synthetic tests: assume bbox bottom-center is already floor meters."""
    def projector(bbox: Sequence[float]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        return float(cx), float(y2)
    return projector