"""Load shared camera projectors for the application."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence



def _import_projector():
    """Import the shared Projector class."""
    from app.paths import ensure_import_paths
    ensure_import_paths()
    from src.geometry.projectors import Projector
    return Projector


def _norm_cam(cam_id: str) -> str:
    """Map 'camera_new_2' → 'cam2'. Leave other names unchanged."""
    from app.paths import ensure_import_paths
    ensure_import_paths()
    from src.geometry.cameras import normalize_camera_id
    return normalize_camera_id(cam_id)


def available_camera_aliases() -> dict[str, str]:
    """Return {orig_style_id: projector_style_id} for all known cameras."""
    Projector = _import_projector()
    return {
        f"camera_new_{i}": f"cam{i}"
        for i in range(1, 6)
        if f"cam{i}" in Projector.HOMOGRAPHIES
    }


def load_projectors(camera_ids: Iterable[str]) -> dict:
    """Build {original_cam_id: Projector}. Unknown cams are silently skipped."""
    Projector = _import_projector()
    known = set(Projector.HOMOGRAPHIES.keys())
    out: dict = {}
    for cam in camera_ids:
        alias = _norm_cam(cam)
        if alias in known:
            out[cam] = Projector.get(alias)
    return out


def project_bbox(proj, bbox: Sequence[float]) -> Optional[tuple[float, float]]:
    try:
        return proj.bbox_bottom_center_to_plan(bbox)
    except Exception:
        return None


def project_point(proj, point: Sequence[float]) -> Optional[tuple[float, float]]:
    try:
        return proj.image_to_plan(point)
    except Exception:
        return None
