"""Load Projector from mtmc-person2/src/geometry/projectors.py.

We load the module by file path, bypassing mtmc-person2/src/geometry/__init__.py,
because that __init__.py eagerly imports .filter -> src.geometry.office_model,
which requires mtmc-person2/ (not mtmc-person2/src/) on sys.path.
We only need the Projector class, which has no such dependencies.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Iterable, Optional, Sequence

from app.paths import PERSON2

_MODULE_NAME = "_mtmc_person2_projectors"
_PROJECTOR_CLS = None


def _import_projector():
    """Import Projector class directly from projectors.py (bypass package init)."""
    global _PROJECTOR_CLS
    if _PROJECTOR_CLS is not None:
        return _PROJECTOR_CLS

    path = PERSON2 / "src" / "geometry" / "projectors.py"
    if not path.exists():
        raise RuntimeError(
            f"Не найден {path}. Проверьте структуру mtmc-person2/."
        )

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Не удалось создать spec для {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    _PROJECTOR_CLS = module.Projector
    return _PROJECTOR_CLS


def _norm_cam(cam_id: str) -> str:
    """Map 'camera_new_2' → 'cam2'. Leave other names unchanged."""
    if cam_id.startswith("camera_new_"):
        return "cam" + cam_id[len("camera_new_"):]
    return cam_id


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