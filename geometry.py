"""Apply shared geometry before local tracks are created."""
from app.paths import ensure_import_paths


def filter_mirror_detections(detections, config=None):
    cfg = config or {}
    if not cfg.get("enabled", True):
        return detections
    ensure_import_paths()
    from src.geometry import OfficeModel, filter_detections_by_mirror, project_detection
    office = OfficeModel(
        mirror_margin=float(cfg.get("mirror_margin", 0.25)),
        reject_outside_floor=bool(cfg.get("reject_outside_floor", True)),
    )
    return filter_detections_by_mirror(detections, office, project_detection)
