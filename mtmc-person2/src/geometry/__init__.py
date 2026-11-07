"""Geometry package: office model + mirror filter + projectors."""
from .office_model import OfficeModel, Camera, Segment
from .filter import filter_detections_by_mirror
from .projectors import Projector, project_detection

__all__ = [
    "OfficeModel",
    "Camera",
    "Segment",
    "filter_detections_by_mirror",
    "Projector",
    "project_detection",
]