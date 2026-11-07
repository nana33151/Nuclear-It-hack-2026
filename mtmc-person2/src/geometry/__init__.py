"""Geometry package: office model + mirror filter."""
from .office_model import OfficeModel, Camera, Segment
from .filter import filter_detections_by_mirror, make_identity_projector_for_tests

__all__ = [
    "OfficeModel",
    "Camera",
    "Segment",
    "filter_detections_by_mirror",
    "make_identity_projector_for_tests",
]
