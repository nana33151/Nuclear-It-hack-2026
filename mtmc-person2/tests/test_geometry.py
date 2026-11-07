"""Synthetic tests for the office geometry and mirror filter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry.office_model import OfficeModel
from src.geometry.filter import filter_detections_by_mirror, make_identity_projector_for_tests


def test_mirror_detection():
    office = OfficeModel()
    assert office.is_on_mirror((4.86, 0.0)) is True
    assert office.is_on_mirror((4.86, 0.3)) is False
    assert office.is_on_mirror((2.0, 3.0)) is False
    print("test_mirror_detection: OK")


def test_only_cam2_filtered():
    office = OfficeModel(mirror_margin=0.2)
    projector = make_identity_projector_for_tests()

    detections = [
        # Reflection seen by cam2 → should be dropped
        {"camera_id": "cam2", "bbox": [4.7, -0.1, 5.0, 0.05], "score": 0.85},
        # Same "reflection" coordinates but from cam1 → must be kept
        {"camera_id": "cam1", "bbox": [4.7, -0.1, 5.0, 0.05], "score": 0.85},
        # Real person from cam2
        {"camera_id": "cam2", "bbox": [4.6, 0.1, 5.1, 0.4], "score": 0.9},
        # Person from cam3
        {"camera_id": "cam3", "bbox": [1.0, 2.0, 1.5, 2.5], "score": 0.95},
    ]

    filtered = filter_detections_by_mirror(detections, office, projector)
    assert len(filtered) == 3, f"Expected 3 kept, got {len(filtered)}"

    # The only dropped one must be the cam2 reflection
    dropped = [d for d in detections if d not in filtered]
    assert len(dropped) == 1
    assert dropped[0]["camera_id"] == "cam2"
    print("test_only_cam2_filtered: OK")


def test_camera_flags():
    office = OfficeModel()
    assert office.camera_sees_mirror("cam2") is True
    assert office.camera_sees_mirror("cam1") is False
    assert office.camera_sees_mirror("cam3") is False
    assert office.camera_sees_mirror("cam4") is False
    assert office.camera_sees_mirror("cam5") is False
    print("test_camera_flags: OK")


if __name__ == "__main__":
    test_mirror_detection()
    test_camera_flags()
    test_only_cam2_filtered()
    print("\nAll geometry tests passed.")
