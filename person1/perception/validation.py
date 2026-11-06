"""Validate an export before handing it to person2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .reader import ContractError, PerceptionReader


def validate_export(output_dir: str | Path, verify_images: bool = True) -> dict[str, Any]:
    """Validate every row and optionally decode every crop/mask; bounded memory."""
    reader = PerceptionReader(output_dir)
    result: dict[str, Any] = {"schema_version": "1.0", "valid": True, "frames": 0, "detections": 0, "usable_for_reid": 0, "cameras": {}}
    for camera_id in reader.camera_ids:
        counts = {"frames": 0, "empty_frames": 0, "detections": 0, "usable_for_reid": 0}
        for frame in reader.iter_frames(camera_id):
            counts["frames"] += 1
            counts["empty_frames"] += int(not frame.detections)
            counts["detections"] += len(frame.detections)
            for detection in frame.detections:
                counts["usable_for_reid"] += int(detection["usable_for_reid"])
                if verify_images:
                    _validate_images(reader, detection)
        result["cameras"][camera_id] = counts
        for key in ("frames", "detections", "usable_for_reid"):
            result[key] += counts[key]
    return result


def _validate_images(reader: PerceptionReader, detection: dict) -> None:
    from PIL import Image

    label = detection["detection_id"]
    size = tuple(detection["mask_size"])
    try:
        with Image.open(reader.asset_path(detection["crop_path"])) as crop:
            crop.load()
            if crop.size != size or crop.mode != "RGB":
                raise ContractError(f"{label}: crop must be RGB with size {size}")
        with Image.open(reader.asset_path(detection["mask_path"])) as mask:
            mask.load()
            if mask.format != "PNG" or mask.size != size or mask.mode != "L":
                raise ContractError(f"{label}: mask must be grayscale PNG with size {size}")
            values = np.asarray(mask)
            if not np.all((values == 0) | (values == 255)):
                raise ContractError(f"{label}: mask values must be 0 or 255")
            if not np.any(values):
                raise ContractError(f"{label}: mask is empty")
    except OSError as exc:
        raise ContractError(f"{label}: cannot decode image: {exc}") from exc
