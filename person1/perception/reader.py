"""Streaming, tracker-independent reader for the person1 -> person2 contract."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


class ContractError(ValueError):
    """An export violates the documented interchange contract."""


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ContractError(f"{name}: expected integer >= {minimum}, got {value!r}")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ContractError(f"{name}: expected a finite number, got {value!r}")
    return float(value)


def _required(obj: dict, fields: tuple[str, ...], name: str) -> None:
    missing = [field for field in fields if field not in obj]
    if missing:
        raise ContractError(f"{name}: missing fields: {', '.join(missing)}")


def _json_lines(path: Path) -> Iterator[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except (ValueError, TypeError) as exc:
                    raise ContractError(f"{path.name}:{number}: invalid JSON: {exc}") from exc
                if not isinstance(value, dict):
                    raise ContractError(f"{path.name}:{number}: expected an object")
                yield value
    except OSError as exc:
        raise ContractError(f"Cannot read {path}: {exc}") from exc


@dataclass(frozen=True)
class FrameDetections:
    """One source frame, including frames with zero detected people."""

    camera_id: str
    frame_idx: int
    timestamp: float
    width: int
    height: int
    detections: list[dict[str, Any]]
    timestamp_source: str

    @property
    def detection_ids(self) -> tuple[str, ...]:
        return tuple(record["detection_id"] for record in self.detections)

    @property
    def boxes(self) -> np.ndarray:
        """N x 5 float32: x1, y1, x2, y2, confidence, in ORIGINAL pixels."""
        return np.asarray(
            [record["bbox"] + [record["score"]] for record in self.detections],
            dtype=np.float32,
        ).reshape(-1, 5)


class PerceptionReader:
    """Read one completed export without a dependency on any MOT package.

    Set require_complete=False only when inspecting a failed/partial run. A
    partially written last JSONL line remains an error; this is not a live tail.
    """

    def __init__(self, output_dir: str | Path, require_complete: bool = True):
        self.root = Path(output_dir).resolve()
        try:
            self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ContractError(f"Cannot read manifest.json in {self.root}: {exc}") from exc
        if not isinstance(self.manifest, dict):
            raise ContractError("manifest: expected an object")
        _required(self.manifest, ("schema_version", "status", "created_at", "config", "cameras"), "manifest")
        if self.manifest["schema_version"] != "1.0":
            raise ContractError(f"Unsupported schema_version: {self.manifest['schema_version']!r}")
        if self.manifest["status"] not in {"running", "complete", "failed"}:
            raise ContractError("manifest.status: expected running, complete or failed")
        if require_complete and self.manifest["status"] != "complete":
            raise ContractError("Export is incomplete; wait for status='complete' or inspect with require_complete=False")
        if not isinstance(self.manifest["created_at"], str) or not self.manifest["created_at"]:
            raise ContractError("manifest.created_at: expected a non-empty ISO timestamp string")
        if not isinstance(self.manifest["config"], dict):
            raise ContractError("manifest.config: expected an object")
        cameras = self.manifest["cameras"]
        if not isinstance(cameras, list) or not cameras:
            raise ContractError("manifest.cameras: expected a non-empty list")
        self.cameras: dict[str, dict[str, Any]] = {}
        for camera in cameras:
            if not isinstance(camera, dict):
                raise ContractError("manifest.cameras entries must be objects")
            _required(camera, ("camera_id", "source_video", "width", "height", "fps", "frame_count_expected", "status"), "camera")
            camera_id = camera["camera_id"]
            if not isinstance(camera_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", camera_id):
                raise ContractError("camera_id must contain only letters, numbers, underscores and hyphens")
            if camera_id in self.cameras:
                raise ContractError(f"Duplicate camera_id: {camera_id}")
            _integer(camera["width"], f"{camera_id}.width", 1)
            _integer(camera["height"], f"{camera_id}.height", 1)
            if _number(camera["fps"], f"{camera_id}.fps") <= 0:
                raise ContractError(f"{camera_id}.fps must be positive")
            if camera["frame_count_expected"] is not None:
                _integer(camera["frame_count_expected"], f"{camera_id}.frame_count_expected")
            if not isinstance(camera["source_video"], str) or not camera["source_video"]:
                raise ContractError(f"{camera_id}.source_video must be a non-empty string")
            if camera["status"] not in {"pending", "running", "complete", "failed"}:
                raise ContractError(f"{camera_id}.status is invalid")
            if require_complete and camera["status"] != "complete":
                raise ContractError(f"Camera {camera_id} is incomplete")
            if camera["status"] == "complete":
                _required(camera, ("processed_frames", "detections_count"), camera_id)
                _integer(camera["processed_frames"], f"{camera_id}.processed_frames")
                _integer(camera["detections_count"], f"{camera_id}.detections_count")
            if require_complete or camera["status"] == "complete":
                _required(camera, ("frames_path", "detections_path"), camera_id)
            for key in ("frames_path", "detections_path"):
                if key in camera:
                    self.asset_path(camera[key])
            self.cameras[camera_id] = camera

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(self.cameras)

    def asset_path(self, relative_path: str) -> Path:
        """Resolve an export-relative path, rejecting escapes (including symlinks)."""
        if not isinstance(relative_path, str) or not relative_path:
            raise ContractError("Asset path must be a non-empty string")
        value = Path(relative_path)
        if value.is_absolute() or ".." in value.parts or "\\" in relative_path:
            raise ContractError(f"Asset path must be a safe relative POSIX path: {relative_path!r}")
        resolved = (self.root / value).resolve()
        if not resolved.is_relative_to(self.root):
            raise ContractError(f"Asset path escapes export root: {relative_path!r}")
        if not resolved.is_file():
            raise ContractError(f"Export file is missing: {relative_path}")
        return resolved

    def read_crop(self, detection: dict[str, Any]) -> np.ndarray:
        """Return uint8 RGB pixels; convert to BGR if your ReID code expects BGR."""
        from PIL import Image

        with Image.open(self.asset_path(detection["crop_path"])) as image:
            return np.asarray(image.convert("RGB"))

    def read_mask(self, detection: dict[str, Any]) -> np.ndarray:
        """Return the bool silhouette mask in crop coordinates."""
        from PIL import Image

        with Image.open(self.asset_path(detection["mask_path"])) as image:
            return np.asarray(image.convert("L")) != 0

    def _validate_detection(self, record: dict, frame: dict) -> None:
        _required(record, ("detection_id", "camera_id", "frame_idx", "timestamp", "bbox", "score", "class_id", "class_name", "mask_path", "crop_path", "mask_format", "mask_origin", "mask_size", "usable_for_reid", "quality", "upper_color_hist", "lower_color_hist", "color_method", "color_quality"), "detection")
        label = str(record["detection_id"])
        prefix = f"{frame['camera_id']}:f{frame['frame_idx']:09d}:d"
        if not isinstance(record["detection_id"], str) or not re.fullmatch(re.escape(prefix) + r"\d{3,}", record["detection_id"]):
            raise ContractError(f"{label}: detection_id must identify its camera and source frame")
        if record["camera_id"] != frame["camera_id"] or record["frame_idx"] != frame["frame_idx"]:
            raise ContractError(f"{label}: camera_id/frame_idx does not match frame")
        _integer(record["frame_idx"], f"{label}.frame_idx")
        if abs(_number(record["timestamp"], f"{label}.timestamp") - frame["timestamp"]) > 1e-6:
            raise ContractError(f"{label}: timestamp does not match frame")
        bbox = record["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ContractError(f"{label}: bbox must be [x1,y1,x2,y2]")
        for value in bbox:
            _integer(value, f"{label}.bbox")
        x1, y1, x2, y2 = bbox
        if not (x1 < x2 <= frame["width"] and y1 < y2 <= frame["height"]):
            raise ContractError(f"{label}: bbox outside original frame or empty")
        score = _number(record["score"], f"{label}.score")
        if not 0 <= score <= 1:
            raise ContractError(f"{label}: score outside [0,1]")
        if type(record["class_id"]) is not int or record["class_id"] != 0 or record["class_name"] != "person":
            raise ContractError(f"{label}: expected class_id=0 and class_name='person'")
        if record["mask_format"] != "png_crop" or record["mask_origin"] != [x1, y1] or record["mask_size"] != [x2 - x1, y2 - y1]:
            raise ContractError(f"{label}: invalid mask geometry or mask_format")
        if type(record["usable_for_reid"]) is not bool:
            raise ContractError(f"{label}: usable_for_reid must be boolean")
        if not isinstance(record["quality"], dict) or not isinstance(record["quality"].get("reasons"), list):
            raise ContractError(f"{label}: quality.reasons must be a list")
        if not isinstance(record["color_quality"], dict):
            raise ContractError(f"{label}: color_quality must be an object")
        if record["color_method"] not in {"spatial_hsv", "schp_atr_hsv", "none"}:
            raise ContractError(f"{label}: unsupported color_method")
        for key in ("upper_color_hist", "lower_color_hist"):
            histogram = record[key]
            if histogram is None:
                continue
            if not isinstance(histogram, list) or len(histogram) != 32:
                raise ContractError(f"{label}.{key}: expected null or 32 histogram values")
            values = [_number(value, f"{label}.{key}") for value in histogram]
            if min(values) < 0 or abs(sum(values) - 1) > 1e-5:
                raise ContractError(f"{label}.{key}: histogram must be nonnegative and sum to 1")
            if any(abs(sum(values[start:end]) - 1 / 3) > 1e-5 for start, end in ((0, 16), (16, 24), (24, 32))):
                raise ContractError(f"{label}.{key}: each H/S/V block must sum to 1/3")
        if record["color_method"] == "none" and any(record[key] is not None for key in ("upper_color_hist", "lower_color_hist")):
            raise ContractError(f"{label}: color_method='none' requires null color histograms")
        self.asset_path(record["crop_path"])
        self.asset_path(record["mask_path"])

    def iter_frames(self, camera_id: str) -> Iterator[FrameDetections]:
        """Yield frames in source order, checking the two JSONL streams agree.

        Exhaust the iterator to check totals and detect orphan detections. Memory
        use is proportional to people in one frame, not total video duration.
        """
        if camera_id not in self.cameras:
            raise ContractError(f"Unknown camera {camera_id!r}; available: {', '.join(self.camera_ids)}")
        camera = self.cameras[camera_id]
        if "frames_path" not in camera or "detections_path" not in camera:
            raise ContractError(f"Camera {camera_id} has no exported records yet (status={camera['status']})")
        detections = _json_lines(self.asset_path(camera["detections_path"]))
        previous_frame = -1
        previous_timestamp = -math.inf
        frame_count = detection_count = 0
        for frame in _json_lines(self.asset_path(camera["frames_path"])):
            _required(frame, ("camera_id", "frame_idx", "timestamp", "timestamp_source", "width", "height", "detection_ids"), "frame")
            index = _integer(frame["frame_idx"], "frame_idx")
            timestamp = _number(frame["timestamp"], "frame.timestamp")
            if frame["camera_id"] != camera_id:
                raise ContractError(f"{camera_id}: frame belongs to another camera")
            if index <= previous_frame or timestamp < previous_timestamp:
                raise ContractError(f"{camera_id}: frames/timestamps are not in source order")
            if frame["timestamp_source"] not in {"pts", "fps"}:
                raise ContractError(f"{camera_id}: invalid timestamp_source")
            if frame["width"] != camera["width"] or frame["height"] != camera["height"]:
                raise ContractError(f"{camera_id}: frame dimensions differ from manifest")
            ids = frame["detection_ids"]
            if not isinstance(ids, list) or any(not isinstance(value, str) for value in ids) or len(ids) != len(set(ids)):
                raise ContractError(f"{camera_id} frame {index}: detection_ids must be distinct strings")
            records = []
            for expected_id in ids:
                try:
                    detection = next(detections)
                except StopIteration as exc:
                    raise ContractError(f"{camera_id} frame {index}: missing detection {expected_id}") from exc
                if detection.get("detection_id") != expected_id:
                    raise ContractError(f"{camera_id} frame {index}: detection_ids order does not match detections.jsonl")
                self._validate_detection(detection, frame)
                records.append(detection)
            frame_count += 1
            detection_count += len(records)
            previous_frame, previous_timestamp = index, timestamp
            yield FrameDetections(camera_id, index, timestamp, frame["width"], frame["height"], records, frame["timestamp_source"])
        if next(detections, None) is not None:
            raise ContractError(f"{camera_id}: orphan detection without a matching frame")
        if camera["status"] == "complete":
            if frame_count != camera["processed_frames"] or detection_count != camera["detections_count"]:
                raise ContractError(f"{camera_id}: manifest totals do not match JSONL files")
