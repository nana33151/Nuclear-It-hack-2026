"""Crop quality and reproducible colour descriptors (no tracking or ReID).

SCHP backend targets pirocheto/schp-atr-18 ONNX exports, not arbitrary SCHP
checkpoints. Its verified processor uses PIL RGB, a direct 512x512 bilinear
resize and the mean/std below. Reference:
https://huggingface.co/pirocheto/schp-atr-18/blob/main/image_processing_schp.py
No downloaded Python code is executed by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


HISTOGRAM_BINS = (16, 8, 8)
HISTOGRAM_RANGES = ((0, 180), (0, 256), (0, 256))
SCHP_MEAN = np.array([0.406, 0.456, 0.485], dtype=np.float32)
SCHP_STD = np.array([0.225, 0.224, 0.229], dtype=np.float32)


@dataclass(frozen=True)
class QualityConfig:
    min_height: int = 64
    min_width: int = 24
    min_blur: float = 20.0
    min_mask_fraction: float = 0.1
    border_margin: int = 2

    def __post_init__(self) -> None:
        if self.min_height < 1 or self.min_width < 1:
            raise ValueError("Quality minimum crop dimensions must be positive")
        if not np.isfinite(self.min_blur) or self.min_blur < 0:
            raise ValueError("min_blur must be finite and non-negative")
        if not 0 <= self.min_mask_fraction <= 1:
            raise ValueError("min_mask_fraction must be between 0 and 1")
        if self.border_margin < 0:
            raise ValueError("border_margin must be non-negative")


def _validate_crop(crop_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3 or crop_bgr.dtype != np.uint8:
        raise ValueError("crop_bgr must be an HxWx3 uint8 BGR image")
    if not crop_bgr.shape[0] or not crop_bgr.shape[1]:
        raise ValueError("crop_bgr must not be empty")
    if mask.ndim != 2 or mask.shape != crop_bgr.shape[:2]:
        raise ValueError("mask must have the same HxW size as crop_bgr")
    return mask > 0


def assess_quality(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
    bbox: list[int] | tuple[int, int, int, int],
    frame_shape: tuple[int, int],
    config: QualityConfig = QualityConfig(),
) -> dict[str, Any]:
    """Flag poor ReID evidence while preserving the detection for the tracker.

This is a heuristic, not a reliable occlusion/full-body detector. Border
contact is deliberately conservative; sitting people are not rejected by pose.
Sharpness is measured inside an eroded person mask so that background texture
and the person/background boundary do not make a blurred crop look sharp.
"""
    person = _validate_crop(crop_bgr, mask)
    height, width = person.shape
    frame_height, frame_width = frame_shape
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= frame_width and 0 <= y1 < y2 <= frame_height):
        raise ValueError("bbox must be inside the original frame")
    if (y2 - y1, x2 - x1) != (height, width):
        raise ValueError("crop dimensions must match the integer xyxy bbox")
    inner = cv2.erode(person.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian[inner].var()) if np.any(inner) else 0.0
    fraction = float(person.mean())
    margin = config.border_margin
    border = bool(x1 <= margin or y1 <= margin or x2 >= frame_width - margin or y2 >= frame_height - margin)
    reasons = []
    if height < config.min_height or width < config.min_width:
        reasons.append("too_small")
    if sharpness < config.min_blur:
        reasons.append("blurry")
    if not np.any(person) or fraction < config.min_mask_fraction:
        reasons.append("insufficient_person_mask")
    if border:
        reasons.append("touches_frame_border")
    return {
        "usable_for_reid": not reasons,
        "quality": {
            "reasons": reasons,
            "width": width,
            "height": height,
            "blur_score": sharpness,
            "mask_fraction": fraction,
            "touches_border": border,
            "method": "size_mask_laplacian_border_v1",
        },
    }


def _histogram(hsv: np.ndarray, region: np.ndarray, min_pixels: int) -> list[float] | None:
    pixels = hsv[region]
    if len(pixels) < min_pixels:
        return None
    parts = []
    for channel, (bins, interval) in enumerate(zip(HISTOGRAM_BINS, HISTOGRAM_RANGES)):
        counts, _ = np.histogram(pixels[:, channel], bins=bins, range=interval)
        parts.append(counts.astype(np.float64) / len(pixels) / 3.0)
    return np.concatenate(parts).tolist()


def _colour_result(
    crop_bgr: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    method: str,
    min_pixels: int,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    upper_hist = _histogram(hsv, upper, min_pixels)
    lower_hist = _histogram(hsv, lower, min_pixels)
    return {
        "upper_color_hist": upper_hist,
        "lower_color_hist": lower_hist,
        "color_method": method,
        "color_quality": {
            "upper_pixels": int(np.count_nonzero(upper)),
            "lower_pixels": int(np.count_nonzero(lower)),
            "upper_available": upper_hist is not None,
            "lower_available": lower_hist is not None,
            "min_pixels": min_pixels,
            **(details or {}),
        },
    }


class SpatialColorExtractor:
    """Explicit approximate fallback; geometric bands are not garment parsing."""

    method = "spatial_hsv"

    def __init__(self, min_pixels: int = 32) -> None:
        if min_pixels < 1:
            raise ValueError("min_pixels must be positive")
        self.min_pixels = min_pixels

    def extract(self, crop_bgr: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
        person = _validate_crop(crop_bgr, mask)
        height = person.shape[0]
        rows = np.arange(height)[:, None]
        upper = person & (rows >= int(height * 0.15)) & (rows < int(height * 0.55))
        lower = person & (rows >= int(height * 0.55)) & (rows < int(height * 0.90))
        return _colour_result(
            crop_bgr, upper, lower, self.method, self.min_pixels,
            {"approximate": True, "region_policy": "bbox_y_15_55_and_55_90_percent"},
        )


class SCHPColorExtractor:
    """ATR garment parsing through a local ONNX model, with no silent fallback.

ATR: upper clothing=4, skirt=5, pants=6, dress=7. Dress pixels above/below
55% of crop height contribute to upper/lower respectively. Skin, bags, shoes
and scarves are not used. Missing garments yield null rather than black.
"""

    method = "schp_atr_hsv"

    def __init__(self, model_path: str | Path, min_pixels: int = 32, threads: int = 4) -> None:
        if min_pixels < 1 or threads < 1:
            raise ValueError("min_pixels and threads must be positive")
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"SCHP ONNX model is missing: {path}. Download the ATR ONNX weights "
                "or explicitly select --color-mode spatial for approximate colours."
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("SCHP needs onnxruntime; install the project dependencies") from exc
        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        outputs = {item.name for item in self.session.get_outputs()}
        if len(inputs) != 1 or inputs[0].name != "pixel_values" or "logits" not in outputs:
            raise ValueError("Expected pirocheto/schp-atr-18 ONNX with pixel_values input and logits output")
        shape = inputs[0].shape
        if len(shape) != 4 or any(isinstance(value, int) and value != expected for value, expected in zip(shape[1:], (3, 512, 512))):
            raise ValueError(f"Expected SCHP ATR NCHW input with 512x512 images, got {shape}")
        self.min_pixels = min_pixels

    @staticmethod
    def preprocess(crop_bgr: np.ndarray) -> np.ndarray:
        # Match the published wrapper exactly: RGB channels with its documented
        # BGR-indexed constants. Do not substitute ImageNet RGB constants.
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = Image.fromarray(rgb).resize((512, 512), resample=Image.Resampling.BILINEAR)
        data = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
        normalized = (data - SCHP_MEAN) / SCHP_STD
        return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32)

    def parse(self, crop_bgr: np.ndarray) -> np.ndarray:
        logits = self.session.run(["logits"], {"pixel_values": self.preprocess(crop_bgr)})[0]
        if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 18:
            raise ValueError(f"Expected ATR logits [1,18,H,W], got {logits.shape}")
        if not np.isfinite(logits).all():
            raise RuntimeError("SCHP produced non-finite logits")
        height, width = crop_bgr.shape[:2]
        # Resize scores before argmax so interpolation respects class boundaries.
        resized = cv2.resize(logits[0].transpose(1, 2, 0), (width, height), interpolation=cv2.INTER_LINEAR)
        return resized.argmax(axis=2).astype(np.uint8)

    def extract_from_labels(self, crop_bgr: np.ndarray, mask: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
        person = _validate_crop(crop_bgr, mask)
        if labels.shape != person.shape:
            raise ValueError("SCHP labels must have the original crop HxW size")
        rows = np.arange(person.shape[0])[:, None]
        split = int(person.shape[0] * 0.55)
        dress = labels == 7
        upper = person & ((labels == 4) | (dress & (rows < split)))
        lower = person & ((labels == 5) | (labels == 6) | (dress & (rows >= split)))
        return _colour_result(
            crop_bgr, upper, lower, self.method, self.min_pixels,
            {"approximate": False, "label_set": "ATR18", "dress_policy": "split_at_55_percent_crop_height"},
        )

    def extract(self, crop_bgr: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
        _validate_crop(crop_bgr, mask)
        return self.extract_from_labels(crop_bgr, mask, self.parse(crop_bgr))
