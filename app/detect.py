"""YOLO11-seg person boxes (no mask fill — boxes only for UI)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class FrameDet:
    camera_id: str
    frame_idx: int
    timestamp: float
    bbox: list[float]  # x1,y1,x2,y2 in original pixels
    score: float
    color_top: list[float]
    color_bottom: list[float]
    usable_for_reid: bool
    reid: list[float] | None = None  # optional SOLIDER (or other) embedding


def choose_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_yolo_seg(weights: str | Path, device: str = "auto"):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    return model, choose_device(device)


def _mean_rgb(crop: np.ndarray) -> list[float]:
    if crop.size == 0:
        return [0.0, 0.0, 0.0]
    # BGR → RGB mean in [0,1]
    mean = crop.reshape(-1, 3).mean(axis=0).astype(np.float64)
    b, g, r = mean
    return [float(r / 255.0), float(g / 255.0), float(b / 255.0)]


def color_from_bbox(image: np.ndarray, bbox: list[float]) -> tuple[list[float], list[float]]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    crop = image[y1:y2, x1:x2]
    mid = max(1, (y2 - y1) // 2)
    return _mean_rgb(crop[:mid]), _mean_rgb(crop[mid:])


def iter_video_sampled(
    path: Path,
    camera_id: str,
    start_sec: float,
    max_sec: float,
    stride: int,
) -> Iterator[tuple[int, float, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    start_f = int(start_sec * fps)
    end_f = int((start_sec + max_sec) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    idx = start_f
    try:
        while idx < end_f:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - start_f) % stride == 0:
                yield idx, idx / fps, frame
            idx += 1
    finally:
        cap.release()


def detect_people(
    model,
    image: np.ndarray,
    camera_id: str,
    frame_idx: int,
    timestamp: float,
    *,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    min_h: int = 40,
    min_w: int = 16,
    reid_model=None,
) -> list[FrameDet]:
    # Boxes from seg model; masks ignored for drawing (no fill).
    result = model.predict(
        source=image,
        classes=[0],
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
        retina_masks=False,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    out: list[FrameDet] = []
    h, w = image.shape[:2]
    crops: list[np.ndarray | None] = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = [float(v) for v in box]
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(w), x2), min(float(h), y2)
        if x2 - x1 < min_w or y2 - y1 < min_h:
            continue
        bbox = [x1, y1, x2, y2]
        top, bottom = color_from_bbox(image, bbox)
        usable = float(score) >= 0.4 and (y2 - y1) >= 64
        if reid_model is not None and usable:
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            crops.append(image[iy1:iy2, ix1:ix2].copy())
        else:
            crops.append(None)
        out.append(
            FrameDet(
                camera_id=camera_id,
                frame_idx=frame_idx,
                timestamp=float(timestamp),
                bbox=bbox,
                score=float(score),
                color_top=top,
                color_bottom=bottom,
                usable_for_reid=usable,
            )
        )
    if reid_model is not None and any(c is not None for c in crops):
        # Batch only usable crops; map embeddings back.
        batch_idx = [i for i, c in enumerate(crops) if c is not None]
        batch_crops = [crops[i] for i in batch_idx]
        embs = reid_model.embed_bgr_crops(batch_crops)
        for i, emb in zip(batch_idx, embs):
            if emb is not None:
                out[i].reid = emb.tolist()
    return out
