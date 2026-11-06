"""Shared bbox + track-id overlay (no mask fill)."""

from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np

ID_COLORS = [
    (40, 180, 90),
    (40, 120, 255),
    (220, 120, 40),
    (200, 60, 200),
    (60, 200, 220),
    (255, 80, 80),
    (180, 180, 40),
    (120, 80, 255),
]


def color_for(track_id: int) -> tuple[int, int, int]:
    return ID_COLORS[(int(track_id) - 1) % len(ID_COLORS)]


def scale_items(items: Sequence[dict[str, Any]], scale: float) -> list[dict[str, Any]]:
    if abs(scale - 1.0) < 1e-6:
        return [dict(it) for it in items]
    out = []
    for it in items:
        box = it.get("bbox") or [0, 0, 0, 0]
        out.append(
            {
                **it,
                "bbox": [float(v) * scale for v in box],
            }
        )
    return out


def resize_max(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale >= 0.999:
        return image, 1.0
    resized = cv2.resize(
        image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )
    return resized, scale


def nearest_overlay(
    cam_overlays: dict,
    frame_idx: int,
    tol_frames: int = 8,
) -> list[dict]:
    """Find overlays near frame_idx (pipeline may use stride)."""
    if not cam_overlays:
        return []
    key = str(int(frame_idx))
    if key in cam_overlays:
        return cam_overlays[key]
    best, best_d = None, tol_frames + 1
    for k, items in cam_overlays.items():
        d = abs(int(k) - frame_idx)
        if d < best_d:
            best_d, best = d, items
    if best is None or best_d > tol_frames:
        return []
    return best


def draw_boxes(
    frame: np.ndarray,
    items: Sequence[dict[str, Any]],
    selected: int = 0,
    show_all: bool = True,
) -> np.ndarray:
    """Draw boxes + readable ID labels. Call after resize for visible text."""
    out = frame.copy()
    h, _w = out.shape[:2]
    font_scale = max(0.8, h / 900.0)
    base_thickness = max(2, int(round(h / 480.0)))
    text_thickness = max(2, base_thickness)

    for item in items:
        tid = int(item["track_id"])
        if not show_all and selected > 0 and tid != selected:
            continue
        if selected > 0 and tid != selected and show_all:
            thickness = max(1, base_thickness - 1)
        else:
            thickness = base_thickness + (1 if selected == tid else 0)

        x1, y1, x2, y2 = [int(round(v)) for v in item["bbox"]]
        color = color_for(tid)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        label = f"ID {tid}"
        (_tw, th), _bl = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )
        # Prefer above box; if near top edge, place inside below top edge.
        text_y = y1 - 8
        if text_y - th < 2:
            text_y = min(h - 4, y1 + th + 8)
        text_x = max(2, x1)

        # Black outline for contrast on busy backgrounds
        for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1), (-2, 0), (2, 0), (0, -2), (0, 2)):
            cv2.putText(
                out,
                label,
                (text_x + dx, text_y + dy),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                text_thickness + 2,
                cv2.LINE_AA,
            )
        cv2.putText(
            out,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            text_thickness,
            cv2.LINE_AA,
        )
    return out
