"""Synthetic detections in the Person 1 contract — unblock MOT without real seg."""

from __future__ import annotations

import math
import random

from src.contract import Detection


def make_mock_detections(
    camera_id: str = "cam_01",
    num_frames: int = 60,
    fps: float = 25.0,
    num_persons: int = 2,
    width: int = 1280,
    height: int = 720,
    seed: int = 42,
) -> list[Detection]:
    """Two people crossing with a short occlusion-style gap — good MOT smoke test."""
    rng = random.Random(seed)
    detections: list[Detection] = []

    for person_i in range(num_persons):
        # Horizontal walk; person 1 briefly "missing" mid-clip to exercise T_local.
        y0 = 200 + person_i * 180
        x_start = 80 + person_i * 40
        speed = 8 + person_i * 2
        gap_start = num_frames // 3 if person_i == 0 else None
        gap_end = gap_start + 3 if gap_start is not None else None

        for frame_idx in range(num_frames):
            if gap_start is not None and gap_start <= frame_idx < gap_end:
                continue

            x1 = x_start + speed * frame_idx + 5 * math.sin(frame_idx / 7 + person_i)
            y1 = y0 + 3 * math.sin(frame_idx / 5)
            bw, bh = 70, 160
            x2, y2 = x1 + bw, y1 + bh
            if x2 < 0 or x1 > width or y2 < 0 or y1 > height:
                continue

            score = round(0.75 + 0.2 * rng.random(), 3)
            usable = bh >= 120 and score >= 0.8
            detections.append(
                Detection(
                    camera_id=camera_id,
                    frame_idx=frame_idx,
                    timestamp=round(frame_idx / fps, 4),
                    bbox=[round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    score=score,
                    usable_for_reid=usable,
                    mask=None,
                    crop_path=None,
                    color_top=[0.6 + 0.1 * person_i, 0.2, 0.2],
                    color_bottom=[0.2, 0.2, 0.5 + 0.1 * person_i],
                    det_id=f"{camera_id}_p{person_i}_f{frame_idx}",
                )
            )

    detections.sort(key=lambda d: (d.frame_idx, d.det_id or ""))
    return detections
