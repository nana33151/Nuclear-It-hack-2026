"""Export annotated preview mp4 per camera after tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from app.draw import draw_boxes, nearest_overlay, resize_max, scale_items
from app.paths import ROOT


def export_camera_mp4(
    camera_id: str,
    source_path: Path,
    cam_overlays: dict,
    *,
    start_sec: float,
    max_sec: float,
    stride: int,
    fps: float,
    out_path: Path,
    display_max_side: int = 960,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source_path}")

    fps = float(fps or cap.get(cv2.CAP_PROP_FPS) or 25.0)
    start_f = int(start_sec * fps)
    end_f = int((start_sec + max_sec) * fps)
    tol = max(int(stride), int(round(fps * 0.4)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Empty window for {camera_id}")

    preview, scale = resize_max(frame, display_max_side)
    h, w = preview.shape[:2]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open VideoWriter: {out_path}")

    frame_idx = start_f
    n_written = 0
    try:
        while frame_idx < end_f:
            if frame is None:
                break
            preview, scale = resize_max(frame, display_max_side)
            items = nearest_overlay(cam_overlays, frame_idx, tol_frames=tol)
            items = scale_items(items, scale)
            drawn = draw_boxes(preview, items, selected=0, show_all=True)
            if drawn.shape[1] != w or drawn.shape[0] != h:
                drawn = cv2.resize(drawn, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(drawn)
            n_written += 1

            frame_idx += 1
            ok, frame = cap.read()
            if not ok:
                break
            if n_written % 100 == 0:
                print(f"  [{camera_id}] export frames={n_written}")
    finally:
        writer.release()
        cap.release()

    print(f"  [{camera_id}] wrote {out_path} ({n_written} frames)")
    return out_path


def export_previews(
    camera_meta: dict[str, Any],
    overlays: dict[str, dict],
    *,
    start_sec: float,
    max_sec: float,
    stride: int,
    out_dir: Path,
    display_max_side: int = 960,
) -> dict[str, str]:
    """Write output/videos/{camera_id}.mp4; return relative paths from ROOT."""
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    relative: dict[str, str] = {}
    for camera_id, meta in camera_meta.items():
        source = ROOT / meta["path"]
        out_path = videos_dir / f"{camera_id}.mp4"
        export_camera_mp4(
            camera_id,
            source,
            overlays.get(camera_id, {}),
            start_sec=start_sec,
            max_sec=max_sec,
            stride=stride,
            fps=float(meta.get("fps") or 25.0),
            out_path=out_path,
            display_max_side=display_max_side,
        )
        rel = str(out_path.relative_to(ROOT))
        relative[camera_id] = rel
        meta["preview_mp4"] = rel
    return relative
