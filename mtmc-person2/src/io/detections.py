"""Load / save Person 1 detections. Swap file path when real export arrives."""

from __future__ import annotations

import json
from pathlib import Path

from src.contract import Detection


def load_detections(path: str | Path) -> list[Detection]:
    """Load detections from .jsonl (preferred) or .json list/object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Detections not found: {path}")

    if path.suffix.lower() == ".jsonl":
        rows = [_read_json_line(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "detections" in payload:
            rows = payload["detections"]
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError("JSON detections must be a list or {\"detections\": [...]}")

    return [Detection.from_dict(row) for row in rows]


def save_detections(detections: list[Detection], path: str | Path) -> Path:
    """Write detections as JSONL — same format Person 1 should produce."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for det in detections:
            f.write(json.dumps(det.to_dict(), ensure_ascii=False) + "\n")
    return path


def group_by_camera(detections: list[Detection]) -> dict[str, list[Detection]]:
    grouped: dict[str, list[Detection]] = {}
    for det in detections:
        grouped.setdefault(det.camera_id, []).append(det)
    for cam in grouped:
        grouped[cam].sort(key=lambda d: (d.frame_idx, d.timestamp))
    return grouped


def _read_json_line(line: str) -> dict:
    row = json.loads(line)
    if not isinstance(row, dict):
        raise ValueError("Each JSONL line must be an object")
    return row
