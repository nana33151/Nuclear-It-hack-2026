"""Serialize Person 2 tracklets for Person 3."""

from __future__ import annotations

import json
from pathlib import Path

from src.contract import Tracklet


def save_tracklets(tracklets: list[Tracklet], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for tr in tracklets:
            f.write(json.dumps(tr.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_tracklets(path: str | Path) -> list[dict]:
    path = Path(path)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
