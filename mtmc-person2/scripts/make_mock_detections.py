#!/usr/bin/env python3
"""Write mock detections in the Person 1 JSONL contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io.detections import save_detections
from src.io.mock import make_mock_detections


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/input/mock_detections.jsonl")
    p.add_argument("--camera-id", default="cam_01")
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--persons", type=int, default=2)
    args = p.parse_args()

    dets = make_mock_detections(
        camera_id=args.camera_id,
        num_frames=args.frames,
        num_persons=args.persons,
    )
    path = save_detections(dets, ROOT / args.out)
    print(f"wrote {len(dets)} detections → {path}")


if __name__ == "__main__":
    main()
