#!/usr/bin/env python3
"""Verify ReID dispersion split cuts a dirty tracklet."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.contract import BoxObservation, Tracklet
from src.tracklets.split import split_by_reid_dispersion


def main() -> None:
    boxes = [
        BoxObservation(i, float(i) / 25.0, [0, 0, 10, 20], 0.9, True) for i in range(10)
    ]
    # First half person A, second half person B (orthogonal-ish vectors)
    embs: list[list[float] | None] = [[1.0, 0.0]] * 5 + [[0.0, 1.0]] * 5
    tr = Tracklet(
        camera_id="cam_01",
        local_id="dirty_1",
        t_start=0.0,
        t_end=boxes[-1].timestamp,
        boxes=boxes,
        reid_emb=[0.7, 0.7],
        color_top=None,
        color_bottom=None,
        representative_frame=5,
    )
    out = split_by_reid_dispersion([tr], threshold=0.3, per_frame_embs={"dirty_1": embs})
    assert len(out) == 2, f"expected 2 pieces, got {len(out)}"
    assert out[0].local_id.endswith("_s0")
    assert out[1].split_reason == "reid_dispersion"
    print(f"ok: split into {len(out)} tracklets {[t.local_id for t in out]}")


if __name__ == "__main__":
    main()
