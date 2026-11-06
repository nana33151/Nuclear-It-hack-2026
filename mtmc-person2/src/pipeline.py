"""Person 2 end-to-end: detections → local tracklets.

Person 1 plug-in: set detections.source=jsonl and detections.path to their export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.io.detections import group_by_camera, load_detections, save_detections
from src.io.mock import make_mock_detections
from src.io.tracklets import save_tracklets
from src.mot.factory import build_mot
from src.reid.factory import build_embedder
from src.tracklets.build import raw_tracks_to_tracklets
from src.tracklets.split import split_by_reid_dispersion


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_detections(cfg: dict[str, Any]):
    det_cfg = cfg["detections"]
    source = det_cfg["source"]
    if source == "mock":
        mock = det_cfg.get("mock", {})
        dets = make_mock_detections(**mock)
        out = Path("data/input/mock_detections.jsonl")
        save_detections(dets, out)
        return dets, out
    if source in ("jsonl", "json", "file"):
        path = Path(det_cfg["path"])
        return load_detections(path), path
    raise ValueError(f"Unknown detections.source: {source}")


def run(cfg: dict[str, Any]):
    detections, source_path = resolve_detections(cfg)
    mot_cfg = cfg.get("mot", {})
    mot = build_mot(
        mot_cfg.get("backend", "bytetrack"),
        track_thresh=mot_cfg.get("track_thresh", 0.5),
        track_buffer=mot_cfg.get("track_buffer", 30),
        match_thresh=mot_cfg.get("match_thresh", 0.8),
        frame_rate=mot_cfg.get("frame_rate", 25),
    )

    reid_cfg = cfg.get("reid", {})
    embedder = None
    if reid_cfg.get("enabled", False):
        embedder = build_embedder(
            reid_cfg.get("backend", "color_mock"),
            weights=reid_cfg.get("weights"),
            only_usable_for_reid=reid_cfg.get("only_usable_for_reid", True),
            device=reid_cfg.get("device"),
        )
    embed_fn = (embedder.embed if embedder is not None else None)

    all_tracklets = []
    for camera_id, cam_dets in group_by_camera(detections).items():
        raw = mot.track(cam_dets)
        tracklets, per_frame = raw_tracks_to_tracklets(
            camera_id=camera_id,
            raw_tracks=raw,
            embed_fn=embed_fn,
            min_frames=int(cfg["tracklet"].get("min_frames", 3)),
            representative=cfg["tracklet"].get("representative", "middle"),
        )
        tracklets = split_by_reid_dispersion(
            tracklets,
            threshold=float(cfg["tracklet"].get("split_dispersion_threshold", 0.35)),
            per_frame_embs=per_frame if embed_fn else None,
        )
        all_tracklets.extend(tracklets)

    out_dir = Path(cfg["output"]["dir"])
    out_path = out_dir / cfg["output"].get("tracklets_name", "tracklets.jsonl")
    save_tracklets(all_tracklets, out_path)

    with_reid = sum(1 for t in all_tracklets if t.reid_emb is not None)
    return {
        "source_path": str(source_path),
        "n_detections": len(detections),
        "n_tracklets": len(all_tracklets),
        "n_with_reid": with_reid,
        "mot_backend": mot_cfg.get("backend"),
        "reid_backend": reid_cfg.get("backend") if reid_cfg.get("enabled") else None,
        "tracklets_path": str(out_path),
        "cameras": sorted({t.camera_id for t in all_tracklets}),
    }
