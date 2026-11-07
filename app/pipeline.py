"""End-to-end offline MTMC demo pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from app.detect import detect_people, iter_video_sampled, load_yolo_seg
from app.export_video import export_previews
from app.local_track import run_mot
from app.merge import build_overlays, to_global_tracks
from app.paths import ROOT
from app.reid_solider import build_solider_from_cfg


VIDEO_EXTS = {".mkv", ".mp4", ".avi", ".mov"}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else ROOT / "app" / "config.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_videos(video_dir: Path, cameras: list[str] | None) -> list[tuple[str, Path]]:
    files = sorted(
        p for p in video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS and p.is_file()
    )
    if cameras:
        wanted = set(cameras)
        files = [p for p in files if p.stem in wanted]
    if not files:
        raise FileNotFoundError(f"No videos in {video_dir}")
    return [(p.stem, p) for p in files]


def _probe_fps(path: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()
    return fps


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    video_dir = ROOT / cfg["video_dir"]
    cams = discover_videos(video_dir, cfg.get("cameras"))
    model_path = ROOT / cfg["model"]
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {model_path}")

    model, device = load_yolo_seg(model_path, cfg.get("device", "auto"))
    reid_model = build_solider_from_cfg(cfg.get("reid"), device=device if device not in ("auto",) else None)
    reid_name = "solider" if reid_model is not None else str((cfg.get("reid") or {}).get("backend", "color"))
    print(f"device={device} model={model_path.name} cameras={len(cams)} reid={reid_name}")

    all_tracklets = []
    camera_meta: dict[str, Any] = {}
    for camera_id, path in cams:
        fps = _probe_fps(path)
        print(f"[{camera_id}] detect {cfg['start_sec']}..{cfg['start_sec']+cfg['max_sec']}s stride={cfg['stride']}")
        dets = []
        n_frames = 0
        for frame_idx, ts, image in iter_video_sampled(
            path,
            camera_id,
            start_sec=float(cfg["start_sec"]),
            max_sec=float(cfg["max_sec"]),
            stride=int(cfg["stride"]),
        ):
            n_frames += 1
            dets.extend(
                detect_people(
                    model,
                    image,
                    camera_id,
                    frame_idx,
                    ts,
                    conf=float(cfg["confidence"]),
                    iou=float(cfg["iou"]),
                    imgsz=int(cfg["imgsz"]),
                    device=device,
                    reid_model=reid_model,
                )
            )
            if n_frames % 20 == 0:
                print(f"  frames={n_frames} dets_total={len(dets)}")

        mot_cfg = cfg.get("mot") or {}
        tracklets = run_mot(
            dets,
            track_thresh=float(mot_cfg.get("track_thresh", 0.4)),
            track_buffer=int(mot_cfg.get("track_buffer", 30)),
            match_thresh=float(mot_cfg.get("match_thresh", 0.8)),
            frame_rate=int(round(fps)),
            min_frames=int(mot_cfg.get("min_frames", 2)),
            n_init=int(mot_cfg.get("n_init", 3)),
            reid_weight=float(mot_cfg.get("reid_weight", 0.45)),
            split_reid_thresh=float(mot_cfg.get("split_reid_thresh", 0.45)),
            stitch_min_sim=float(mot_cfg.get("stitch_min_sim", 0.6)),
            stitch_max_gap=float(mot_cfg.get("stitch_max_gap", 8.0)),
        )
        print(f"  tracklets={len(tracklets)} from {len(dets)} detections / {n_frames} frames")
        all_tracklets.extend(tracklets)
        camera_meta[camera_id] = {
            "path": str(path.relative_to(ROOT)),
            "fps": fps,
            "n_detections": len(dets),
            "n_tracklets": len(tracklets),
        }

    
    out_dir = ROOT / cfg.get("output_dir", "output")
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = out_dir / cfg.get("tracks_name", "tracks.json")
    overlays_path = out_dir / cfg.get("overlays_name", "overlays.json")


    merge_cfg = cfg.get("merge", {})
    max_tt = float(merge_cfg.get("max_transition_time", 20.0))
    tracks = to_global_tracks(
        all_tracklets,
        T=float(merge_cfg.get("T", 8.0)),
        min_sim=float(merge_cfg.get("min_sim", 0.45)),
        max_transition_time=max_tt,
        method=str(merge_cfg.get("method", "lifecycle")),
        merge_cfg=merge_cfg,
        min_hits=int(merge_cfg.get("min_hits", 2)),
        min_confirm_sec=float(merge_cfg.get("min_confirm_sec", 0.4)),
        lost_timeout_sec=merge_cfg.get("lost_timeout_sec"),
        enable_bev=bool(merge_cfg.get("enable_bev", True)),      # NEW
        debug=bool(merge_cfg.get("debug", False)),               # NEW
        debug_path=str(out_dir / "tracker_decisions.json"),      # NEW
    )

    overlays = build_overlays(tracks)

    # Drop heavy boxes from tracks.json segments for readability; overlays keep them.
    tracks_light = []
    for tr in tracks:
        tracks_light.append(
            {
                "track_id": tr["track_id"],
                "t_start": tr["t_start"],
                "t_end": tr["t_end"],
                "cameras": tr["cameras"],
                "close_reason": tr["close_reason"],
                "state": tr.get("state", "confident"),
                "segments": [
                    {
                        "camera_id": s["camera_id"],
                        "local_id": s["local_id"],
                        "t_start": s["t_start"],
                        "t_end": s["t_end"],
                        "n_boxes": len(s.get("boxes") or []),
                    }
                    for s in tr["segments"]
                ],
            }
        )

    payload = {
        "n_tracks": len(tracks_light),
        "n_cross_camera": sum(1 for t in tracks_light if len(t["cameras"]) > 1),
        "reid": reid_name,
        "window": {
            "start_sec": cfg["start_sec"],
            "max_sec": cfg["max_sec"],
            "stride": cfg["stride"],
        },
        "cameras": camera_meta,
        "tracks": tracks_light,
    }
    if cfg.get("export_mp4", True):
        print("exporting preview mp4…")
        export_previews(
            camera_meta,
            overlays,
            start_sec=float(cfg["start_sec"]),
            max_sec=float(cfg["max_sec"]),
            stride=int(cfg["stride"]),
            out_dir=out_dir,
            display_max_side=int(cfg.get("display_max_side", 960)),
        )
        payload["cameras"] = camera_meta

    tracks_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    overlays_path.write_text(
        json.dumps(
            {
                "cameras": camera_meta,
                "overlays": overlays,
                "n_tracks": len(tracks_light),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"n_tracks={len(tracks_light)} cross_cam={payload['n_cross_camera']}")
    print(f"wrote {tracks_path}")
    print(f"wrote {overlays_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline multi-cam tracking pipeline")
    parser.add_argument("--config", default=str(ROOT / "app" / "config.yaml"))
    parser.add_argument("--start-sec", type=float, default=None)
    parser.add_argument("--max-sec", type=float, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--cameras", nargs="*", default=None)
    parser.add_argument("--no-export-mp4", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.start_sec is not None:
        cfg["start_sec"] = args.start_sec
    if args.max_sec is not None:
        cfg["max_sec"] = args.max_sec
    if args.stride is not None:
        cfg["stride"] = args.stride
    if args.cameras:
        cfg["cameras"] = args.cameras
    if args.no_export_mp4:
        cfg["export_mp4"] = False
    run(cfg)


if __name__ == "__main__":
    main()
