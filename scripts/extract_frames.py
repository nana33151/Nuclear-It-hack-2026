"""Extract one full-resolution frame per video at a given time.

Usage:
  python scripts/extract_frames.py --time 175.4
  python scripts/extract_frames.py --time 175.4 --video-dir Video --out-dir output/calib_frames

Notes:
  - Does NOT resize. Saves exactly the pixel dimensions of the source video.
  - Time is seconds from video start (same convention as overlays.json.timestamp).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def grab_frame(video_path: Path, t_sec: float, out_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path.name}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_idx = int(round(t_sec * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read frame at t={t_sec}s "
                               f"(frame_idx={frame_idx}, fps={fps})")
    finally:
        cap.release()

    actual_h, actual_w = frame.shape[:2]
    if (actual_w, actual_h) != (W, H):
        # OpenCV-reported size differs from decoded frame — use decoded one.
        print(f"    note: header {W}×{H} != decoded {actual_w}×{actual_h}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), frame):
        raise RuntimeError(f"cannot write {out_path}")
    return actual_w, actual_h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", type=float, required=True,
                    help="seconds from video start (e.g. 175.4)")
    ap.add_argument("--video-dir", default=str(ROOT / "Video"),
                    help="folder with videos (default: <repo>/Video)")
    ap.add_argument("--out-dir", default=str(ROOT / "output" / "calib_frames"))
    ap.add_argument("--exts", default=".mkv,.mp4,.avi,.mov,.webm")
    args = ap.parse_args()

    vid_dir = Path(args.video_dir)
    out_dir = Path(args.out_dir)
    if not vid_dir.exists():
        raise SystemExit(f"video dir not found: {vid_dir}")

    exts = {e.strip().lower() for e in args.exts.split(",") if e.strip()}
    videos = sorted(p for p in vid_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in exts)
    if not videos:
        raise SystemExit(f"no videos in {vid_dir} (looked for {sorted(exts)})")

    print(f"extracting t={args.time}s from {len(videos)} video(s)")
    for v in videos:
        out = out_dir / f"{v.stem}_t{args.time:g}.png"
        try:
            w, h = grab_frame(v, args.time, out)
            print(f"  ✓ {v.name:30s} → {out.name}  ({w}×{h})")
        except Exception as e:
            print(f"  ✗ {v.name:30s} → {e}")


if __name__ == "__main__":
    main()