"""Interactive point clicker. Opens a frame in matplotlib, saves paired
(image_px, plan_m) points to JSON for calibrate_homography.py.

Usage:
  python scripts/click_calibration_points.py \
      --frame output/calib_frames/camera_new_2_t175.4.png \
      --camera-id camera_new_2 \
      --plan-points "2.30 7.80; 2.70 7.80; 7.40 3.20; 7.80 3.20" \
      --out output/calib_points.json

  # plan points from file (one "X Y" per line):
  python scripts/click_calibration_points.py \
      --frame ... --camera-id ... \
      --plan-points calib/camera_new_2_plan.txt \
      --out output/calib_points.json

Controls:
  Left click   — place next point (order matches --plan-points)
  Right click  — undo last point
  Toolbar      — zoom / pan (zoom in for precision before clicking)

Multiple cameras: run once per camera. JSON is merged by camera_id
(pass --overwrite to replace an existing entry).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_points(text_or_path: str) -> list[tuple[float, float]]:
    """Parse 'X Y; X Y; ...' or a text file with one 'X Y' per line."""
    p = Path(text_or_path)
    text = p.read_text(encoding="utf-8") if (p.exists() and p.is_file()) else text_or_path
    out: list[tuple[float, float]] = []
    for token in text.replace(";", "\n").splitlines():
        token = token.strip().rstrip(",").strip()
        if not token:
            continue
        parts = token.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            out.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return out


def click_points(image_path: Path, plan_pts: list[tuple[float, float]]) -> list[dict]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"cannot read {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    n = len(plan_pts)
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(img_rgb)
    ax.set_axis_off()

    clicked: list[tuple[float, float]] = []
    artists: list[list] = []

    def update_title() -> None:
        if len(clicked) == n:
            ax.set_title("All points placed — close the window",
                         color="green", fontsize=13)
            return
        nxt = plan_pts[len(clicked)]
        ax.set_title(
            f"Click point {len(clicked)+1}/{n} on the floor  "
            f"(plan target = X {nxt[0]:.2f} м, Y {nxt[1]:.2f} м)  ·  "
            f"right-click = undo  ·  toolbar = zoom",
            fontsize=12,
        )

    def on_click(event) -> None:
        if event.inaxes != ax or event.xdata is None:
            return
        if event.button == 3:  # right-click undo
            if clicked:
                clicked.pop()
                if artists:
                    for obj in artists.pop():
                        try:
                            obj.remove()
                        except Exception:
                            pass
                update_title()
                fig.canvas.draw_idle()
            return
        if event.button != 1:
            return

        clicked.append((event.xdata, event.ydata))
        idx = len(clicked)
        pt = plan_pts[idx - 1]
        m, = ax.plot(event.xdata, event.ydata, "r+", ms=18, mew=3)
        t = ax.annotate(
            f"{idx}: ({pt[0]:.2f}, {pt[1]:.2f})",
            (event.xdata, event.ydata),
            color="yellow", fontsize=11, fontweight="bold",
            xytext=(10, 10), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="yellow", alpha=0.75),
        )
        artists.append([m, t])
        update_title()
        fig.canvas.draw_idle()
        if len(clicked) == n:
            plt.pause(0.35)
            plt.close(fig)

    update_title()
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()

    if len(clicked) != n:
        raise RuntimeError(f"expected {n} clicks, got {len(clicked)}")

    return [
        {
            "index": i + 1,
            "image": [float(clicked[i][0]), float(clicked[i][1])],
            "plan": [float(plan_pts[i][0]), float(plan_pts[i][1])],
        }
        for i in range(n)
    ]


def save_points(
    out_path: Path,
    camera_id: str,
    frame_path: Path,
    points: list[dict],
    overwrite: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if out_path.exists() and not overwrite:
        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    img = cv2.imread(str(frame_path))
    frame_size = [int(img.shape[1]), int(img.shape[0])] if img is not None else None
    data[camera_id] = {
        "frame": str(frame_path),
        "frame_size": frame_size,
        "points": points,
    }
    out_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"saved {len(points)} point(s) for '{camera_id}' → {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True,
                    help="path to full-resolution frame (PNG)")
    ap.add_argument("--camera-id", required=True,
                    help="e.g. camera_new_2 (same key as in overlays.json)")
    ap.add_argument("--plan-points", required=True,
                    help="'X Y; X Y; ...' or path to .txt with one 'X Y' per line")
    ap.add_argument("--out", default=str(ROOT / "output" / "calib_points.json"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    frame_path = Path(args.frame)
    if not frame_path.exists():
        print(f"frame not found: {frame_path}", file=sys.stderr)
        return 1
    plan_pts = parse_points(args.plan_points)
    if len(plan_pts) < 4:
        print(f"need >=4 plan points, got {len(plan_pts)}", file=sys.stderr)
        return 1

    try:
        points = click_points(frame_path, plan_pts)
    except RuntimeError as e:
        print(f"clicking failed: {e}", file=sys.stderr)
        return 1

    save_points(Path(args.out), args.camera_id, frame_path, points, args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())