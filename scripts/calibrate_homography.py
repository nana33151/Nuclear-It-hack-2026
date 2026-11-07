"""Compute per-camera homography from paired pixel↔plan points.

Pure OpenCV. No matplotlib. Consumes output of click_calibration_points.py
or inline pairs of the form  "u1 v1 X1 Y1; u2 v2 X2 Y2; ...".

Usage:
  # from paired points JSON, one camera:
  python scripts/calibrate_homography.py \
      --input output/calib_points.json \
      --camera-id camera_new_2 \
      --out output/homographies.json

  # all cameras in the file:
  python scripts/calibrate_homography.py \
      --input output/calib_points.json --all \
      --out output/homographies.json

  # inline pairs (no intermediate JSON):
  python scripts/calibrate_homography.py \
      --camera-id camera_new_2 \
      --points "530 1030 2.30 7.80; 640 1030 2.70 7.80; 1750 520 7.40 3.20; 1860 520 7.80 3.20" \
      --out output/homographies.json

Output:
  output/homographies.json    {"camera_new_2": [[3x3]], ...}  (merge by key)
  output/<cam>_check.png      OpenCV-rendered plan validation
  stdout                      paste-ready snippet for Projector.HOMOGRAPHIES
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_inline_pairs(text: str) -> tuple[np.ndarray, np.ndarray]:
    """'u1 v1 X1 Y1; u2 v2 X2 Y2; ...' → (img_pts, plan_pts)."""
    img, plan = [], []
    for token in text.replace(";", "\n").splitlines():
        token = token.strip().rstrip(",").strip()
        if not token:
            continue
        parts = token.replace(",", " ").split()
        if len(parts) != 4:
            raise ValueError(f"need 4 values per pair (u v X Y), got {token!r}")
        u, v, X, Y = map(float, parts)
        img.append((u, v))
        plan.append((X, Y))
    return np.asarray(img, np.float32), np.asarray(plan, np.float32)


def load_pairs_from_json(
    path: Path, camera_id: str | None = None
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cam, entry in data.items():
        if camera_id and cam != camera_id:
            continue
        pts = entry.get("points", [])
        img = [tuple(p["image"]) for p in pts]
        plan = [tuple(p["plan"]) for p in pts]
        if not img or not plan:
            continue
        out[cam] = (np.asarray(img, np.float32), np.asarray(plan, np.float32))
    if camera_id and camera_id not in out:
        raise KeyError(f"camera '{camera_id}' not found in {path}")
    return out


def compute_homography(img_pts: np.ndarray, plan_pts: np.ndarray):
    n = len(img_pts)
    if n > 4:
        H, mask = cv2.findHomography(
            img_pts, plan_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
    else:
        H, mask = cv2.findHomography(img_pts, plan_pts, method=0)
    if H is None:
        return None, None
    inliers = mask.ravel().astype(bool) if mask is not None else np.ones(n, bool)
    return H, inliers


def verdict_from_max_err(mx: float) -> str:
    if mx > 1.0:
        return "bad"
    if mx > 0.5:
        return "rough"
    if mx > 0.2:
        return "ok_bev"
    return "good"


def render_check_image(
    camera_id: str,
    plan_pts: np.ndarray,
    reproj_pts: np.ndarray,
    inliers: np.ndarray,
    out_path: Path,
    px_per_m: int = 100,
    pad_m: float = 1.0,
) -> None:
    """OpenCV-only validation plot: plan targets (dots) vs reprojections (x)."""
    finite = (
        np.isfinite(plan_pts).all(axis=1)
        & np.isfinite(reproj_pts).all(axis=1)
    )
    if finite.any():
        plan_pts = plan_pts[finite]
        reproj_pts = reproj_pts[finite]
        inliers = inliers[finite]
    if len(plan_pts) == 0:
        return

    xs = plan_pts[:, 0]
    ys = plan_pts[:, 1]
    x0, x1 = float(xs.min()) - pad_m, float(xs.max()) + pad_m
    y0, y1 = float(ys.min()) - pad_m, float(ys.max()) + pad_m
    W = max(200, int((x1 - x0) * px_per_m))
    Hh = max(200, int((y1 - y0) * px_per_m))
    canvas = np.full((Hh, W, 3), 255, np.uint8)

    def to_px(X: float, Y: float) -> tuple[int, int]:
        return (
            int(np.clip((X - x0) * px_per_m, 0, W - 1)),
            int(np.clip((Y - y0) * px_per_m, 0, Hh - 1)),
        )

    for gx in range(int(np.ceil(x0)), int(np.floor(x1)) + 1):
        cv2.line(canvas, to_px(gx, y0), to_px(gx, y1), (235, 235, 235), 1)
        cv2.putText(canvas, f"{gx}", (to_px(gx, y0)[0] + 2, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1,
                    cv2.LINE_AA)
    for gy in range(int(np.ceil(y0)), int(np.floor(y1)) + 1):
        cv2.line(canvas, to_px(x0, gy), to_px(x1, gy), (235, 235, 235), 1)
        cv2.putText(canvas, f"{gy}", (2, to_px(x0, gy)[1] + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1,
                    cv2.LINE_AA)

    for i in range(len(plan_pts)):
        tx, ty = float(plan_pts[i, 0]), float(plan_pts[i, 1])
        rx, ry = float(reproj_pts[i, 0]), float(reproj_pts[i, 1])
        col_reproj = (0, 0, 220) if bool(inliers[i]) else (0, 165, 255)
        cv2.circle(canvas, to_px(tx, ty), 6, (200, 100, 0), -1)
        cv2.drawMarker(canvas, to_px(rx, ry), col_reproj,
                       cv2.MARKER_TILTED_CROSS, 14, 2)
        cv2.line(canvas, to_px(tx, ty), to_px(rx, ry), (130, 130, 130), 1)
        cv2.putText(canvas, str(i + 1), to_px(tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1,
                    cv2.LINE_AA)

    errs = np.linalg.norm(reproj_pts - plan_pts, axis=1)
    max_err = float(errs.max())
    header = (f"{camera_id}  max_err={max_err:.2f} m  "
              f"({verdict_from_max_err(max_err)})")
    cv2.rectangle(canvas, (0, Hh - 26), (W, Hh), (30, 30, 30), -1)
    cv2.putText(canvas, header, (8, Hh - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (250, 250, 250), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def process_camera(
    camera_id: str,
    img_pts: np.ndarray,
    plan_pts: np.ndarray,
    out_json: Path,
    overwrite: bool,
    save_check: bool = True,
) -> bool:
    if len(img_pts) != len(plan_pts):
        print(f"❌ {camera_id}: len mismatch img={len(img_pts)} plan={len(plan_pts)}")
        return False
    if len(img_pts) < 4:
        print(f"❌ {camera_id}: need >=4 points, got {len(img_pts)}")
        return False

    H, inliers = compute_homography(img_pts, plan_pts)
    if H is None:
        print(f"❌ {camera_id}: findHomography failed "
              f"(collinear points or wrong pairing)")
        return False

    proj = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    errs = np.linalg.norm(proj - plan_pts, axis=1)
    inl_err = errs[inliers] if inliers.any() else errs
    max_err = float(inl_err.max())
    verdict = verdict_from_max_err(max_err)

    print(f"\n=== {camera_id} ===")
    print(f"  points: {len(img_pts)}  inliers: {int(inliers.sum())}/{len(inliers)}")
    for i in range(len(img_pts)):
        tag = "✓" if inliers[i] else "✗ OUTLIER"
        print(
            f"    [{i+1}] img=({img_pts[i,0]:7.1f},{img_pts[i,1]:7.1f})  "
            f"plan=({plan_pts[i,0]:6.2f},{plan_pts[i,1]:6.2f})  "
            f"err={errs[i]:.3f} м  {tag}"
        )
    print(f"  inlier err: mean={inl_err.mean():.3f}  "
          f"median={float(np.median(inl_err)):.3f}  "
          f"max={max_err:.3f} м  [{verdict}]")
    if verdict == "bad":
        print("  ❌ max_err > 1 м — калибровка не годится, проверь порядок точек")
    elif verdict == "rough":
        print("  ⚠️  max_err > 0.5 м — BEV будет заметно гулять")
    elif verdict == "ok_bev":
        print("  ⚠️  приемлемо для BEV, не годится для mirror-фильтра")
    else:
        print("  ✅ хорошая калибровка")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if out_json.exists() and not overwrite:
        try:
            data = json.loads(out_json.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data[camera_id] = H.tolist()
    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"  → saved H to {out_json}[{camera_id}]")

    if save_check:
        check_path = out_json.parent / f"{camera_id}_check.png"
        render_check_image(camera_id, plan_pts, proj, inliers, check_path)
        print(f"  → {check_path}")

    print("  # paste into Projector.HOMOGRAPHIES (key = camN, not camera_new_N):")
    for row in H:
        print(f"    np.array([{row[0]: .6f}, {row[1]: .6f}, {row[2]: .6f}]),")

    return verdict in ("good", "ok_bev")


def main() -> int:
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--input",
                     help="paired points JSON from click_calibration_points.py")
    grp.add_argument("--points",
                     help="inline 'u v X Y; ...' pairs (no JSON needed)")

    ap.add_argument("--camera-id",
                    help="required with --input unless --all is used")
    ap.add_argument("--all", action="store_true",
                    help="process all cameras present in --input")
    ap.add_argument("--out", default=str(ROOT / "output" / "homographies.json"))
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing entries instead of merging")
    ap.add_argument("--no-check", action="store_true",
                    help="skip writing <cam>_check.png")
    args = ap.parse_args()

    out_json = Path(args.out)
    all_ok = True

    if args.input:
        try:
            pairs = load_pairs_from_json(
                Path(args.input),
                camera_id=None if args.all else args.camera_id,
            )
        except (KeyError, FileNotFoundError, json.JSONDecodeError) as e:
            print(f"❌ {e}", file=sys.stderr)
            return 1
        if not pairs:
            print(f"❌ no points found in {args.input}", file=sys.stderr)
            return 1
        for cam, (img_pts, plan_pts) in pairs.items():
            ok = process_camera(
                cam, img_pts, plan_pts, out_json, args.overwrite,
                save_check=not args.no_check,
            )
            all_ok = all_ok and ok
    else:
        if not args.camera_id:
            print("❌ --camera-id required with --points", file=sys.stderr)
            return 1
        try:
            img_pts, plan_pts = parse_inline_pairs(args.points)
        except ValueError as e:
            print(f"❌ {e}", file=sys.stderr)
            return 1
        ok = process_camera(
            args.camera_id, img_pts, plan_pts, out_json, args.overwrite,
            save_check=not args.no_check,
        )
        all_ok = ok

    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())