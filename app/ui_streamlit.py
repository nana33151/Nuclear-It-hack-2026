"""Streamlit demo: sync multi-cam playback + global track IDs (bbox only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.draw import draw_boxes, nearest_overlay, resize_max, scale_items
from app.paths import ROOT

st.set_page_config(layout="wide", page_title="MTMC Demo")


@st.cache_data
def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_frame_at(path: Path, t_sec: float) -> tuple[object | None, int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, 0, 0.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_idx = int(round(t_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None, frame_idx, fps
    return frame, frame_idx, fps


def main() -> None:
    st.title("Offline Multi-Camera Tracking")
    st.caption("YOLO11-seg boxes · ByteTrack+SOLIDER ReID · GlobalTracker cross-cam merge")

    with st.sidebar:
        st.header("Данные")
        tracks_file = st.text_input("tracks.json", str(ROOT / "output" / "tracks.json"))
        overlays_file = st.text_input("overlays.json", str(ROOT / "output" / "overlays.json"))
        selected_track = st.number_input("track_id (0 = все)", min_value=0, value=0, step=1)
        show_all = st.checkbox("Показывать все ID", value=True)
        max_side = st.slider("Макс. сторона превью", 480, 1600, 960, 40)
        cam_limit = st.slider("Сколько камер показать", 1, 5, 2)

    if not Path(tracks_file).exists() or not Path(overlays_file).exists():
        st.warning(
            "Сначала прогоните пайплайн:\n\n"
            "`python -m app.pipeline --start-sec 0 --max-sec 30 --stride 5 "
            "--cameras camera_new_1 camera_new_2`"
        )
        return

    data = load_json(tracks_file)
    over = load_json(overlays_file)
    tracks = data.get("tracks", [])
    cameras = over.get("cameras") or data.get("cameras") or {}
    overlays = over.get("overlays", {})

    c1, c2, c3 = st.columns(3)
    c1.metric("Сквозных треков", data.get("n_tracks", len(tracks)))
    c2.metric("Cross-camera", data.get("n_cross_camera", 0))
    win = data.get("window", {})
    c3.metric("Окно (сек)", f"{win.get('start_sec', 0)}+{win.get('max_sec', '?')}")

    cam_ids = list(cameras.keys())[:cam_limit]
    stride = int(win.get("stride") or 5)
    tol = max(stride * 2, 8)

    # Annotated mp4 previews (playable)
    preview_cams = [
        cid
        for cid in cam_ids
        if cameras.get(cid, {}).get("preview_mp4")
        and (ROOT / cameras[cid]["preview_mp4"]).exists()
    ]
    if preview_cams:
        st.subheader("Preview MP4")
        vcols = st.columns(len(preview_cams))
        for i, cam_id in enumerate(preview_cams):
            mp4_path = ROOT / cameras[cam_id]["preview_mp4"]
            with vcols[i]:
                st.caption(cam_id)
                st.video(str(mp4_path))

    t_min = float(win.get("start_sec", 0.0))
    t_max = t_min + float(win.get("max_sec", 60.0))
    st.subheader("Кадр-инспектор")
    t = st.slider("Время (сек)", t_min, t_max, t_min, 0.2)

    cols = st.columns(len(cam_ids) or 1)
    for i, cam_id in enumerate(cam_ids):
        meta = cameras[cam_id]
        path = ROOT / meta["path"]
        with cols[i]:
            st.subheader(cam_id)
            frame, frame_idx, fps = read_frame_at(path, t)
            if frame is None:
                st.error(f"Не удалось прочитать {path.name}")
                continue
            items = nearest_overlay(overlays.get(cam_id, {}), frame_idx, tol_frames=tol)
            preview, scale = resize_max(frame, max_side)
            items = scale_items(items, scale)
            drawn = draw_boxes(preview, items, int(selected_track), show_all)
            st.image(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"frame {frame_idx} · {len(items)} boxes · tol={tol}")

    st.subheader("Список треков")
    if tracks:
        rows = [
            {
                "track_id": tr["track_id"],
                "t_start": round(tr["t_start"], 2),
                "t_end": round(tr["t_end"], 2),
                "cameras": ", ".join(tr["cameras"]),
                "n_segments": len(tr["segments"]),
            }
            for tr in tracks
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    if selected_track > 0:
        tr = next((x for x in tracks if x["track_id"] == selected_track), None)
        if tr:
            st.subheader(f"Трек {selected_track}")
            st.write(f"Время: **{tr['t_start']:.1f} – {tr['t_end']:.1f}** с")
            st.write(f"Камеры: {', '.join(tr['cameras'])}")
            st.json(tr["segments"])
        else:
            st.info("Такого track_id нет")


if __name__ == "__main__":
    main()
