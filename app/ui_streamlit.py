"""Streamlit demo: sync multi-cam playback + BEV (live bbox → plan projection)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.draw import draw_boxes, resize_max, scale_items
from app.paths import ROOT
from app import bev_render, bev_state
from app.projectors_loader import load_projectors, project_bbox


st.set_page_config(layout="wide", page_title="MTMC Demo (BEV)")


# ---------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------

@st.cache_data
def _load_json(path: str, mtime: float) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_json(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    return _load_json(str(p), p.stat().st_mtime)


@st.cache_resource
def _get_projectors(cams: tuple[str, ...]) -> dict:
    return load_projectors(cams)


def get_projectors(cams: list[str]) -> dict:
    return _get_projectors(tuple(sorted(cams)))


@st.cache_data
def _compute_bev_events(
    overlays_path: str, overlays_mtime: float,
    tracks_path: str, tracks_mtime: float,
    cams: tuple[str, ...],
) -> list:
    overlays = _load_json(overlays_path, overlays_mtime)
    tracks = _load_json(tracks_path, tracks_mtime)
    projs = load_projectors(cams)
    bbox_events = list(bev_state.iter_bbox_events(overlays, tracks))
    return bev_state.project_events(bbox_events, projs)


def compute_bev_events(overlays_file, tracks_file, cams) -> list:
    op, tp = Path(overlays_file), Path(tracks_file)
    if not op.exists() or not tp.exists() or not cams:
        return []
    return _compute_bev_events(
        str(op), op.stat().st_mtime,
        str(tp), tp.stat().st_mtime,
        tuple(sorted(cams)),
    )


# ---------------------------------------------------------------------
# Video reader
# ---------------------------------------------------------------------

def read_frame_at(path: Path, t_sec: float) -> tuple[object | None, int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, 0, 0.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_idx = int(round(t_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return (frame, frame_idx, fps) if ok else (None, frame_idx, fps)


# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------

def tab_cameras(overlays_doc, tracks_doc, cameras, cam_ids,
                t, tol, selected_track, show_all, max_side):
    cols = st.columns(len(cam_ids) or 1)
    for i, cam_id in enumerate(cam_ids):
        meta = cameras[cam_id]
        with cols[i]:
            st.subheader(cam_id)
            frame, frame_idx, _ = read_frame_at(ROOT / meta["path"], t)
            if frame is None:
                st.error(f"Не читается {meta.get('path')}")
                continue
            items = bev_state.bboxes_for_cam_at(overlays_doc, cam_id, t, tol, tracks_doc)
            draw_items = [
                {
                    "track_id": it["global_id"],
                    "bbox": it["bbox"],
                    "score": it.get("score", 0.0),
                }
                for it in items if it["global_id"] is not None
            ]
            preview, scale = resize_max(frame, max_side)
            drawn = draw_boxes(preview, scale_items(draw_items, scale),
                               int(selected_track), show_all)
            st.image(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB), use_container_width=True)
            gids = sorted({it["track_id"] for it in draw_items})
            st.caption(
                f"frame {frame_idx} · {len(draw_items)}/{len(items)} boxes "
                f"(confirmed/unmapped) · global IDs: {gids or '—'}"
            )


def tab_bev(events, footprints, selected_track, t, tol, tail_sec):
    if not bev_render.HAS_PLOTLY:
        st.warning("Установите `plotly`: `pip install plotly`")
        return
    if not events:
        st.warning("BEV-события не построены. Проверьте, что камеры в overlays "
                   "имеют гомографии в Projector.HOMOGRAPHIES (cam1..cam5).")
        return
    active = bev_state.active_at(events, t, tol)
    trail = (bev_state.trail_for(events, selected_track, t, tail_sec)
             if selected_track > 0 else [])
    fig = bev_render.render_bev_map(footprints, active, trail, int(selected_track))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"t={t:.1f}s · active points: {len(active)} · "
        f"trail: {len(trail)} · selected: {selected_track or '—'}"
    )


def tab_projection(overlays_doc, tracks_doc, cameras, cam_ids, projectors,
                   footprints, selected_track, t, tol, max_side):
    if not bev_render.HAS_PLOTLY:
        st.warning("Установите `plotly`.")
        return
    if selected_track <= 0:
        st.info("Выберите track_id в сайдбаре (или на вкладке «Треки»)")
        return

    hits: list[tuple[str, dict]] = []
    for cam in cam_ids:
        for it in bev_state.bboxes_for_cam_at(overlays_doc, cam, t, tol, tracks_doc):
            if it["global_id"] == selected_track:
                hits.append((cam, it))
    if not hits:
        st.info(f"Трек {selected_track} не виден в t={t:.1f}s (± {tol:.1f}s)")
        return

    for cam, item in hits:
        proj = projectors.get(cam)
        if proj is None:
            st.warning(f"Нет проектора для {cam}")
            continue
        plan_xy = project_bbox(proj, item["bbox"])
        if plan_xy is None:
            st.warning(f"Не удалось спроецировать bbox ({cam})")
            continue

        st.subheader(f"{cam} · global ID {selected_track}")
        c1, c2 = st.columns([1.1, 1])
        with c1:
            frame, frame_idx, _ = read_frame_at(ROOT / cameras[cam]["path"], t)
            if frame is None:
                st.error(f"Не читается {cam}")
                continue
            preview, _ = resize_max(frame, max_side)
            drawn = bev_render.draw_bbox_anchor(preview, item["bbox"])
            st.image(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB),
                     use_container_width=True,
                     caption=f"frame {frame_idx} · bottom-center = anchor")
        with c2:
            fig = bev_render.render_plan_with_anchor(
                plan_xy=plan_xy,
                bbox_xyxy=item["bbox"],
                proj=proj,
                footprint=footprints.get(cam),
                cam=cam,
            )
            st.plotly_chart(fig, use_container_width=True)
        st.caption(f"plan = ({plan_xy[0]:.2f}, {plan_xy[1]:.2f}) м")


def tab_tracks(tracks, selected_track):
    if tracks:
        rows = [
            {
                "track_id": tr["track_id"],
                "t_start": round(tr["t_start"], 2),
                "t_end": round(tr["t_end"], 2),
                "cameras": ", ".join(tr["cameras"]),
                "n_segments": len(tr["segments"]),
                "state": tr.get("state", ""),
                "close_reason": tr.get("close_reason", ""),
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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    st.title("Offline Multi-Camera Tracking — BEV live")
    st.caption("YOLO11-seg · ByteTrack+SOLIDER · GlobalTracker · floor-plan projection")

    with st.sidebar:
        st.header("Данные")
        tracks_file = st.text_input("tracks.json", str(ROOT / "output" / "tracks.json"))
        overlays_file = st.text_input("overlays.json", str(ROOT / "output" / "overlays.json"))
        selected_track = st.number_input("track_id (0 = все)", min_value=0, value=0, step=1)
        show_all = st.checkbox("Показывать все ID", value=True)
        max_side = st.slider("Макс. сторона превью", 480, 1600, 960, 40)
        cam_limit = st.slider(
            "Сколько камер на вкладке «Камеры»", 1, 5, 2,
            help="BEV-вкладки всегда показывают все камеры с известными гомографиями",
        )

        st.header("BEV (live)")
        enable_bev = st.checkbox("Включить live-проекцию bbox → BEV", value=True)
        tol_sec = st.slider("Допуск по времени, с", 0.05, 2.0, 0.5, 0.05)
        tail_sec = st.slider("Хвост траектории, с", 0.0, 30.0, 5.0, 0.5)

    overlays_doc = load_json(overlays_file)
    tracks_doc = load_json(tracks_file)
    if overlays_doc is None or tracks_doc is None:
        st.warning(
            "Нужны `output/tracks.json` и `output/overlays.json`.\n\n"
            "Запустите пайплайн, например:\n"
            "`python -m app.pipeline --start-sec 175 --max-sec 15 --stride 5`"
        )
        return

    tracks = tracks_doc.get("tracks", [])
    cameras = overlays_doc.get("cameras") or tracks_doc.get("cameras") or {}
    window = tracks_doc.get("window") or {}
    all_cam_ids = list(cameras.keys())
    cam_ids_view = all_cam_ids[:cam_limit]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Треков", tracks_doc.get("n_tracks", len(tracks)))
    c2.metric("Cross-camera", tracks_doc.get("n_cross_camera", 0))
    c3.metric("Окно, с", f"{window.get('start_sec', 0)}+{window.get('max_sec', '?')}")
    c4.metric("Камер", len(cameras))

    projectors = get_projectors(all_cam_ids) if enable_bev else {}
    events = compute_bev_events(overlays_file, tracks_file, all_cam_ids) if enable_bev else []
    footprints = bev_state.footprint_per_camera(events) if events else {}

    t_min = float(window.get("start_sec", 0.0))
    t_max = t_min + float(window.get("max_sec", 60.0))
    if t_max <= t_min:
        t_max = t_min + 1.0

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📷 Камеры", "🗺 BEV + Live", "🔗 Проекция (bbox → план)", "📋 Треки"]
    )

    with tab1:
        t1 = st.slider("Время (сек)", t_min, t_max, t_min, 0.2, key="cam_t")
        tab_cameras(overlays_doc, tracks_doc, cameras, cam_ids_view,
                    t1, tol_sec, int(selected_track), show_all, max_side)

    with tab2:
        if not enable_bev:
            st.info("Включите «Live-проекция» в сайдбаре.")
        else:
            t2 = st.slider("Время BEV (сек)", t_min, t_max, t_min, 0.2, key="bev_t")
            tab_bev(events, footprints, int(selected_track), t2, tol_sec, tail_sec)

    with tab3:
        if not enable_bev:
            st.info("Включите «Live-проекция» в сайдбаре.")
        else:
            t3 = st.slider("Время проекции (сек)", t_min, t_max, t_min, 0.2, key="proj_t")
            tab_projection(overlays_doc, tracks_doc, cameras, all_cam_ids,
                           projectors, footprints,
                           int(selected_track), t3, tol_sec, max_side)

    with tab4:
        tab_tracks(tracks, int(selected_track))


if __name__ == "__main__":
    main()