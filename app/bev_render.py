"""Plotly renderers + OpenCV helpers for the BEV UI tab. No Streamlit here."""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


CAM_COLORS = {
    "camera_new_1": "#e74c3c",
    "camera_new_2": "#27ae60",
    "camera_new_3": "#2980b9",
    "camera_new_4": "#8e44ad",
    "camera_new_5": "#d35400",
}


def _cam_color(cam: str) -> str:
    return CAM_COLORS.get(cam, "#7f8c8d")


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------
# OpenCV: camera frame with bbox + bottom-center anchor
# ---------------------------------------------------------------------

def draw_bbox_anchor(
    frame: np.ndarray,
    bbox: Sequence[float],
    bbox_color: tuple[int, int, int] = (0, 0, 255),
    anchor_color: tuple[int, int, int] = (0, 220, 255),
) -> np.ndarray:
    """Draw bbox + big dot at bottom-center on a BGR frame."""
    out = frame.copy()
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(out, (x1, y1), (x2, y2), bbox_color, 3)
    cx, cy = (x1 + x2) // 2, y2
    cv2.circle(out, (cx, cy), 12, (0, 0, 0), -1)
    cv2.circle(out, (cx, cy), 9, anchor_color, -1)
    label = f"anchor ({cx},{cy})"
    for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        cv2.putText(out, label, (cx + 14 + dx, cy - 8 + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, label, (cx + 14, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, anchor_color, 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------
# Plotly: BEV map
# ---------------------------------------------------------------------

def render_bev_map(
    footprints: dict[str, list[tuple[float, float]]],
    active: Sequence,
    trail: Sequence,
    selected_gid: int,
    title: str = "BEV",
) -> "go.Figure":
    fig = go.Figure()

    for cam, pts in footprints.items():
        if len(pts) < 3:
            continue
        xs = [p[0] for p in pts] + [pts[0][0]]
        ys = [p[1] for p in pts] + [pts[0][1]]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=_cam_color(cam), width=1.5),
            fill="toself",
            fillcolor=_hex_to_rgba(_cam_color(cam), 0.10),
            name=f"{cam} (FOV≈)",
            hoverinfo="skip",
        ))

    if trail:
        fig.add_trace(go.Scatter(
            x=[e.x for e in trail], y=[e.y for e in trail],
            mode="lines+markers",
            line=dict(color="rgba(241, 196, 15, 0.85)", width=3),
            marker=dict(size=5),
            name=f"trail id={selected_gid}" if selected_gid else "trail",
            hoverinfo="skip",
        ))

    if active:
        labels = [str(e.global_id) for e in active]
        cams = [e.cam for e in active]
        colors = [
            "#e74c3c" if e.global_id == selected_gid else _cam_color(e.cam)
            for e in active
        ]
        fig.add_trace(go.Scatter(
            x=[e.x for e in active], y=[e.y for e in active],
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=12, color=colors, line=dict(width=2, color="white")),
            customdata=cams,
            hovertemplate="ID %{text}<br>cam=%{customdata}<br>"
                          "x=%{x:.2f} м, y=%{y:.2f} м<extra></extra>",
            name="active",
            showlegend=False,
        ))

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        height=600,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="X, м", yaxis_title="Y, м",
        title=title,
        legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
    )
    return fig


# ---------------------------------------------------------------------
# Plotly: plan view with bbox outline + anchor
# ---------------------------------------------------------------------

def render_plan_with_anchor(
    plan_xy: tuple[float, float],
    bbox_xyxy: Sequence[float],
    proj,
    footprint: Optional[list[tuple[float, float]]],
    cam: str,
) -> "go.Figure":
    fig = go.Figure()

    if footprint and len(footprint) >= 3:
        xs = [p[0] for p in footprint] + [footprint[0][0]]
        ys = [p[1] for p in footprint] + [footprint[0][1]]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=_cam_color(cam), width=2),
            fill="toself",
            fillcolor=_hex_to_rgba(_cam_color(cam), 0.10),
            name=f"{cam} (FOV≈)",
            hoverinfo="skip",
        ))

    x1, y1, x2, y2 = bbox_xyxy
    sampled = [
        (x1, y1), ((x1 + x2) / 2, y1), (x2, y1),
        (x1, (y1 + y2) / 2), (x2, (y1 + y2) / 2),
        (x1, y2), ((x1 + x2) / 2, y2), (x2, y2),
    ]
    proj_pts = []
    for pt in sampled:
        try:
            proj_pts.append(proj.image_to_plan(pt))
        except Exception:
            proj_pts.append(None)
    if all(p is not None for p in proj_pts):
        pts = [p for p in proj_pts if p is not None]
        order = [0, 1, 2, 4, 7, 6, 5, 3, 0]   # trace outline
        fig.add_trace(go.Scatter(
            x=[pts[i][0] for i in order],
            y=[pts[i][1] for i in order],
            mode="lines",
            line=dict(color="#27ae60", width=2, dash="dot"),
            name="projected bbox outline",
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=[plan_xy[0]], y=[plan_xy[1]],
        mode="markers+text",
        text=[f"({plan_xy[0]:.2f}, {plan_xy[1]:.2f}) м"],
        textposition="top center",
        marker=dict(size=16, color="#f1c40f", line=dict(width=2, color="black")),
        name="bottom-center → plan",
        hovertemplate=f"plan=({plan_xy[0]:.2f}, {plan_xy[1]:.2f}) м<extra></extra>",
    ))

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="X, м", yaxis_title="Y, м",
        title=f"{cam}: проекция bbox на план",
        legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
    )
    return fig