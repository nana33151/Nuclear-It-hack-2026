# app.py
import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import json

st.set_page_config(layout="wide", page_title="Multi-Cam Tracker")

st.title("Offline Multi-Camera Tracking — Demo")

# --- боковая панель ---
with st.sidebar:
    st.header("Управление")
    video_dir = st.text_input("Папка с видео", "data/videos")
    tracks_file = st.text_input("JSON с глобальными треками", "output/tracks.json")
    selected_track = st.number_input("Показать track_id", min_value=0, value=0)
    show_all = st.checkbox("Показывать все ID", value=True)

# Загрузка треков
@st.cache_data
def load_tracks(path):
    with open(path) as f:
        data = json.load(f)
    return data

try:
    tracks = load_tracks(tracks_file)
    n_tracks = len(tracks)
    st.metric("Уникальных сквозных треков", n_tracks)
except Exception:
    tracks = []
    st.warning("Треки ещё не загружены")

# Выбор камер
cam_files = sorted(Path(video_dir).glob("*.mp4")) if Path(video_dir).exists() else []
cols = st.columns(min(2, len(cam_files) or 2))

# Простая синхронизация по времени (для демо)
t = st.slider("Время (сек)", 0.0, 120.0, 0.0, 0.1)

for i, col in enumerate(cols):
    if i >= len(cam_files):
        break
    with col:
        st.subheader(cam_files[i].stem)
        # Здесь нужно открыть видео, seek на t, нарисовать bbox + track_id
        # (псевдокод — в реальном коде используй cv2 + st.image)
        st.info(f"Видео {cam_files[i].name} @ t={t:.1f}s")
        # TODO: overlay detections + global IDs

# Карточка выбранного трека
if selected_track > 0 and tracks:
    tr = next((t for t in tracks if t["track_id"] == selected_track), None)
    if tr:
        st.subheader(f"Трек {selected_track}")
        st.write(f"Время: {tr['t_start']:.1f} – {tr['t_end']:.1f} с")
        st.write(f"Камеры: {', '.join(tr['cameras'])}")
        st.json(tr["segments"])