# Nuclear-It-hack-2026 — offline multi-camera tracking

Склейка зон ответственности:

| Зона | Папка | Роль в демо |
|------|--------|-------------|
| Person 1 | `person1/` | YOLO11-seg (`yolo11s-seg.pt`) → боксы person |
| Person 2 | `mtmc-person2/` | ByteTrack + SOLIDER / color ReID |
| Person 3 | `person3/` | `GlobalTracker` (межкамерная склейка) + UI |

Собранный прототип: пакет `app/`.

## Быстрый старт

```bash
cd /path/to/Nuclear-It-hack-2026
pip install -r requirements.txt
bash scripts/download_solider_weights.sh   # ~190MB Swin-S Market → weights/

# короткий прогон на 2 камерах (~30 с, каждый 5-й кадр)
python -m app.pipeline --start-sec 0 --max-sec 30 --stride 5 --cameras camera_new_1 camera_new_2

# UI (из корня репо)
streamlit run app/ui_streamlit.py
```

Окно обработки: `start_sec` / `max_sec` в `app/config.yaml` или CLI `--start-sec` / `--max-sec`.  
Без mp4: `--no-export-mp4`.  
Без SOLIDER: в конфиге `reid.backend: color`.

Видео: `Video/camera_new_*.mkv`. Артефакты: `output/tracks.json`, `output/overlays.json`, `output/videos/*.mp4` (боксы + ID).

## Что делает пайплайн

1. **Детекция** — YOLO11-seg, класс person. В UI только прямоугольники (без заливки маски).
2. **Локальный MOT** — ByteTrack из `mtmc-person2` на каждой камере.
3. **ReID** — SOLIDER Swin-S (Market) на usable кропах; fallback — средний цвет верх/низ.
4. **Склейка** — `person3/GlobalTracker.py`: cosine ReID + цвет + время; закрытие при gap > `T`.
5. **Export** — annotated mp4 с глобальными ID на боксах.
6. **UI** — `st.video` превью + слайдер-инспектор, счётчик и карточка трека.

Конфиг: `app/config.yaml` (`reid`, `start_sec`, `max_sec`, `stride`, `export_mp4`, пороги merge).

## Ограничения демо

- Полные ролики ~20 мин @ 2592×1944 — для ноутбука берите короткое окно и `stride`.
- SOLIDER на CPU медленный; GPU сильно ускоряет.
- Стекло / чужая комната в FOV всё ещё дают ложные детекции — ROI зон отдельно.

## Сложность / модель

**STANDARD.** Для правок UI/обвязки хватит Composer 2.5; пороги `T` / `min_sim` лучше сверять более сильной моделью.
