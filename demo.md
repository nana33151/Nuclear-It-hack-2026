# Demo — Streamlit UI

Оффлайн демо межкамерного трекинга: синхронные камеры, боксы и глобальные ID.

## 1. Окружение

```bash
cd /path/to/Nuclear-It-hack-2026
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Веса YOLO: `person1/models/yolo11s-seg.pt` (уже в репо).

SOLIDER (рекомендуется):

```bash
bash scripts/download_solider_weights.sh   # → weights/swin_s_market.pth (~190MB)
```

Без SOLIDER в `app/config.yaml`: `reid.backend: color`.

Видео: `Video/camera_new_*.mkv`.

## 2. Прогон пайплайна (обязательно до UI)

UI читает готовые артефакты из `output/`. Сначала посчитайте треки:

```bash
# из корня репо; параметры можно взять из app/config.yaml
python -m app.pipeline \
  --cameras camera_new_1 camera_new_2 \
  --start-sec 170 --max-sec 25 --stride 5
```

Или короче (возьмёт окно из конфига):

```bash
python -m app.pipeline --cameras camera_new_1 camera_new_2
```

Без превью-mp4 (быстрее):

```bash
python -m app.pipeline --cameras camera_new_1 camera_new_2 --no-export-mp4
```

Ожидаемые файлы:

| Файл | Назначение |
|------|------------|
| `output/tracks.json` | список треков, `n_tracks`, камеры |
| `output/overlays.json` | боксы + global ID по кадрам |
| `output/videos/*.mp4` | превью с ID (если `export_mp4: true`) |

На CPU + SOLIDER короткий клип (`max-sec` 20–30, `stride` 5–10) — разумный минимум для демо.

## 3. Запуск Streamlit

```bash
cd /path/to/Nuclear-It-hack-2026
source .venv/bin/activate
streamlit run app/ui_streamlit.py
```

Браузер откроется на `http://localhost:8501`.

Если артефактов нет, UI покажет предупреждение с командой пайплайна.

## 4. Что смотреть в интерфейсе

**Сайдбар**

- пути к `tracks.json` / `overlays.json` (по умолчанию `output/…`);
- `track_id` — подсветка одного трека (`0` = все);
- число камер (1–5);
- размер превью.

**Основная зона**

- метрики: число сквозных треков, cross-camera, ReID backend;
- синхронные видео / кадры с боксами и ID;
- таблица треков и карточка выбранного (камеры, интервалы).

## 5. Типичные настройки демо

В `app/config.yaml`:

```yaml
start_sec: 170.0
max_sec: 25.0
stride: 5
confidence: 0.65
reid:
  backend: solider   # или color
merge:
  T: 8.0
  min_sim: 0.55
  max_transition_time: 12.0
```

После смены конфига **перегоните пайплайн**, затем обновите страницу Streamlit (или перезапустите).

## 6. Если что-то не работает

| Проблема | Что сделать |
|----------|-------------|
| «Сначала прогоните пайплайн» | нет `output/tracks.json` — шаг 2 |
| Нет боксов на видео | перегон с `export_mp4: true` без `--no-export-mp4` |
| `SOLIDER weights not found` | `bash scripts/download_solider_weights.sh` или `reid.backend: color` |
| Очень медленно | GPU / уменьшить `max_sec` / увеличить `stride` / `reid.backend: color` |
| Другой порт | `streamlit run app/ui_streamlit.py --server.port 8502` |

## 7. Одной строкой (уже есть output/)

```bash
streamlit run app/ui_streamlit.py
```
