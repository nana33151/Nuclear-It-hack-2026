
## Запуск
### 1. Требования
- Python ≥ 3.10
- 8 GB RAM

### 2. Клонирование и окружение
Распакуйте архив SOLIDER - это ReID модель и ее веса - он ждет вас на диске https://drive.google.com/drive/folders/1z7bkaUkUyJsUFcrJDotyzNJ1t_-WqR6w?usp=sharing


```bash
git clone <repo>
cd Nuclear-It-hack-2026
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> Windows: вместо `source .venv/bin/activate` → `.venv\Scripts\activate`

### 3. Веса

```bash
bash scripts/download_solider_weights.sh
```

### 4. Данные
Положить 5 видео в `Video/` (см. `config/example.env`).

## Запуск

Собрать UI из готовых треков:

```bash
streamlit run app/ui_streamlit.py
# http://localhost:8501
```
Запустить обработку заново со своими параметрами:

```bash
python -m app.pipeline --start-sec 175 --max-sec 15 --stride 5 \
    --cameras camera_new_2 camera_new_3
```

Опции:
- `--no-export-mp4` — только json (~30 с вместо ~2 мин)
- `--cameras ...` — только указанные камеры
- `--stride N` — каждый N-й кадр


Для изменения остальных параметров запуска используйте app/config.yaml



# src — карта кода



## Модули

### `app/` — собранный прототип (оркестрация)

| Файл | Что |
|---|---|
| `pipeline.py` | точка входа: детекция → MOT → merge → export → запись json |
| `detect.py` | YOLO11-seg, `iter_video_sampled`, `detect_people` |
| `local_track.py` | ByteTrack + SOLIDER, split/stitch |
| `merge.py` | `to_global_tracks`, `build_overlays`, парсинг transitions |
| `reid_solider.py` | обёртка SOLIDER, fallback на цвет |
| `draw.py` | общие примитивы отрисовки (bbox + ID) |
| `export_video.py` | рендер annotated mp4 |
| `projectors_loader.py` | загрузка `Projector` из person2 без импорта пакета |
| `bev_state.py` | BEV-логика UI: `BBoxEvent`, `BEVEvent`, маппинг local→global |
| `bev_render.py` | Plotly-отрисовка плана |
| `ui_streamlit.py` | 4 вкладки UI |
| `paths.py` | ROOT и загрузка классов из person1/2/3 |
| `config.yaml` | все параметры пайплайна |

### `person1/` — детекция

- `perception/cli.py`, `pipeline.py`, `handoff.py` — CLI-обёртка, экспорт детекций.
- `models/yolo11s-seg.pt` — веса.

### `mtmc-person2/` — локальный MOT + ReID + геометрия

- `src/mot/` — ByteTrack (`byte_tracker_core.py`, `bytetrack.py`).
- `src/reid/` — SOLIDER + color fallback.
- `src/geometry/projectors.py` — 5 гомографий image→plan.
- `src/geometry/office_model.py` — план помещения + зона зеркала.
- `src/geometry/filter.py` — mirror filter.
- `src/tracklets/` — build/split треклетов (person2-side).
- `src/contract.py` — dataclass-контракты.

### `person3/` — глобальный трекер

- `GlobalTracker.py` — lifecycle + Hungarian + BEV-гейт.
- `test_lifecycle.py` — smoke-тесты.

## Скрипт: `tracks.json` → CSV

В `artifacts/parse_tracks.py` лежит генератор плоских CSV:

```bash
python artifacts/parse_tracks.py output/tracks.json artifacts/
# → artifacts/tracks.csv, artifacts/segments.csv
