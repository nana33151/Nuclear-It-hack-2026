# Решение и алгоритм

## Контекст

В помещении 5 камер, их записи идут синхронного. Ручная обработка записей с камер занимает слишком много времени, а использование простых ByteTrack и не aug-learned ReID не подходит для сложных пересечений и ракурсов. Задача требует сопоставить людей и их положения в пространстве для синхронной обработки 5 камер

## Схема

Видео → YOLO11-seg → ByteTrack+SOLIDER → GlobalTracker → BEV+UI
(person1) (mtmc-person2,app) (person3) (app)
text


## Слой 1 — детекция (YOLO11-seg)

- Один класс `person` (COCO 0).
- `imgsz`: 1024 для CPU
- `conf=0.45` — баланс между recall в коридоре и ложными срабатываниями.
- На каждый бокс: `bbox`, `score`, `color_top/bottom`, `usable_for_reid`.
- `usable_for_reid = score ≥ 0.25 && h ≥ 40`.

Почему YOLO: one-stage, real-time, сегментация + детекция в одной голове.
Альтернативы (Faster-RCNN, DETR) — медленнее в 3–5× на CPU, что для 5 камер
критично.

## Слой 2 — локальный MOT (ByteTrack + ReID)

- ByteTrack: Kalman + Hungarian по IoU, эвристика, без обучения.
- ReID-терм подмешивается в cost (`reid_weight=0.4`).
- `track_buffer=15` (~3 с при stride 5) — короткий LOST.
- Split by ReID: режем трек при `1 − cos > 0.55` **или** при смене размерности
  эмбеддинга (Solider 768 ↔ color 32).
- Stitch same-camera: склейка осколков при `cos ≥ 0.45` и `gap < 8 с`.
- Инвариант: `stitch_min_sim ≤ 1 − split_reid_thresh`.

ReID: SOLIDER Swin-S (768-dim, Market1501). При недоступности бокса для ReID опираемся на цвет (mean RGB top/bottom,
32-dim).

## Слой 3 — глобальный трекер (lifecycle + Hungarian)

Hungarian алгоритм нужен для оптимального назначения id людям при ограничениях, заданных ниже.
lifecycle - эвристика, позволяющая с помощью unconfident-состояний проводить детекцию отложенно

Машина состояний: `UNCONFIDENT → CONFIDENT → LOST → DIED`.

- **Confirm:** `n_segments ≥ 2` или `duration ≥ 0.2 с`.
- **Lost:** разрыв; ждёт `lost_timeout_sec` (10 с) до смерти.
- **Died:** `now − coverage_end > lost_timeout`.

На каждом тике — Hungarian по новым сегментам. Скор:

score = (w_reid·cos + w_color·color_sim + w_time·time_sim + w_bev·bev_sim)
/ (w_reid + w_color + w_time + w_bev)
text


**Жесткие отсечения:**

1. `same_cam_overlap` — нельзя клеить пересекающиеся на одной камере.
2. **Topology:** `(src_cam, dst_cam)` должно быть в `transitions`.
3. **`min_time`:** если > 0 и трек не LOST — reject (камеры не видят одновременно).
4. **BEV feasibility:** `dist ≤ bev_max_speed · travel`.
5. **`min_sim=0.55`.**
6. **Same-cam ReID gate:** при повторном входе в ту же камеру `cos ≥ 0.6`.

Каждый reject — в `output/tracker_decisions.json` (`debug: true`).

## Слой 4 — BEV (гомография + mirror filter)
Технология BEV работает с помощью гомографической проекции и позволяет связать людей с их положением на плоскости.

- `H: image_px → plan_m`, 5 матриц 3×3.
- Якорь = `bbox_bottom_center` (ноги).
- **Mirror filter:** `OfficeModel` + `filter_detections_by_mirror` отбрасывают
  детекции, попадающие в зону зеркала.
  Активен только для cam2 (`sees_mirror=True`).
- **BEV-гейт** в `GlobalTracker` - эвристика средней допустимой скорости вместо распределения скоростей: `bev_sim = exp(−dist / σ)`, где
  `σ = bev_sigma_scale · bev_typical_speed · travel`. Если
  `dist > bev_max_speed · travel` — reject.

## Слой 5 — UI

Streamlit, 4 вкладки: **Камеры** (синхронный плеер), **BEV+Live** (план),
**Проекция** (один трек + якорь), **Треки** (таблица).

`bev_state.bboxes_for_cam_at` резолвит `(camera_id, local_id) → global_id`
по `tracks.json`. `local_id` кладётся в `overlays.json` на этапе `build_overlays`.

## Ключевые технические решения

| Решение | Альтернатива | Почему так |
|---|---|---|
| YOLO11-seg | Faster-RCNN, DETR | скорость на CPU, one-stage |
| ByteTrack | BoT-SORT, OC-SORT | простота, скорость интеграции |
| SOLIDER Swin-S | OSNet, ResNet50 | качество на Market1501 |
| Гомография | PnP, bundle adjustment | простота, 4 точки на камеру |
| Lifecycle + Hungarian | жадный матчинг | детерминированный, объяснимый |
| Offline batch | online | нет требования real-time в задаче |

# Перенос проекта (handoff)

Что нужно знать человеку, который подхватывает репозиторий.

## Среда

- Python ≥ 3.10, тестировалось на 3.12.
- Linux x86-64 / macOS arm64.
- Опционально: CUDA или MPS.
- Веса: `person1/models/yolo11s-seg.pt` (~20 MB), `weights/swin_s_market.pth`
  (~190 MB). Скачиваются скриптом `scripts/download_solider_weights.sh`.
- Внешний код: `third_party/SOLIDER-REID/` — вендоренный репозиторий, не
  трогаем. Обновляется отдельно.

## Точки входа

| Команда | Что делает |
|---|---|
| `python -m app.pipeline` | полный прогон: детекция → MOT → merge → export → json |
| `streamlit run app/ui_streamlit.py` | UI |
| `python person3/test_lifecycle.py` | smoke-тесты `GlobalTracker` |
| `python artifacts/parse_tracks.py output/tracks.json artifacts/` | CSV из tracks.json |

## Что где лежит

- `app/config.yaml` — все параметры пайплайна. Меняешь — перезапускаешь пайплайн.
- `person3/GlobalTracker.py` — глобальный трекер. Ключевые методы:
  - `__init__` — параметры (`lost_timeout_sec`, `same_cam_reid_min_sim`, `projectors`);
  - `_match_score_explained` — гейты + скор;
  - `_seg_plan_pos` — BEV-позиция сегмента (принимает dict с `bbox`);
  - `_lost_timeout` — таймаут LOST→DIED.
- `app/merge.py` — `to_global_tracks`, `build_overlays`. Здесь решается,
  что попадёт в `overlays.json` (в т.ч. `local_id` для UI).
- `app/bev_state.py` — маппинг `(camera_id, local_id) → global_id` для UI.

## Что проверить при первом запуске

```bash
# 1. Веса на месте
ls person1/models/yolo11s-seg.pt weights/swin_s_market.pth

# 2. Видео на месте
ls Video/camera_new_*.mkv

# 3. Пайплайн работает (без mp4, 2 камеры, 15 с)
python -m app.pipeline --no-export-mp4 \
    --start-sec 175 --max-sec 15 --stride 5 \
    --cameras camera_new_2 camera_new_3

# 4. tracks.json осмысленный
jq '.n_tracks, .n_cross_camera, .tracks[] | .track_id' output/tracks.json

# 5. UI открывается
streamlit run app/ui_streamlit.py
