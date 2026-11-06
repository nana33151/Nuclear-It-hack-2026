# mtmc-person2

Роль **человека 2**: однокамерный MOT + ReID-признаки → локальные треклеты для человека 3.

## Подключить Person 1

Он кладёт JSONL по `schemas/detection.schema.json`. В `configs/default.yaml`:

```yaml
detections:
  source: jsonl
  path: data/input/detections.jsonl
```

Обязательные поля: `camera_id`, `frame_idx`, `timestamp`, `bbox`, `score`, `usable_for_reid`.  
Полезно сразу: `color_top` / `color_bottom`, `crop_path` (для SOLIDER).

## Что умеет сейчас

| Компонент | Статус |
|---|---|
| ByteTrack (vendored, без BoxMOT/AGPL) | работает |
| Stub MOT | запасной |
| ReID `color_mock` из цветов P1 | работает |
| Резка по дисперсии ReID | работает |
| SOLIDER | `third_party/SOLIDER-REID` + `weights/swin_s_market.pth` |

## Запуск

```bash
cd C:\Users\tcher\Projects\mtmc-person2
py -3 -m pip install -r requirements.txt
py -3 scripts/run_tracklets.py
py -3 scripts/smoke_split.py
```

Выход: `data/output/tracklets.jsonl` для Person 3.

## SOLIDER

1. `bash scripts/download_solider_weights.sh` (из корня репо)  
2. В `configs/default.yaml`: `reid.backend: solider`  
3. Inference: `src/reid/solider.py` → `app.reid_solider.SoliderReID` ([SOLIDER-REID](https://github.com/tinyvision/SOLIDER-REID))

## Граница роли

Не делает: YOLO/SCHP, межкамерную склейку, UI.
