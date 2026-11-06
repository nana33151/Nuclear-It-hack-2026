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
| SOLIDER | слот под веса + torch (ещё не граф модели) |

## Запуск

```bash
cd C:\Users\tcher\Projects\mtmc-person2
py -3 -m pip install -r requirements.txt
py -3 scripts/run_tracklets.py
py -3 scripts/smoke_split.py
```

Выход: `data/output/tracklets.jsonl` для Person 3.

## SOLIDER позже

1. Python 3.10–3.11 + torch  
2. Веса Swin-B в `weights/solider_swin_b.pth`  
3. В конфиге: `reid.backend: solider`  
4. Допилить `src/reid/solider.py` под inference из [SOLIDER-REID](https://github.com/tinyvision/SOLIDER-REID)

## Граница роли

Не делает: YOLO/SCHP, межкамерную склейку, UI.
