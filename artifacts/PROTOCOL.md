# Протокол обмена данными между модулями

Контракты JSON и dataclass'ов, которые связывают слои пайплайна.
Изменение поля = breaking change: править синхронно `contract.py` (person2),
`GlobalTracker.py` (person3), `merge.py` и `bev_state.py` (app).

## Поток

Video → [1] Detection → [2] Tracklet → [3] GlobalTrack → [4] Overlay → UI
(person1) (person2) (person3) (app)
text


---

## 1. Detection — person1 → person2

Источник: `app/detect.py::detect_people`, `person1/perception/`.
Схема: `mtmc-person2/schemas/detection.schema.json`.

```json
{
  "camera_id": "camera_new_2",
  "frame_idx": 4385,
  "timestamp": 175.4,
  "bbox": [1454.6, 163.2, 1587.4, 504.2],
  "score": 0.676,
  "usable_for_reid": true,
  "color_top": [0.42, 0.38, 0.35],
  "color_bottom": [0.21, 0.20, 0.19],
  "reid": [0.012, -0.033, ...],
  "det_id": "cam2_4385_0"
}

2. Tracklet — person2 → person3

Источник: mtmc-person2/src/tracklets/build.py, app/local_track.py.
Схема: mtmc-person2/schemas/tracklet.schema.json.
json

{
  "camera_id": "camera_new_3",
  "local_id": 1,
  "t_start": 177.6,
  "t_end": 188.8,
  "boxes": [
    {
      "frame_idx": 4440,
      "timestamp": 177.6,
      "bbox": [971.6, 108.2, 1035.0, 306.2],
      "score": 0.665,
      "usable_for_reid": true
    }
  ],
  "reid_emb": [0.011, -0.027, ...],
  "color_top": [0.42, 0.38, 0.35],
  "color_bottom": [0.21, 0.20, 0.19],
  "representative_frame": 4460,
  "split_reason": null,
  "quality": {"n_frames": 56, "n_reid_crops": 42}
}

3. GlobalTrack — person3 → output

Источник: GlobalTracker._to_global, app/merge.py::to_global_tracks.
Кладётся в output/tracks.json.
json

{
  "track_id": 4,
  "t_start": 177.6,
  "t_end": 189.6,
  "cameras": ["camera_new_2", "camera_new_3"],
  "close_reason": "gap_exceeded",
  "state": "died",
  "segments": [
    {
      "camera_id": "camera_new_3",
      "local_id": 1,
      "t_start": 177.6,
      "t_end": 188.8,
      "n_boxes": 58,
      "plan_x": 3.14,
      "plan_y": 12.72
    }
  ]
}

4. Overlay — output → UI

Источник: app/merge.py::build_overlays. Кладётся в output/overlays.json.
json

{
  "cameras": {
    "camera_new_2": {"path": "Video/camera_new_2.mkv", "fps": 25.0}
  },
  "overlays": {
    "camera_new_2": {
      "4385": [
        {
          "track_id": 1,
          "local_id": 1,
          "camera_id": "camera_new_2",
          "bbox": [1454.6, 163.2, 1587.4, 504.2],
          "score": 0.676,
          "timestamp": 175.4
        }
      ]
    }
  },
  "n_tracks": 4
}

5. Конфиг app/config.yaml

Не протокол в строгом смысле, но контракт между кодом и оркестратором.
Ключевые поля с инвариантами:
Секция	Поле	Инвариант
mot	stitch_min_sim	≤ 1 − split_reid_thresh
mot	track_buffer	кадров, не секунд; для stride=5 и FPS=25 реальный таймаут = track_buffer · stride / 25
merge	transitions[(src, dst)].min_time	> 0 ⇒ камеры не видят человека одновременно
merge	bev_max_speed	м/с, физический предел; dist > bev_max_speed · travel ⇒ reject
merge	lost_timeout_sec	если null, fallback на max max_time по исходящим рёбрам
merge	enable_bev	true ⇒ projectors передаются в GlobalTracker
merge	debug	true ⇒ пишется output/tracker_decisions.json

6. Лог решений output/tracker_decisions.json

Пишется при merge.debug: true. Формат — list[dict]:
json

{
  "seg": ["camera_new_3", 1, 177.6],
  "track_state": "lost",
  "score": null,
  "reason": "bev_infeasible"
}

Возможные reason:
reason	Что значит
"ok"	матч принят
"track_dead"	трек уже DIED
"same_cam_overlap"	сегменты пересекаются по времени на одной камере
"no_topology_edge"	пары (src_cam, dst_cam) нет в transitions
"edge_min_time_requires_lost"	min_time > 0 и трек не LOST
"bev_infeasible"	dist > bev_max_speed · travel
"below_min_sim"	итоговый скор < min_sim
"same_cam_reid_dim_mismatch"	размерности ReID не совпадают (Solider vs color)
"same_cam_reid_below_gate"	повторный вход в ту же камеру, но cos < same_cam_reid_min_sim
"never_confirmed"
