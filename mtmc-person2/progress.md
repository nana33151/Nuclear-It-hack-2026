# progress — person 2

## Done

- [x] Репозиторий + контракт Detection / Tracklet
- [x] I/O JSONL + mock в формате Person 1
- [x] Vendored ByteTrack + пайплайн
- [x] ReID `color_mock` + средний emb / цвет на треклете
- [x] Резка по дисперсии (`smoke_split.py`)
- [x] Смоук: 117 det → 2 tracklets, оба с reid_emb
- [x] SOLIDER inference: `app/reid_solider.py` + `weights/swin_s_market.pth` + vendored `third_party/SOLIDER-REID`

## Next

- [ ] Когда будет файл Person 1: `detections.source: jsonl`
- [ ] Визуальная проверка ID на одном реальном ролике (оверлей bbox+local_id)
- [ ] Опционально: Swin-B веса, если нужен сильнее ReID
