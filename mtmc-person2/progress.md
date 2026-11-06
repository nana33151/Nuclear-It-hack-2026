# progress — person 2

## Done

- [x] Репозиторий + контракт Detection / Tracklet
- [x] I/O JSONL + mock в формате Person 1
- [x] Vendored ByteTrack + пайплайн
- [x] ReID `color_mock` + средний emb / цвет на треклете
- [x] Резка по дисперсии (`smoke_split.py`)
- [x] Смоук: 117 det → 2 tracklets, оба с reid_emb

## Next

- [ ] Env 3.10–3.11 + torch + веса SOLIDER, дописать `src/reid/solider.py`
- [ ] Когда будет файл Person 1: `detections.source: jsonl`
- [ ] Визуальная проверка ID на одном реальном ролике (оверлей bbox+local_id)
