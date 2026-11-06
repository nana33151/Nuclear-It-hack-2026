# Готовые модели и воспроизводимость

Веса загружает `.venv/bin/python -m perception.download_models`. Уже скачанные
файлы проверяются по SHA256; существующий файл с неверной суммой не затирается.
Для загрузки нужна сеть. Для дальнейшей обработки — только локальные файлы.
Сведения об источнике сохраняются рядом с весами в `*.source.json` и копируются
в manifest каждого запуска при совпадении SHA256.

## YOLO11s-seg

- Источник: [Ultralytics assets v8.3.0](https://github.com/ultralytics/assets/releases/tag/v8.3.0).
- Файл: `models/yolo11s-seg.pt`.
- SHA256: `1caa81c0195412efa411b632bcfb8c184939dddb6ae41f6a80c41b211ff257c3`.
- Используется класс COCO `person`, без обучения. Модель сразу выдаёт bbox и маску.
- `retina_masks=True` нужен для получения маски в разрешении исходного кадра.
- [Документация YOLO11](https://docs.ultralytics.com/models/yolo11/).
- [Документация сегментации](https://docs.ultralytics.com/tasks/segment/).
- [Лицензия кода Ultralytics: AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

Размер s выбран для работоспособного прототипа на ноутбуке. Для другого варианта
YOLO11-seg используйте `--model путь/к/весам.pt`; после замены заново проверьте
дальних людей, перекрытия и время обработки. Автоматического обучения нет.

## SCHP ATR18

- Оригинальная архитектура: [Self-Correction Human Parsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing).
- Обёртка и ONNX-экспорт: [pirocheto/schp-human-parsing](https://github.com/pirocheto/schp-human-parsing).
- Репозиторий весов: [pirocheto/schp-atr-18](https://huggingface.co/pirocheto/schp-atr-18).
- Зафиксированная ревизия: `a54fa65e7e4f27f21011f652ffc0053ebb24b292`.
- Исходный файл: `onnx/schp-atr-18-int8-static.onnx`, **INT8 static**, один файл.
- Локальное имя: `models/schp/schp-atr-18.onnx` (это INT8, несмотря на короткое имя).
- SHA256: `4420d8db8c1f266967c89485786b01209f6d405f320fc0f87e8ced49392cefb5`.
- Исполнитель: ONNX Runtime, CPU, 4 потока. Python-код с Hugging Face не исполняется.

Preprocessing соответствует [опубликованному image processor](https://huggingface.co/pirocheto/schp-atr-18/blob/main/image_processing_schp.py):
цвет RGB, прямой PIL bilinear resize 512×512, деление на 255, mean
`[0.406,0.456,0.485]`, std `[0.225,0.224,0.229]`, NCHW float32.
Константы намеренно не заменены на обычный порядок ImageNet RGB: они повторяют
опубликованный wrapper этой модели. Логиты интерполируются до размера кропа,
после чего выбирается класс каждого пикселя.

ATR: верхняя одежда — класс 4, юбка — 5, брюки — 6, платье — 7. Платье делится
на верх/низ по 55% высоты кропа. Волосы, кожа, сумки, обувь и фон исключаются.
Маска одежды пересекается с силуэтом конкретной YOLO-детекции.

У оригинального кода SCHP опубликована MIT-лицензия; атрибуцию архитектуры,
обёртки и весов сохраняйте отдельно. Условия выбранных готовых компонентов
не определяются тем, что обучение в этом проекте не выполняется.

## Что реализовано в проекте

Декодирование и временные метки, экспорт исходных координат, кропы/маски,
эвристики качества, HSV-гистограммы по областям одежды, явный spatial fallback,
версионированный контракт, проверка целостности и лёгкий загрузчик person2.
MOT, SOLIDER и межкамерная ассоциация в этот код не входят.
