# Пайплайн: стадии, данные, API

## Общая схема

```
raw annotation (train_annot.txt, Pascal-VOC боксы)
        │
        ▼
[data_prep]  load → split (по изображениям) → build YOLO dataset (images/ + labels/ + data.yaml)
        │
        ▼
[autolabel]  GroundingDINO на not_annotated → расширение датасета (conf 0.35–0.6)
        │
        ▼
[detection]  YOLO26 (yolo26m_ocr.yaml) train → best.pt
        │
        ▼
[detection→predict]  инференс на test/ → predictions.txt (image_name,x1,y1,x2,y2,conf)
        │
        ▼
[ocr]  fast_plate_ocr по ROI → plate текст + постобработка (chinese hack)
        │
        ▼
[evaluation]  CER / CRR / точных совпадений
```

## Данные

| Данные | Путь | Описание |
|---|---|---|
| Аннотация (трейн) | `data/raw/annotation/train_annot.txt` | 4511 номеров, 4074 изображения |
| Train изображения | `data/raw/images/train/` | 4074 шт. |
| Test изображения | `data/raw/images/test/` | 1343 шт. (без разметки) |
| Не размечено | `data/raw/images/not_annotated/` | 3466 шт. |
| YOLO-датасет №1 | `dataset_for_yolo/` | только ручная разметка |
| YOLO-датасет №2 | `dataset_for_yolo_extended/` | + GroundingDINO conf 0.35 |
| YOLO-датасет №3 | `dataset_for_yolo_extended_v2/` | + GroundingDINO conf 0.6 |

Формат аннотации — CSV:
`image_name,x_1,y_1,x_2,y_2,plate`
Пара «номеров» на одно изображение — обычное явление (до 3 на ТС).

## Стадия 1: подготовка данных

Код: `src/plates/data_prep/`.

- `load_annotations()` — читает CSV, типизирует координаты.
- `split_by_images()` — делит по уникальным изображениям (без утечки):
  все номера одной машины в одну выборку.
- `build_yolo_dataset()` — копирует изображения, пишет YOLO-метки
  (нормализованные центральные xywh) и генерирует `data.yaml`.

## Стадия 2: авторазметка

Код: `src/plates/autolabel/grounding.py`.

Использует zero-shot детектор `IDEA-Research/grounding-dino-tiny`
с промптом `"license plate"`. Боксы фильтруются по уверенности
(`conf`), расширяют train-часть. ВАЖНО: порог сильно влияет на чистоту
датасета — 0.35 даёт много мусорных боксов, 0.6 — чище, но меньше
покрытия. В `dataset_for_yolo_extended_v2` использован 0.6.

## Стадия 3: детекция

Код: `src/plates/detection/`.

- `build_model()` → YOLO из yaml + предобученные веса.
- `train()` — обёртка над `ultralytics.YOLO.train()` (отдельные
  конфиги: `configs/detection/default*.yaml`).
- `YoloDetector.predict()` — инференс на одном изображении.

Ключевые конфиги обучения:

| Конфиг | имgsz | эпохи | Примечание |
|---|---|---|---|
| `default.yaml` | 640 | 100 | база (m) |
| `default_little.yaml` | 640 | 50 | nano |
| `default_little_ext.yaml` | 640 | 50 | nano+ext |
| `default_little_ext_v2.yaml` | 640 | 50 | nano+ext_v2 |
| `default_little_ext_v2_960.yaml` | 960 | 50 | nano+960 |
| `default_ext_v2.yaml` | 640 | 100 | m+ext_v2 (лучший) |

## Стадия 4: OCR

Код: `src/plates/ocr/`.

- `FastPlateRecognizer` — обёртка над `fast_plate_ocr`.
- `recognize_images()` — вырезает ROI по боксам, распознаёт,
  применяет постобработку.
- `postprocess_plate()` — чистка + «китайский хак» (граничные номера
  `粤X...港`).

## Стадия 5: метрики

Код: `src/plates/evaluation/metrics.py`.

- `character_error_rate()` — CER по Левенштейну.
- `summarize_crr()` — сводка CRR/CER/exact-match по батчу.
- `evaluate_predictions()` — для списка `PlatePrediction`.

## Точки расширения

- Новый OCR-бэкенд: реализовать `PlateRecognizer` (ABC) и добавить в
  `build_recognizer()`.
- Экспорт YOLO в ONNX/TensorRT: в `detection/` можно добавить `export()`.
- Оценка mAP: `ultralytics.YOLO.val()` уже даёт mAP на val-сплите.