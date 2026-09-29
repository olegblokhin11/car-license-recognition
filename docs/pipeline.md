# Пайплайн: стадии и API

Что делает каждая стадия, какими функциями и скриптами она представлена.
Команды запуска и структура данных — в [README](../README.md), разбор
экспериментов — в [analysis.md](analysis.md).

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
[ocr]  fast_plate_ocr по ROI → текст номера + постобработка («китайский хак»)
        │
        ▼
[evaluation]  CER / CRR (conditional и end-to-end) + mAP на val
```

## Стадия 1: подготовка данных

Код: `src/plates/data_prep/`, CLI — `scripts/prepare_data.py`.

- `load_annotations()` — читает CSV `image_name,x_1,y_1,x_2,y_2,plate`,
  типизирует координаты.
- `split_by_images()` — делит по уникальным изображениям: все номера одной
  машины попадают в одну выборку, утечки между train и val нет.
- `build_yolo_dataset()` — копирует изображения, пишет YOLO-метки
  (нормализованные центральные xywh) и генерирует `data.yaml`.
- `list_split_images()` / `filter_annotations_by_split()` — читают уже
  собранный датасет: какие изображения попали в `images/val` (или `train`),
  и оставляют из аннотаций только их. На них опирается честная оценка
  на валидации.

Одно изображение может содержать 2–3 номера — это учитывается на всех стадиях.

## Стадия 2: авторазметка

Код: `src/plates/autolabel/grounding.py`, CLI — `scripts/autolabel.py`.

Zero-shot детектор `IDEA-Research/grounding-dino-tiny` с промптом
`"license plate"`. Найденные боксы фильтруются по уверенности и расширяют
train-часть датасета. Порог сильно влияет на чистоту: conf 0.35 даёт много
мусорных боксов, conf 0.6 — чище, но с меньшим покрытием.

## Стадия 3: детекция

Код: `src/plates/detection/`, CLI — `scripts/train_detector.py`
и `scripts/predict_detector.py`.

- `build_model()` — YOLO из yaml-спеки (`configs/detection/yolo26m_ocr.yaml`,
  один класс) плюс предобученные веса.
- `train()` — обёртка над `ultralytics.YOLO.train()`; параметры обучения
  задаются конфигами `configs/detection/default*.yaml`.
- `YoloDetector.predict()` — инференс на одном изображении.

## Стадия 4: OCR

Код: `src/plates/ocr/`, CLI — `scripts/run_ocr.py`.

- `FastPlateRecognizer` — обёртка над `fast_plate_ocr` (ONNX Runtime).
- `recognize_images()` — вырезает ROI по боксам, распознаёт, применяет
  постобработку.
- `postprocess_plate()` — чистка строки и «китайский хак»: граничные номера
  вида `粤X…港` восстанавливаются по префиксу юрисдикции.

## Стадия 5: метрики и оценка

Код: `src/plates/evaluation/`.

`metrics.py` — сами метрики:

- `character_error_rate()` — CER по расстоянию Левенштейна;
- `summarize_crr()` — сводка CRR / CER / exact-match по батчу;
- `evaluate_predictions()` — оценка списка `PlatePrediction`
  (используется в `ocr/pipeline.py`).

`pipeline_eval.py` — оценка системы целиком:

- `box_iou()` / `match_boxes()` — IoU и жадное сопоставление детекций
  с эталонами один-к-одному (порог IoU настраивается);
- `build_eval_images()` — собирает выборку из аннотаций (группирует номера
  по изображению, разрешает префиксы каталогов в `image_name`);
- `evaluate_ocr_only()` — CER/CRR по **эталонным** боксам (потолок OCR);
- `evaluate_pipeline()` — детекция + OCR, возвращает `PipelineMetrics`
  с двумя версиями CRR:
  - `end_to_end` — по всем эталонным номерам, пропуск детекции = ошибка;
  - `conditional` — только по сматченным боксам (качество OCR при корректной
    детекции);

  плюс счётчики `n_matched` / `n_missed` / `n_false_positives` и
  `detection_recall`.

CLI: `scripts/eval_val.py` — единственная точка входа для оценки. Считает
три фронта (mAP через `ultralytics.YOLO.val()`, OCR по эталонным боксам,
полный пайплайн), первые два отключаются флагами `--skip-detection` /
`--skip-ocr-only`, результат пишется в JSON по `--json`.

## Демо

Демо собрано из двух частей: `app.py` — интерфейс Streamlit, а логика лежит
в `src/plates/demo/` и от Streamlit не зависит.

- `discover_weights()` — находит обученные веса `assets/<эксперимент>/best.pt`;
- `run_pipeline()` — детекция и OCR по одному изображению, возвращает
  `DemoResult` с боксами, уверенностью детектора, сырым и обработанным текстом
  и таймингами стадий;
- `draw_detections()` — боксы с подписью «номер и уверенность», цвет по
  уверенности (`CONF_GOOD` / `CONF_MEDIUM`); сам текст номера на изображение
  не наносится, потому что `cv2.putText` не умеет китайские символы;
- `temporary_image()` — загруженная в браузере картинка пишется во временный
  файл, так как детектор Ultralytics принимает путь к файлу.

Логика покрыта `tests/test_demo.py` (заглушки вместо моделей) и
`tests/test_app.py` (smoke-тесты страницы через `streamlit.testing`).

## Точки расширения

- **Новый OCR-бэкенд**: реализовать `PlateRecognizer` (ABC) и добавить в
  `build_recognizer()`.
- **Экспорт YOLO в ONNX/TensorRT**: добавить `export()` в `detection/`.
- **Новая метрика**: чистые функции в `evaluation/metrics.py`
  (покрываются юнит-тестами без моделей и GPU).
