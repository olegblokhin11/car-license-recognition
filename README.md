# Распознавание автомобильных номеров (Гонконг / Китай / Макао)

Финальный проект курса: автоматическая система детекции и распознавания
автомобильных номеров на кропах транспортных средств. Два подзадачи:

1. **Детекция** всех номеров на кропе ТС (включая номера соседних машин).
2. **Распознавание** (посимвольный OCR) каждого задетектированного номера.

Номера — трёх юрисдикций: китайские (GB), гонконгские (HK) и макао-стиль.
В Гонконге ~10% номеров — произвольные надписи (например `ILOVECV`).

> **Авторство задания:** Финальный проект выполнен в рамках курса
> [Karpov.Courses «Deep Learning»](https://karpov.courses/deep-learning).
> Условия практического задания принадлежат ООО «Карпов Курсы».
> Исходные данные курса (изображения и разметка базы данных курса)
> в репозиторий не включены.

## Структура

```
├── src/plates/           # пакет пайплайна
│   ├── config.py         # централизованные пути/параметры
│   ├── data_prep/        # загрузка аннотаций, сплит, построение YOLO-датасета
│   ├── autolabel/        # авторазметка GroundingDINO (расширение датасета)
│   ├── detection/        # YOLO-детекция: обучение и инференс
│   ├── ocr/              # распознавание через fast_plate_ocr + постобработка
│   └── evaluation/       # метрики CER/CRR
├── scripts/              # CLI-обёртки над стадиями пайплайна
├── configs/detection/    # конфиги обучения Ultralytics
├── tests/                # юнит-тесты чистых частей
├── data/                 # сырые данные (см. ниже)
├── assets/               # веса моделей (.pt)
└── docs/                 # документация и анализ
```

## Установка

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
# CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e '.[ocr,detection,autolabel,dev]'
```

Проверка:

```bash
python -c "import torch; print(torch.cuda.is_available())"   # True
pytest -q
```

## Данные

Сырые данные (кропы ТС, разметка номеров) ожидаются в `data/raw/`:

```
data/raw/
    images/
        train/        # изображения с разметкой (train_annot.txt)
        test/         # неразмеченные тестовые изображения
        not_annotated/# еще не размеченные (для авторазметки)
    annotation/
        train_annot.txt   # image_name,x_1,y_1,x_2,y_2,plate
```

В этом репозитории датасеты уже собраны:
`dataset_for_yolo`, `dataset_for_yolo_extended`, `dataset_for_yolo_extended_v2`.

## Быстрый старт

### 1. Собрать YOLO-датасет из аннотаций

```bash
python scripts/prepare_data.py \
    --data-dir data/raw/images \
    --annotation data/raw/annotation/train_annot.txt \
    --out dataset_for_yolo \
    --val-fraction 0.2
```

### 2. Доразметить неразмеченные изображения (GroundingDINO)

```bash
python scripts/autolabel.py \
    --images data/raw/images/not_annotated \
    --existing dataset_for_yolo \
    --out dataset_for_yolo_extended \
    --conf 0.35
```

### 3. Обучить детектор

```bash
python scripts/train_detector.py \
    --data dataset_for_yolo_extended_v2/data.yaml \
    --cfg configs/detection/default_ext_v2.yaml \
    --model yolo26m_ocr.yaml \
    --weights yolo26m.pt \
    --device 0
```

### 4. Инференс детекции → CSV боксов

```bash
python scripts/predict_detector.py \
    --weights runs/detect/yolo26_ocr/base_ext_v2/weights/best.pt \
    --images data/raw/images/test \
    --out out/predictions.txt --conf 0.25 --imgsz 960
```

### 5. OCR по боксам

```bash
python scripts/run_ocr.py \
    --predictions out/predictions.txt \
    --images data/raw/images \
    --out out/recognition_results.txt
```

### 6. Полный инференс одной командой

```bash
python scripts/infer.py \
    --weights assets/yolo26m.pt \
    --images data/raw/images/test \
    --out out/inference_results.txt
```

## Текущие результаты

| Этап | Метрика | Значение |
|---|---|---|
| Детекция (mAP50-95, YOLO26m 100 эп.) | mAP50-95 | ~0.81–0.83 |
| OCR (fast_plate_ocr, cct-s) | CER / CRR | ~0.047 / ~0.953 |
| OCR после детекции (полный пайплайн) | CRR | ~0.81–0.85 |

> Подробный разбор слабых мест и идей улучшения — в [docs/analysis.md](docs/analysis.md).

## Воспроизводимость

- все стадии используют фиксированный seed (24);
- CLI принимают явные пути, значения по умолчанию переопределяются через
  переменные окружения `PLATES_*` (см. `src/plates/config.py`);
- результаты обучения сохраняются в `runs/`, артефакты — в `out/`.

## Лицензия

MIT. Внимание: обученные веса Ultralytics распространяются под AGPL-3.0 —
проверьте совместимость при коммерческом использовании.