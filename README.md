# Распознавание автомобильных номеров (Гонконг / Китай / Макао)

[![CI](https://github.com/olegblokhin11/car-license-recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/olegblokhin11/car-license-recognition/actions/workflows/ci.yml)

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
│   └── evaluation/       # метрики CER/CRR + оценка пайплайна
├── scripts/              # CLI-обёртки над стадиями пайплайна
├── configs/detection/    # конфиги обучения Ultralytics
├── tests/                # юнит-тесты чистых частей
├── data/                 # сырые данные (см. ниже)
├── assets/               # веса моделей (.pt)
├── docs/
│   ├── pipeline.md       # стадии пайплайна и API
│   └── analysis.md       # эксперименты, выводы, точки роста
├── requirements.txt      # зависимости (для разработки — requirements-dev.txt)
├── ruff.toml             # конфиг линтера
├── pytest.ini            # конфиг тестов
└── LICENSE               # MIT
```

## Установка

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
# CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt        # рантайм-зависимости
pip install -r requirements-dev.txt    # + pytest и ruff (для разработки)
```

Проверка:

```bash
python -c "import torch; print(torch.cuda.is_available())"   # True
pytest -q
```

Пакет `src/plates` не устанавливается как дистрибутив: скрипты запускаются
из корня репозитория и сами добавляют `src/` в `sys.path`
(в тестах то же делает `pytest.ini`).

Линтер и формат — [ruff](https://docs.astral.sh/ruff/) (конфигурация
в `ruff.toml`):

```bash
ruff check src scripts tests     # линт
ruff format src scripts tests    # форматирование
```

CI (GitHub Actions, `.github/workflows/ci.yml`) на каждый пуш и PR прогоняет
линт, юнит-тесты и smoke-проверку CLI (`--help` каждого скрипта). Обучение,
инференс на GPU и оценка качества в CI не запускаются: для них нужны веса
и датасет, которых в репозитории нет.

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

### 7. Smoke-тест пайплайна

Быстрая проверка, что детекция и OCR работают (20 изображений, требует GPU
и весов):

```bash
python scripts/smoke_test.py --images data/raw/images/test --n 20
```

### 8. Оценка на валидации

Три фронта сразу — детекция (mAP), OCR по эталонным боксам и полный пайплайн:

```bash
python scripts/eval_val.py \
    --weights assets/base_ext_v2/best.pt \
    --dataset dataset_for_yolo_extended_v2 --split val \
    --json out/eval_val.json
```

Только пайплайн (детекция + OCR), если mAP и OCR по эталонным боксам не нужны:

```bash
python scripts/eval_val.py \
    --weights assets/base_ext_v2/best.pt \
    --dataset dataset_for_yolo_extended_v2 --split val \
    --skip-detection --skip-ocr-only
```

Скрипт берёт выборку из уже собранного датасета (`images/<split>`), считает
`end-to-end` CRR (пропуск детекции = ошибка) и `conditional` CRR (только
сматченные боксы), а также recall детекции, число пропусков и лишних
детекций.

## Текущие результаты

Замер на **чистой валидации**: 814 изображений / 895 номеров из
`dataset_for_yolo_extended_v2/images/val`.

**Что именно оценивалось:**

- **Детекция:** `assets/base_ext_v2/best.pt` —
  YOLO26m по спеке `configs/detection/yolo26m_ocr.yaml`, обучение с
  `configs/detection/default_ext_v2.yaml` (100 эпох, imgsz 640, batch 16,
  patience 100, seed 0).
- **OCR:** `fast_plate_ocr`, модель `cct-s-v2-global-model`
  (см. `src/plates/ocr/recognizer.py`), постобработка `postprocess_plate`
  с «китайским хаком».

| Фронт | Метрика | Значение |
|---|---|---|
| Детекция, imgsz 640 | mAP50 / mAP50-95 / P / R | 0.989 / **0.828** / 0.962 / 0.971 |
| Детекция, imgsz 960 | mAP50 / mAP50-95 / P / R | 0.987 / 0.817 / 0.967 / 0.975 |
| OCR только (боксы GT — «потолок») | CRR / CER / точных | **0.9484** / 0.0516 / 84.2 % (754/895) |
| Пайплайн, **end-to-end** | CRR / CER / точных | **0.9352** / 0.0648 / 83.7 % (749/895) |
| Пайплайн, conditional (только сматченные) | CRR / CER / точных | 0.9544 / 0.0456 / 85.4 % (749/877) |
| Детекция: recall@IoU 0.5, conf 0.25 | сматчено / пропущено / лишних | 877 / **18** / 62 |

Как читать:

- **Итоговая цифра пайплайна — CRR 0.9352 (end-to-end)**: пропущенный
  детекцией номер считается ошибкой. Именно она отвечает на вопрос
  «качество системы целиком».
- **CRR 0.9544 (conditional)** — качество OCR *при условии*, что детекция
  нашла номер (conditional-режим `scripts/eval_val.py`). Разница между ними
  (0.019) — вклад 18 пропущенных номеров.
- OCR по GT-боксам (0.9484) оказался чуть ниже conditional-пайплайна
  (0.9544): боксы детектора для OCR немного удобнее ручной разметки
  (n различается: 877 против 895).
- CRR считается по эталонным номерам, поэтому 62 лишних детекции в него
  не входят: если важно «сколько выданных номеров неверны», нужна отдельная
  precision-метрика.

> Подробный разбор слабых мест, методики и идей улучшения — в [docs/analysis.md](docs/analysis.md).

## Воспроизводимость

- все стадии используют фиксированный seed (24);
- CLI принимают явные пути, значения по умолчанию переопределяются через
  переменные окружения `PLATES_*` (см. `src/plates/config.py`);
- результаты обучения сохраняются в `runs/`, артефакты — в `out/`.

## Об использовании AI-ассистента

**Сделано автором:** постановка задачи и исходный учебный ноутбук курса
(в репозиторий не включён — там данные и условия задания), архитектура
пайплайна (детекция → OCR → постобработка), выбор моделей и датасетов, все
эксперименты обучения (YOLO26m/nano, 640/960 px, разные наборы
авторазметки GroundingDINO), решения по «китайскому хаку» и порогам,
направление доработок и приёмка результата.

**Сделано с помощью AI-ассистента:** рефакторинг ноутбука в пакет
`src/plates/` со стадиями, юнит-тесты, документация (`docs/`, этот README),
скрипты оценки (`scripts/eval_val.py`) и разбор слабых мест в `docs/analysis.md`.


Автор прочитал и разобрал весь код в репозитории, может объяснить любое
решение и воспроизвести любой замер.

## Лицензия

Код распространяется под [MIT License](LICENSE).

Внимание: обученные веса Ultralytics распространяются под AGPL-3.0 —
проверьте совместимость при коммерческом использовании. В репозиторий веса
не включены, модель обучается на этом пайплайне самостоятельно.