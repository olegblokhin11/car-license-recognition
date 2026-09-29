"""Оценка полного пайплайна (детекция → OCR) и OCR по эталонным боксам.

Модуль отвечает на два разных вопроса, которые легко перепутать:

- **conditional CRR** — качество OCR по тем номерам, которые детектор нашёл
  (боксы, сматченные с эталоном при IoU ≥ порога). Так считал прежний
  ``scripts/eval_pipeline.py`` (удалён — обе метрики отдаёт
  ``scripts/eval_val.py``);
- **end-to-end CRR** — качество системы целиком: каждый эталонный номер
  учитывается, а не найденный детекцией считается ошибкой (пустое
  предсказание).

Детектор и распознаватель передаются как объекты с методами
``predict(path, conf=..., imgsz=...)`` и ``recognize(roi)``, поэтому логику
можно тестировать подстановкой заглушек (см. ``tests/test_pipeline_eval.py``).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..ocr.pipeline import recognize_images
from ..ocr.recognizer import PlateRecognizer
from .metrics import CRRResult, summarize_crr

Box = tuple[float, float, float, float]

logger = logging.getLogger(__name__)


def resolve_image_path(image_name: str, images_root: str | Path) -> Path | None:
    """Находит файл изображения по имени из аннотации.

    Пробует ``images_root / image_name``, затем ``images_root / basename``
    (в аннотации имя может храниться с префиксом каталога, например
    ``train/000000_0.jpg``).

    Returns:
        Путь к существующему файлу или ``None``.
    """
    root = Path(images_root)
    direct = root / image_name
    if direct.exists():
        return direct
    alternative = root / Path(image_name).name
    return alternative if alternative.exists() else None


def build_eval_images(
    df: pd.DataFrame,
    images_root: str | Path,
    image_col: str = "image_name",
) -> list[EvalImage]:
    """Собирает список ``EvalImage`` из аннотаций.

    Args:
        df: аннотации (колонки ``image_name``, ``x_1``, ``y_1``, ``x_2``,
            ``y_2``, ``plate``).
        images_root: корень, где лежат изображения.
        image_col: колонка с именем изображения.

    Returns:
        Список изображений с эталонными номерами в порядке первого появления
        в ``df``. Изображения, файл которых не найден, пропускаются
        с предупреждением в лог.
    """
    grouped: dict[str, EvalImage] = {}
    missing: list[str] = []

    for _, row in df.iterrows():
        name = str(row[image_col])
        base = Path(name).name
        if base not in grouped:
            path = resolve_image_path(name, images_root)
            if path is None:
                missing.append(name)
                continue
            grouped[base] = EvalImage(path=path, plates=[])
        grouped[base].plates.append(
            PlateRef(
                box=(
                    float(row["x_1"]),
                    float(row["y_1"]),
                    float(row["x_2"]),
                    float(row["y_2"]),
                ),
                text=str(row["plate"]),
            )
        )

    if missing:
        logger.warning(
            "Не найдено файлов для %d изображений из аннотации, например: %s",
            len(missing),
            ", ".join(missing[:5]),
        )
    return list(grouped.values())


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU двух боксов в формате ``(x1, y1, x2, y2)``.

    Возвращает 0.0 для вырожденных (нулевой площади) и не пересекающихся
    боксов.
    """
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class PlateRef:
    """Эталонный номер: бокс и текст."""

    box: Box
    text: str


@dataclass
class EvalImage:
    """Изображение для оценки: путь к файлу и эталонные номера на нём."""

    path: Path
    plates: list[PlateRef]


def match_boxes(
    det_boxes: Sequence[Sequence[float]],
    gt_boxes: Sequence[Sequence[float]],
    iou_thr: float = 0.5,
) -> list[tuple[int, int]]:
    """Жадно сопоставляет детекции с эталонами один-к-одному.

    Идём по детекциям и для каждой выбираем свободный эталон с максимальным
    IoU; пара принимается, если ``IoU >= iou_thr``.

    Args:
        det_boxes: боксы детектора.
        gt_boxes: эталонные боксы.
        iou_thr: минимальный IoU для сопоставления.

    Returns:
        Список пар ``(индекс детекции, индекс эталона)``.
    """
    pairs: list[tuple[int, int]] = []
    used_gt: set[int] = set()
    for di, det in enumerate(det_boxes):
        best_iou, best_gi = 0.0, -1
        for gi, gt in enumerate(gt_boxes):
            if gi in used_gt:
                continue
            value = box_iou(det, gt)
            if value > best_iou:
                best_iou, best_gi = value, gi
        if best_gi >= 0 and best_iou >= iou_thr:
            used_gt.add(best_gi)
            pairs.append((di, best_gi))
    return pairs


@dataclass
class PipelineMetrics:
    """Сводка оценки полного пайплайна на выборке."""

    n_images: int = 0
    n_gt: int = 0
    n_detections: int = 0
    n_matched: int = 0
    conditional: CRRResult = field(default_factory=CRRResult)
    end_to_end: CRRResult = field(default_factory=CRRResult)

    @property
    def n_missed(self) -> int:
        """Сколько эталонных номеров детекция не нашла."""
        return self.n_gt - self.n_matched

    @property
    def n_false_positives(self) -> int:
        """Сколько детекций не сопоставилось ни с одним эталоном."""
        return self.n_detections - self.n_matched

    @property
    def detection_recall(self) -> float:
        """Доля найденных эталонных номеров (при заданном IoU)."""
        return self.n_matched / self.n_gt if self.n_gt else 0.0


def evaluate_pipeline(
    images: Iterable[EvalImage],
    detector,
    recognizer: PlateRecognizer,
    conf: float = 0.25,
    imgsz: int = 960,
    iou_thr: float = 0.5,
) -> PipelineMetrics:
    """Прогоняет детекцию + OCR по выборке и считает обе версии CRR.

    Args:
        images: изображения с эталонными номерами.
        detector: детектор с методом ``predict(path, conf=..., imgsz=...)``.
        recognizer: распознаватель (см. ``ocr.recognizer``).
        conf: порог уверенности детектора.
        imgsz: разрешение инференса детектора.
        iou_thr: минимальный IoU для сопоставления с эталоном.

    Returns:
        ``PipelineMetrics`` с метриками ``conditional`` и ``end_to_end``.
    """
    images = list(images)
    metrics = PipelineMetrics(n_images=len(images))
    cond_gt: list[str] = []
    cond_pred: list[str] = []
    e2e_gt: list[str] = []
    e2e_pred: list[str] = []

    for item in images:
        path = Path(item.path)
        detections = detector.predict(path, conf=conf, imgsz=imgsz)
        det_boxes = [(d.x1, d.y1, d.x2, d.y2) for d in detections]
        gt_boxes = [plate.box for plate in item.plates]
        pairs = match_boxes(det_boxes, gt_boxes, iou_thr=iou_thr)

        metrics.n_detections += len(det_boxes)
        metrics.n_gt += len(gt_boxes)
        metrics.n_matched += len(pairs)

        predictions = (
            recognize_images(
                str(path.parent),
                [path.name] * len(pairs),
                [det_boxes[di] for di, _ in pairs],
                recognizer,
            )
            if pairs
            else []
        )
        pred_by_gt = {gi: pred.plate for (_, gi), pred in zip(pairs, predictions, strict=True)}

        for gi, plate in enumerate(item.plates):
            pred_text = pred_by_gt.get(gi, "")
            e2e_gt.append(plate.text)
            e2e_pred.append(pred_text)
            if gi in pred_by_gt:
                cond_gt.append(plate.text)
                cond_pred.append(pred_text)

    metrics.conditional = summarize_crr(cond_gt, cond_pred)
    metrics.end_to_end = summarize_crr(e2e_gt, e2e_pred)
    return metrics


def evaluate_ocr_only(
    images: Iterable[EvalImage],
    recognizer: PlateRecognizer,
    expand_pixels: int = 0,
    enhance: bool = False,
) -> CRRResult:
    """Оценивает OCR по эталонным боксам (потолок качества распознавания).

    Args:
        images: изображения с эталонными номерами.
        recognizer: распознаватель.
        expand_pixels: расширение ROI.
        enhance: повышать контраст ROI.

    Returns:
        ``CRRResult`` по всем эталонным номерам выборки.
    """
    gt: list[str] = []
    pred: list[str] = []
    for item in images:
        if not item.plates:
            continue
        path = Path(item.path)
        predictions = recognize_images(
            str(path.parent),
            [path.name] * len(item.plates),
            [plate.box for plate in item.plates],
            recognizer,
            expand_pixels=expand_pixels,
            enhance=enhance,
        )
        gt.extend(plate.text for plate in item.plates)
        pred.extend(p.plate for p in predictions)
    return summarize_crr(gt, pred)


def crr_to_dict(result: CRRResult) -> dict:
    """Представляет ``CRRResult`` как словарь (для JSON-выгрузки)."""
    return {
        "n": result.total,
        "exact_matches": result.exact_matches,
        "exact_match_rate": round(result.exact_match_rate, 4),
        "cer": round(result.cer, 4),
        "crr": round(result.crr, 4),
    }


def pipeline_metrics_to_dict(metrics: PipelineMetrics) -> dict:
    """Представляет ``PipelineMetrics`` как словарь (для JSON-выгрузки)."""
    return {
        "n_images": metrics.n_images,
        "n_gt": metrics.n_gt,
        "n_detections": metrics.n_detections,
        "n_matched": metrics.n_matched,
        "n_missed": metrics.n_missed,
        "n_false_positives": metrics.n_false_positives,
        "detection_recall": round(metrics.detection_recall, 4),
        "conditional": crr_to_dict(metrics.conditional),
        "end_to_end": crr_to_dict(metrics.end_to_end),
    }
