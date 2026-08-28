"""Полный OCR-пайплайн: обрезка ROI по боксам, распознавание, метрики.

Два сценария:
- ``recognize_boxes`` — по готовым боксам (из детектора или ground-truth)
  вырезать ROI, распознать номера, (опционально) посчитать CER/CRR;
- ``recognize_detections`` — то же, но читает CSV детекций детектора.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..evaluation.metrics import summarize_crr
from .postprocess import postprocess_plate
from .recognizer import PlateRecognizer, build_recognizer


@dataclass
class PlatePrediction:
    """Одна предсказанная пластина."""

    image_name: str
    x_1: int
    y_1: int
    x_2: int
    y_2: int
    raw: str
    plate: str  # после постобработки
    ground_truth: Optional[str] = None
    cer: Optional[float] = None


def extract_roi(
    image: np.ndarray,
    box,
    expand_pixels: int = 0,
    enhance: bool = False,
) -> np.ndarray:
    """Вырезает ROI номера из изображения (BGR) с опцией расширения/контраста.

    Возвращает RGB-crop, готовый к подаче в fast_plate_ocr.
    """
    h, w = image.shape[:2]
    x1 = max(0, int(box[0]) - expand_pixels)
    y1 = max(0, int(box[1]) - expand_pixels)
    x2 = min(w, int(box[2]) + expand_pixels)
    y2 = min(h, int(box[3]) + expand_pixels)

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    if enhance:
        gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        roi_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return roi_rgb


def recognize_images(
    images_dir: str | Path,
    image_names: list[str],
    boxes: list[tuple],
    recognizer: PlateRecognizer,
    expand_pixels: int = 0,
    enhance: bool = False,
    use_chinese_hack: bool = True,
) -> list[PlatePrediction]:
    """Распознаёт номера на заданных изображениях/боксах.

    Args:
        images_dir: корень, где лежат изображения.
        image_names: имена (с префиксом каталога или без).
        boxes: список боксов (x1,y1,x2,y2) — столько же, сколько имён.
        recognizer: распознаватель (см. ``ocr/recognizer``).
        expand_pixels: расширение ROI.
        enhance: повышать контраст ROI.
        use_chinese_hack: применять постобработку границ номеров.

    Returns:
        Список ``PlatePrediction``.
    """
    dir_ = Path(images_dir)
    preds: list[PlatePrediction] = []

    for image_name, box in zip(image_names, boxes):
        path = dir_ / image_name
        if not path.exists():
            path = dir_ / Path(image_name).name
        img = cv2.imread(str(path)) if path.exists() else None

        if img is None:
            preds.append(
                PlatePrediction(
                    image_name=image_name,
                    x_1=int(box[0]),
                    y_1=int(box[1]),
                    x_2=int(box[2]),
                    y_2=int(box[3]),
                    raw="",
                    plate="",
                )
            )
            continue

        roi = extract_roi(img, box, expand_pixels=expand_pixels, enhance=enhance)
        raw = recognizer.recognize(roi)
        pred_text = postprocess_plate(raw, enable_hack=use_chinese_hack)
        preds.append(
            PlatePrediction(
                image_name=image_name,
                x_1=int(box[0]),
                y_1=int(box[1]),
                x_2=int(box[2]),
                y_2=int(box[3]),
                raw=raw,
                plate=pred_text,
            )
        )
    return preds


def evaluate_predictions(preds: list[PlatePrediction]) -> dict:
    """Считает CER/CRR, если у предсказаний заполнен ground-truth.

    Returns:
        словарь со сводкой метрик.
    """
    result = summarize_crr(
        [p.ground_truth if p.ground_truth else "" for p in preds],
        [p.plate for p in preds],
    )
    return {
        "total": result.total,
        "exact_matches": result.exact_matches,
        "cer": round(result.cer, 4),
        "crr": round(result.crr, 4),
        "errors": result.errors[:20],
    }
