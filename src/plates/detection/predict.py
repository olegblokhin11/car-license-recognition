"""Инференс обученной модели детекции и формирование предсказаний."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DetectionBox:
    """Один задетектированный бокс номера."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


class YoloDetector:
    """Обёртка над Ultralytics YOLO для детекции номеров."""

    def __init__(self, model_path: str | Path):
        from ultralytics import YOLO  # ленивый импорт

        self._model = YOLO(str(model_path))

    def predict(
        self,
        image_path: str | Path,
        conf: float = 0.25,
        imgsz: int = 960,
    ) -> list[DetectionBox]:
        """Детектирует номера на одном изображении."""
        results = self._model(str(image_path), conf=conf, imgsz=imgsz, verbose=False)[0]
        boxes: list[DetectionBox] = []
        if results.boxes is None:
            return boxes

        xyxy = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        for box, c in zip(xyxy, confs):
            x1, y1, x2, y2 = (int(v) for v in box)
            boxes.append(DetectionBox(x1, y1, x2, y2, float(c)))
        return boxes


def predict_directory(
    detector: YoloDetector,
    images_dir: str | Path,
    image_prefix: str = "trailer/",
    conf: float = 0.25,
    imgsz: int = 960,
) -> pd.DataFrame:
    """Прогнозирует на всех изображениях и собирает DataFrame.

    Args:
        detector: инициализированный детектор.
        images_dir: папка с изображениями (без подписи train/test).
        image_prefix: какие имя в ``image_name`` — совместимость с CSV курса.

    Returns:
        DataFrame c колонками ``image_name, x_1, y_1, x_2, y_2, conf``.
    """
    rows: list[dict] = []
    image_dir = Path(images_dir)
    images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    for img in images:
        dets = detector.predict(img, conf=conf, imgsz=imgsz)
        for d in dets:
            rows.append(
                {
                    "image_name": f"{image_prefix}{img.name}",
                    "x_1": d.x1,
                    "y_1": d.y1,
                    "x_2": d.x2,
                    "y_2": d.y2,
                    "conf": d.confidence,
                }
            )
    return pd.DataFrame(rows, columns=["image_name", "x_1", "y_1", "x_2", "y_2", "conf"])