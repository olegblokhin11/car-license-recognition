"""Логика демо-приложения: поиск весов, прогон пайплайна, отрисовка результата.

Модуль не зависит от Streamlit — его функции вызываются и из ``app.py``,
и из юнит-тестов. Детектор и распознаватель передаются как объекты
с методами ``predict(path, conf=..., imgsz=...)`` и ``recognize(roi)``,
поэтому в тестах вместо моделей подставляются заглушки.
"""

from __future__ import annotations

import contextlib
import tempfile
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..ocr.pipeline import extract_roi
from ..ocr.postprocess import postprocess_plate

# Пороги раскраски боксов по уверенности детектора (только для визуализации).
CONF_GOOD = 0.70
CONF_MEDIUM = 0.40

_COLOR_GOOD = (60, 190, 60)  # BGR
_COLOR_MEDIUM = (0, 190, 255)
_COLOR_LOW = (60, 60, 230)


@dataclass(frozen=True)
class PlateResult:
    """Найденный номер: бокс, уверенность детектора и результат OCR."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    raw: str
    plate: str

    @property
    def box(self) -> tuple[int, int, int, int]:
        """Бокс в формате ``(x1, y1, x2, y2)``."""
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class DemoResult:
    """Результат прогона по одному изображению."""

    plates: list[PlateResult] = field(default_factory=list)
    width: int = 0
    height: int = 0
    detection_ms: float = 0.0
    ocr_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        """Суммарное время инференса, мс."""
        return self.detection_ms + self.ocr_ms

    @property
    def mean_confidence(self) -> float:
        """Средняя уверенность детектора по найденным номерам."""
        if not self.plates:
            return 0.0
        return sum(p.confidence for p in self.plates) / len(self.plates)


def discover_weights(assets_dir: str | Path = "assets") -> list[Path]:
    """Находит обученные веса вида ``assets/<эксперимент>/best.pt``.

    Args:
        assets_dir: каталог с весами.

    Returns:
        Отсортированный список путей (пустой, если весов нет).
    """
    root = Path(assets_dir)
    if not root.is_dir():
        return []
    return sorted(root.glob("*/best.pt"))


def run_pipeline(
    image_path: str | Path,
    detector,
    recognizer,
    conf: float = 0.25,
    imgsz: int = 960,
    expand_pixels: int = 0,
    enhance: bool = False,
    use_chinese_hack: bool = True,
) -> DemoResult:
    """Прогоняет детекцию и OCR по одному изображению.

    Args:
        image_path: путь к изображению.
        detector: детектор с методом ``predict(path, conf=..., imgsz=...)``.
        recognizer: распознаватель с методом ``recognize(roi)``.
        conf: порог уверенности детектора.
        imgsz: разрешение инференса детектора.
        expand_pixels: расширение ROI перед распознаванием.
        enhance: повышать контраст ROI.
        use_chinese_hack: применять постобработку границ номеров.

    Returns:
        ``DemoResult`` с найденными номерами и таймингами стадий.

    Raises:
        FileNotFoundError: изображение не читается.
    """
    path = Path(image_path)
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    height, width = image.shape[:2]

    started = time.perf_counter()
    detections = detector.predict(path, conf=conf, imgsz=imgsz)
    detection_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    plates: list[PlateResult] = []
    for detection in detections:
        box = (detection.x1, detection.y1, detection.x2, detection.y2)
        roi = extract_roi(image, box, expand_pixels=expand_pixels, enhance=enhance)
        raw = recognizer.recognize(roi)
        plates.append(
            PlateResult(
                x1=detection.x1,
                y1=detection.y1,
                x2=detection.x2,
                y2=detection.y2,
                confidence=float(detection.confidence),
                raw=raw,
                plate=postprocess_plate(raw, enable_hack=use_chinese_hack),
            )
        )
    ocr_ms = (time.perf_counter() - started) * 1000

    return DemoResult(
        plates=plates,
        width=width,
        height=height,
        detection_ms=detection_ms,
        ocr_ms=ocr_ms,
    )


def confidence_color(confidence: float) -> tuple[int, int, int]:
    """Цвет бокса по уверенности детектора (BGR)."""
    if confidence >= CONF_GOOD:
        return _COLOR_GOOD
    if confidence >= CONF_MEDIUM:
        return _COLOR_MEDIUM
    return _COLOR_LOW


def draw_detections(
    image_bgr: np.ndarray,
    plates: Iterable[PlateResult],
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Рисует боксы с подписями ``#номер`` и уверенностью.

    Сам текст номера на изображение не наносится: ``cv2.putText`` умеет только
    ASCII, а в номерах есть китайские символы — распознанный текст показывается
    в таблице рядом. Бокс раскрашивается по уверенности
    (``CONF_GOOD`` / ``CONF_MEDIUM``).

    Args:
        image_bgr: исходное изображение в BGR.
        plates: найденные номера.
        thickness: толщина рамки.
        font_scale: размер шрифта подписи.

    Returns:
        Новое изображение в RGB (исходное не изменяется).
    """
    canvas = image_bgr.copy()
    height, width = canvas.shape[:2]

    for index, plate in enumerate(plates, start=1):
        color = confidence_color(plate.confidence)
        cv2.rectangle(canvas, (plate.x1, plate.y1), (plate.x2, plate.y2), color, thickness)

        label = f"#{index} {plate.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        top = plate.y1 - text_h - baseline - 4
        if top < 0:  # над боксом нет места — подпись уходит под него
            top = min(plate.y1 + 2, max(0, height - text_h - baseline - 4))
        left = max(0, min(plate.x1, width - text_w - 6))

        cv2.rectangle(
            canvas, (left, top), (left + text_w + 6, top + text_h + baseline + 4), color, -1
        )
        cv2.putText(
            canvas,
            label,
            (left + 3, top + text_h + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def plates_to_rows(plates: Iterable[PlateResult]) -> list[dict]:
    """Готовит найденные номера к показу в таблице и выгрузке в CSV."""
    return [
        {
            "№": index,
            "уверенность": round(plate.confidence, 4),
            "x1": plate.x1,
            "y1": plate.y1,
            "x2": plate.x2,
            "y2": plate.y2,
            "OCR (сырое)": plate.raw,
            "номер": plate.plate,
        }
        for index, plate in enumerate(plates, start=1)
    ]


def crop_plate(image_bgr: np.ndarray, plate: PlateResult) -> np.ndarray:
    """Возвращает вырезку номера в RGB (для показа рядом с результатом)."""
    return extract_roi(image_bgr, plate.box)


def encode_png(image_rgb: np.ndarray) -> bytes:
    """Кодирует RGB-изображение в PNG (для кнопки скачивания)."""
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Не удалось закодировать изображение в PNG")
    return buffer.tobytes()


@contextlib.contextmanager
def temporary_image(data: bytes, suffix: str = ".jpg") -> Iterator[Path]:
    """Пишет байты изображения во временный файл и удаляет его после прогона.

    Детектор Ultralytics принимает путь к файлу, поэтому загруженная в браузере
    картинка сначала сохраняется на диск.
    """
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(data)
        path = Path(handle.name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)
