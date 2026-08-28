"""Интерфейс распознавателя номеров.

Основной вариант — fast_plate_ocr. Дополнительно описан ABC, чтобы
в будущем подставлять другие модели (PaddleOCR, пер-char CNN и т.п.)
без изменения пайплайна оценки.
"""

from __future__ import annotations

import abc

import numpy as np


class PlateRecognizer(abc.ABC):
    """Абстрактный распознаватель номера по обрезанному изображению."""

    @abc.abstractmethod
    def recognize(self, roi: np.ndarray) -> str:
        """Распознаёт номер на изображении ROI (BGR или RGB ndarray).

        Returns:
            Сырая строка номера (до постобработки), либо ``""`` при ошибке.
        """


class FastPlateRecognizer(PlateRecognizer):
    """Распознаватель поверх fast_plate_ocr."""

    def __init__(self, model_name: str = "cct-s-v2-global-model", **provider_kwargs):
        from fast_plate_ocr import LicensePlateRecognizer  # ленивый импорт

        providers = provider_kwargs.pop(
            "providers", ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self._recognizer = LicensePlateRecognizer(model_name, providers=providers)
        self.model_name = model_name

    def recognize(self, roi: np.ndarray) -> str:
        try:
            result = self._recognizer.run(roi)[0]
            return result.plate or ""
        except Exception:  # noqa: BLE001 - OCR может падать на плохом кадре
            return ""


def build_recognizer(
    backend: str = "fast_plate_ocr",
    model_name: str = "cct-s-v2-global-model",
    **kwargs,
) -> PlateRecognizer:
    """Фабрика распознавателя по имени бэкенда."""
    if backend == "fast_plate_ocr":
        return FastPlateRecognizer(model_name=model_name, **kwargs)
    raise ValueError(f"Неизвестный бэкенд распознавателя: {backend}")