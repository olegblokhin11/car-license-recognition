"""OCR: постобработка, распознаватель и пайплайн."""

from .pipeline import (
    PlatePrediction,
    evaluate_predictions,
    extract_roi,
    recognize_images,
)
from .postprocess import apply_chinese_hack, postprocess_plate
from .recognizer import FastPlateRecognizer, PlateRecognizer, build_recognizer

__all__ = [
    "FastPlateRecognizer",
    "PlatePrediction",
    "PlateRecognizer",
    "apply_chinese_hack",
    "build_recognizer",
    "evaluate_predictions",
    "extract_roi",
    "postprocess_plate",
    "recognize_images",
]
