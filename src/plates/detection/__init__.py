"""Детекция номеров: обучение и инференс YOLO."""

from .predict import DetectionBox, YoloDetector, predict_directory
from .train import build_model, train

__all__ = [
    "DetectionBox",
    "YoloDetector",
    "build_model",
    "predict_directory",
    "train",
]
