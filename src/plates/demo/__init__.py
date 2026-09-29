"""Демо: детекция и распознавание номеров на одном изображении."""

from .inference import (
    CONF_GOOD,
    CONF_MEDIUM,
    DemoResult,
    PlateResult,
    discover_weights,
    draw_detections,
    encode_png,
    plates_to_rows,
    run_pipeline,
    temporary_image,
)

__all__ = [
    "CONF_GOOD",
    "CONF_MEDIUM",
    "DemoResult",
    "PlateResult",
    "discover_weights",
    "draw_detections",
    "encode_png",
    "plates_to_rows",
    "run_pipeline",
    "temporary_image",
]
