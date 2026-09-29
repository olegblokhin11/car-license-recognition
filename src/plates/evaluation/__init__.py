"""Метрики оценки качества распознавания номеров (CER/CRR) и пайплайна."""

from .metrics import (
    CRRResult,
    character_error_rate,
    clean_text,
    levenshtein_distance,
    summarize_crr,
)
from .pipeline_eval import (
    EvalImage,
    PipelineMetrics,
    PlateRef,
    box_iou,
    build_eval_images,
    crr_to_dict,
    evaluate_ocr_only,
    evaluate_pipeline,
    match_boxes,
    pipeline_metrics_to_dict,
    resolve_image_path,
)

__all__ = [
    "CRRResult",
    "EvalImage",
    "PipelineMetrics",
    "PlateRef",
    "box_iou",
    "build_eval_images",
    "character_error_rate",
    "clean_text",
    "crr_to_dict",
    "evaluate_ocr_only",
    "evaluate_pipeline",
    "levenshtein_distance",
    "match_boxes",
    "pipeline_metrics_to_dict",
    "resolve_image_path",
    "summarize_crr",
]
