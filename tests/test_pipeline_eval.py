"""Тесты оценки пайплайна: IoU, сопоставление боксов, учёт пропусков."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plates.data_prep import filter_annotations_by_split, list_split_images
from plates.evaluation import (
    EvalImage,
    PlateRef,
    box_iou,
    build_eval_images,
    crr_to_dict,
    evaluate_ocr_only,
    evaluate_pipeline,
    match_boxes,
    pipeline_metrics_to_dict,
)


class _StubDetector:
    """Детектор-заглушка: отдаёт заранее заданные боксы по имени файла."""

    def __init__(self, boxes_by_image: dict[str, list[tuple]]):
        self.boxes_by_image = boxes_by_image
        self.calls: list[tuple] = []

    def predict(self, image_path, conf=0.25, imgsz=960):
        self.calls.append((str(image_path), conf, imgsz))
        from plates.detection import DetectionBox

        return [DetectionBox(*box, 0.9) for box in self.boxes_by_image.get(str(image_path), [])]


class _StubRecognizer:
    """Распознаватель-заглушка: отдаёт тексты по очереди."""

    def __init__(self, texts: list[str]):
        self.texts = list(texts)
        self.seen_rois = 0

    def recognize(self, roi) -> str:
        self.seen_rois += 1
        return self.texts.pop(0) if self.texts else ""


def _write_image(path, size=(40, 20)) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
    cv2.imwrite(str(path), image)


# ------------------------------------------------------------------ box_iou
def test_box_iou_identical_and_disjoint():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_box_iou_partial_overlap():
    # пересечение 5x10 = 50, объединение 100 + 100 - 50 = 150
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_box_iou_degenerate_box_is_zero():
    assert box_iou((0, 0, 0, 0), (0, 0, 10, 10)) == 0.0


# --------------------------------------------------------------- match_boxes
def test_match_boxes_one_to_one():
    dets = [(0, 0, 10, 10)]
    gts = [(0, 0, 10, 10), (50, 50, 60, 60)]
    assert match_boxes(dets, gts) == [(0, 0)]


def test_match_boxes_respects_threshold():
    dets = [(0, 0, 10, 10)]
    gts = [(0, 0, 10, 11)]  # IoU ≈ 0.909
    assert match_boxes(dets, gts, iou_thr=0.95) == []
    assert match_boxes(dets, gts, iou_thr=0.9) == [(0, 0)]


def test_match_boxes_does_not_reuse_gt():
    dets = [(0, 0, 10, 10), (1, 1, 11, 11)]
    gts = [(0, 0, 10, 10)]
    # обе детекции претендуют на единственный эталон, но он достаётся одной
    assert len(match_boxes(dets, gts)) == 1
    assert match_boxes(dets, gts)[0][1] == 0


def test_match_boxes_picks_best_gt_for_each_detection():
    dets = [(0, 0, 10, 10), (100, 100, 110, 110)]
    gts = [(100, 100, 110, 110), (0, 0, 10, 10)]
    assert sorted(match_boxes(dets, gts)) == [(0, 1), (1, 0)]


# ------------------------------------------------------- evaluate_pipeline
def test_evaluate_pipeline_counts_missed_plates(tmp_path):
    """Один номер найден, второй пропущен: e2e хуже conditional."""
    path = tmp_path / "a.jpg"
    _write_image(path)
    images = [
        EvalImage(
            path=path,
            plates=[
                PlateRef(box=(0, 0, 20, 10), text="AB123"),
                PlateRef(box=(20, 0, 40, 10), text="CD456"),
            ],
        )
    ]
    detector = _StubDetector({str(path): [(0, 0, 20, 10)]})
    recognizer = _StubRecognizer(["AB123"])

    metrics = evaluate_pipeline(images, detector, recognizer, conf=0.25, imgsz=960)

    assert metrics.n_images == 1
    assert metrics.n_gt == 2
    assert metrics.n_detections == 1
    assert metrics.n_matched == 1
    assert metrics.n_missed == 1
    assert metrics.n_false_positives == 0
    assert metrics.detection_recall == pytest.approx(0.5)
    # найденный номер распознан точно
    assert metrics.conditional.crr == pytest.approx(1.0)
    assert metrics.conditional.total == 1
    # пропущенный номер считается ошибкой -> CRR 0.5
    assert metrics.end_to_end.total == 2
    assert metrics.end_to_end.crr == pytest.approx(0.5)


def test_evaluate_pipeline_counts_false_positives(tmp_path):
    """Лишняя детекция не попадает в CRR, но видна в счётчиках."""
    path = tmp_path / "a.jpg"
    _write_image(path)
    images = [EvalImage(path=path, plates=[PlateRef(box=(0, 0, 20, 10), text="AB123")])]
    detector = _StubDetector({str(path): [(0, 0, 20, 10), (20, 0, 40, 10)]})
    recognizer = _StubRecognizer(["AB123"])

    metrics = evaluate_pipeline(images, detector, recognizer)

    assert metrics.n_detections == 2
    assert metrics.n_matched == 1
    assert metrics.n_false_positives == 1
    assert metrics.n_missed == 0
    assert metrics.end_to_end.crr == pytest.approx(1.0)


def test_evaluate_pipeline_empty_selection():
    metrics = evaluate_pipeline([], _StubDetector({}), _StubRecognizer([]))
    assert metrics.n_images == 0
    assert metrics.n_gt == 0
    assert metrics.detection_recall == 0.0
    assert metrics.conditional.crr == 0.0


def test_evaluate_pipeline_passes_conf_and_imgsz(tmp_path):
    path = tmp_path / "a.jpg"
    _write_image(path)
    images = [EvalImage(path=path, plates=[PlateRef(box=(0, 0, 20, 10), text="AB123")])]
    detector = _StubDetector({str(path): []})

    evaluate_pipeline(images, detector, _StubRecognizer([]), conf=0.4, imgsz=640)

    assert detector.calls == [(str(path), 0.4, 640)]


# --------------------------------------------------------- evaluate_ocr_only
def test_evaluate_ocr_only_uses_gt_boxes(tmp_path):
    path = tmp_path / "a.jpg"
    _write_image(path)
    images = [
        EvalImage(
            path=path,
            plates=[
                PlateRef(box=(0, 0, 20, 10), text="AB123"),
                PlateRef(box=(20, 0, 40, 10), text="CD456"),
            ],
        )
    ]
    recognizer = _StubRecognizer(["AB123", "CD456"])

    result = evaluate_ocr_only(images, recognizer)

    assert result.total == 2
    assert result.crr == pytest.approx(1.0)
    assert recognizer.seen_rois == 2


# --------------------------------------------------- split / eval images
def test_list_and_filter_by_split(tmp_path):
    val_dir = tmp_path / "images" / "val"
    val_dir.mkdir(parents=True)
    (val_dir / "a.jpg").write_bytes(b"x")

    df = pd.DataFrame(
        {
            "image_name": ["train/a.jpg", "train/b.jpg", "train/c.jpg"],
            "x_1": [0, 0, 0],
            "y_1": [0, 0, 0],
            "x_2": [1, 1, 1],
            "y_2": [1, 1, 1],
            "plate": ["A1", "B2", "C3"],
        }
    )

    assert list_split_images(tmp_path, "val") == {"a.jpg"}
    filtered = filter_annotations_by_split(df, tmp_path, "val")
    # сопоставление по basename: 'train/a.jpg' попадает в val-раздел
    assert filtered["image_name"].tolist() == ["train/a.jpg"]


def test_list_split_images_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_split_images(tmp_path, "val")


def test_build_eval_images_groups_plates_and_resolves_prefix(tmp_path):
    image_path = tmp_path / "train" / "a.jpg"
    _write_image(image_path)
    df = pd.DataFrame(
        {
            "image_name": ["train/a.jpg", "train/a.jpg", "train/missing.jpg"],
            "x_1": [0, 20, 0],
            "y_1": [0, 0, 0],
            "x_2": [20, 40, 10],
            "y_2": [10, 10, 10],
            "plate": ["AB123", "CD456", "ZZ999"],
        }
    )

    images = build_eval_images(df, tmp_path)

    assert len(images) == 1  # missing.jpg пропущен
    assert images[0].path == image_path
    assert [plate.text for plate in images[0].plates] == ["AB123", "CD456"]


def test_metrics_to_dict_is_json_friendly():
    metrics = evaluate_pipeline([], _StubDetector({}), _StubRecognizer([]))
    payload = pipeline_metrics_to_dict(metrics)

    assert payload["n_gt"] == 0
    assert payload["conditional"] == crr_to_dict(metrics.conditional)
    assert set(payload["end_to_end"]) == {"n", "exact_matches", "exact_match_rate", "cer", "crr"}
