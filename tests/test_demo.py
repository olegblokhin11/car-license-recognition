"""Тесты демо-логики: поиск весов, прогон пайплайна, отрисовка, выгрузка."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

from plates.demo import (
    PlateResult,
    discover_weights,
    draw_detections,
    encode_png,
    plates_to_rows,
    run_pipeline,
    temporary_image,
)


@dataclass
class _Detection:
    """Заглушка ``DetectionBox`` из ``detection.predict``."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


class _StubDetector:
    """Детектор, возвращающий заранее заданные боксы."""

    def __init__(self, detections: list[_Detection]):
        self.detections = detections
        self.calls: list[dict] = []

    def predict(self, image_path, conf: float = 0.25, imgsz: int = 960) -> list[_Detection]:
        self.calls.append({"path": Path(image_path), "conf": conf, "imgsz": imgsz})
        return list(self.detections)


class _StubRecognizer:
    """Распознаватель, отдающий заданные строки и запоминающий размеры ROI."""

    def __init__(self, texts: list[str]):
        self.texts = list(texts)
        self.roi_shapes: list[tuple] = []

    def recognize(self, roi: np.ndarray) -> str:
        self.roi_shapes.append(roi.shape)
        return self.texts.pop(0) if self.texts else ""


def _write_image(path: Path, width: int = 120, height: int = 60) -> Path:
    """Пишет синтетическое изображение с чёрным прямоугольником."""
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (10, 20), (100, 40), (0, 0, 0), -1)
    cv2.imwrite(str(path), image)
    return path


def _read_bgr(path: Path) -> np.ndarray:
    """Читает изображение как массив BGR (для тестов отрисовки)."""
    image = cv2.imread(str(path))
    assert image is not None
    return image


def test_discover_weights_finds_best_pt(tmp_path: Path) -> None:
    for name in ("exp_b", "exp_a"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "best.pt").write_bytes(b"x")
    (tmp_path / "yolo26m.pt").write_bytes(b"x")  # предобученные веса — не эксперимент

    found = discover_weights(tmp_path)

    assert [path.parent.name for path in found] == ["exp_a", "exp_b"]


def test_discover_weights_without_directory(tmp_path: Path) -> None:
    assert discover_weights(tmp_path / "нет-такого-каталога") == []


def test_run_pipeline_returns_plates_after_postprocessing(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "car.jpg")
    detector = _StubDetector([_Detection(10, 20, 100, 40, 0.9)])
    recognizer = _StubRecognizer(["ZAB12"])

    result = run_pipeline(image, detector, recognizer)

    assert len(result.plates) == 1
    plate = result.plates[0]
    assert plate.box == (10, 20, 100, 40)
    assert plate.confidence == pytest.approx(0.9)
    assert plate.raw == "ZAB12"
    assert plate.plate == "粤ZAB12港"  # «китайский хак» включён
    assert (result.width, result.height) == (120, 60)
    assert result.mean_confidence == pytest.approx(0.9)
    assert result.total_ms >= 0


def test_run_pipeline_without_detections(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "car.jpg")
    recognizer = _StubRecognizer([])

    result = run_pipeline(image, _StubDetector([]), recognizer)

    assert result.plates == []
    assert result.mean_confidence == 0.0
    assert recognizer.roi_shapes == []


def test_run_pipeline_passes_parameters_to_detector(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "car.jpg")
    detector = _StubDetector([])

    run_pipeline(image, detector, _StubRecognizer([]), conf=0.4, imgsz=1280)

    assert detector.calls == [{"path": image, "conf": 0.4, "imgsz": 1280}]


def test_run_pipeline_expands_roi(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "car.jpg")
    detector = _StubDetector([_Detection(10, 20, 100, 40, 0.9)])

    plain = _StubRecognizer(["AB1234"])
    run_pipeline(image, detector, plain)

    expanded = _StubRecognizer(["AB1234"])
    run_pipeline(image, detector, expanded, expand_pixels=5)

    assert plain.roi_shapes[0] == (20, 90, 3)  # высота, ширина, каналы
    assert expanded.roi_shapes[0] == (30, 100, 3)


def test_run_pipeline_can_disable_chinese_hack(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "car.jpg")
    detector = _StubDetector([_Detection(10, 20, 100, 40, 0.9)])

    result = run_pipeline(image, detector, _StubRecognizer(["ZAB12"]), use_chinese_hack=False)

    assert result.plates[0].plate == "ZAB12"


def test_run_pipeline_missing_image(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_pipeline(tmp_path / "нет.jpg", _StubDetector([]), _StubRecognizer([]))


def test_draw_detections_marks_box_and_keeps_source(tmp_path: Path) -> None:
    image = _read_bgr(_write_image(tmp_path / "car.jpg"))
    original = image.copy()
    plates = [PlateResult(10, 20, 100, 40, 0.9, "AB1234", "AB1234")]

    drawn = draw_detections(image, plates)

    assert drawn.shape == image.shape
    assert drawn.dtype == np.uint8
    assert np.array_equal(image, original)  # исходное изображение не меняется
    assert not np.array_equal(drawn, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # бокс нарисован
    # пиксель на границе бокса совпадает с цветом «высокой» уверенности (в RGB)
    assert tuple(drawn[20, 55]) == (60, 190, 60)


def test_draw_detections_without_plates_returns_copy(tmp_path: Path) -> None:
    image = _read_bgr(_write_image(tmp_path / "car.jpg"))

    drawn = draw_detections(image, [])

    assert np.array_equal(drawn, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def test_plates_to_rows() -> None:
    plates = [
        PlateResult(1, 2, 3, 4, 0.87654, "ab123", "AB123"),
        PlateResult(5, 6, 7, 8, 0.5, "", ""),
    ]

    rows = plates_to_rows(plates)

    assert [row["№"] for row in rows] == [1, 2]
    assert rows[0]["уверенность"] == 0.8765
    assert rows[0]["номер"] == "AB123"
    assert rows[1]["OCR (сырое)"] == ""


def test_encode_png(tmp_path: Path) -> None:
    image = _read_bgr(_write_image(tmp_path / "car.jpg"))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    data = encode_png(rgb)

    assert data.startswith(b"\x89PNG\r\n\x1a\n")


def test_temporary_image_creates_and_removes_file() -> None:
    payload = b"\x00\x01\x02 not an image, only bytes"

    with temporary_image(payload, suffix=".jpg") as path:
        assert path.exists()
        assert path.read_bytes() == payload

    assert not path.exists()
