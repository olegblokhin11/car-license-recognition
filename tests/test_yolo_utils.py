"""Тесты утилит конвертации боксов."""

import pytest

from plates.utils.yolo import (
    format_label_line,
    pascal_to_yolo,
    pascal_to_yolo_one,
    yolo_to_pascal_one,
)


def test_pascal_to_yolo_basic():
    box = pascal_to_yolo_one([0, 0, 100, 50], img_w=200, img_h=100)
    assert box == pytest.approx((0.25, 0.25, 0.5, 0.5))


def test_pascal_to_yolo_negative_zero():
    box = pascal_to_yolo_one([-10, -20, 30, 40], img_w=200, img_h=100)
    # центр будет отрицательный, но width/height корректны
    assert box[2] == pytest.approx(0.2)
    assert box[3] == pytest.approx(0.6)


def test_zero_size_image_raises():
    with pytest.raises(ValueError):
        pascal_to_yolo_one([0, 0, 10, 10], img_w=0, img_h=10)


def test_inverted_box_raises():
    with pytest.raises(ValueError):
        pascal_to_yolo_one([50, 0, 10, 10], img_w=100, img_h=100)


def test_roundtrip():
    pts = [(10, 20, 110, 80), (0, 0, 50, 50), (200, 100, 250, 150)]
    for x1, y1, x2, y2 in pts:
        yolo = pascal_to_yolo_one([x1, y1, x2, y2], 300, 200)
        back = yolo_to_pascal_one(yolo, 300, 200)
        assert back == (x1, y1, x2, y2)


def test_format_label_line():
    line = format_label_line([0.5, 0.5, 0.1, 0.2], class_id=0)
    assert line == "0 0.500000 0.500000 0.100000 0.200000\n"