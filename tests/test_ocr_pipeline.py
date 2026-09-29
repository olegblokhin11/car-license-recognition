"""Тест OCR-пайплайна с mock-распознавателем (без тяжёлых зависимостей)."""

import numpy as np

from plates.ocr import recognize_images
from plates.ocr.recognizer import PlateRecognizer


class FakeRecognizer(PlateRecognizer):
    """Возвращает заранее заданный номер для любой ROI."""

    def __init__(self, result: str = "AB123CD"):
        self.result = result
        self.calls = 0

    def recognize(self, roi: np.ndarray) -> str:
        self.calls += 1
        return self.result


def test_recognize_images_and_postprocess(tmp_path):
    import cv2

    img = np.full((100, 200, 3), 180, dtype=np.uint8)
    p = tmp_path / "x.jpg"
    cv2.imwrite(str(p), img)

    rec = FakeRecognizer("ab12cd")
    preds = recognize_images(
        str(tmp_path),
        ["x.jpg"],
        [(10, 10, 60, 50)],
        rec,
        expand_pixels=0,
        enhance=False,
    )
    assert len(preds) == 1
    assert preds[0].plate == "AB12CD"  # верхний регистр и очистка


def test_extract_roi_expand(tmp_path):
    from plates.ocr.pipeline import extract_roi

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    roi = extract_roi(img, (20, 20, 40, 40), expand_pixels=10)
    # 40-20 + 2*10 = 40 по каждой оси
    assert roi.shape[:2] == (40, 40)


def test_recognize_images_no_x(tmp_path):
    rec = FakeRecognizer()
    preds = recognize_images(str(tmp_path), ["missing.jpg"], [(0, 0, 10, 10)], rec)
    assert len(preds) == 1
    assert preds[0].plate == ""
