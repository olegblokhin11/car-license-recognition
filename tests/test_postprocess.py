"""Тесты постобработки OCR (включая китайский хак)."""

from plates.ocr.postprocess import apply_chinese_hack, postprocess_plate


def test_normal_plate_unchanged():
    assert postprocess_plate("AB123CD") == "AB123CD"


def test_lowercase_upper():
    assert postprocess_plate(" ab12cd ") == "AB12CD"


def test_chinese_hack():
    # ложнораспознанный граничный номер
    assert apply_chinese_hack("ZAB12") == "粤ZAB12港"


def test_hack_does_not_break_normal():
    assert apply_chinese_hack("ZAB123") == "ZAB123"  # длина != 5
    assert apply_chinese_hack("ABCDE") == "ABCDE"  # не начинается с Z


def test_hack_disabled():
    assert postprocess_plate("ZAB12", enable_hack=False) == "ZAB12"
