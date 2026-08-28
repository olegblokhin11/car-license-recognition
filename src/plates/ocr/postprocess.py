"""Постобработка выхода OCR-распознавателя номеров.

Содержит т.н. «китайский хак»: границы номера Гонконга вида
``粤<текст>港`` распознаватель fast_plate_ocr иногда выдаёт как
латиницу, начинающуюся с ``Z``. Модуль исправляет такие случаи
по эвристике из оригинального ноутбука.
"""

from __future__ import annotations

from ..evaluation.metrics import clean_text  # переиспользуем очистку текста

# Префиксы ходовые для гонконгского «гуаньдунского» формата.
_CN_LEFT = "粤"  # провинция Гуандун (Guangdong)
_CN_RIGHT = "港"  # Hong Kong / Гонконг


def apply_chinese_hack(raw: str) -> str:
    """Исправляет выдачу OCR для леворежур с региональным префиксом.

    Иногда модель возвращает ``ZAB12`` длиной 5, а на самом деле это
    номер ``粤ZAB12港`` (китай-spec граничный номер). Восстанавливаем.

    Args:
        raw: сырой текст от распознавателя.

    Returns:
        Исправленную строку либо исходную без изменений.
    """
    text = clean_text(raw)
    if text.startswith("Z") and len(text) == 5:
        return _CN_LEFT + text + _CN_RIGHT
    return text


def postprocess_plate(raw: str, enable_hack: bool = True) -> str:
    """Полная постобработка сырой строки из OCR.

    Args:
        raw: результат ``recognizer.run(...)[0].plate``.
        enable_hack: применять ли «китайский хак».

    Returns:
        Нормализованный номер (uppercase, без лишних символов).
    """
    text = raw or ""
    if enable_hack:
        return apply_chinese_hack(text)
    return clean_text(text)
