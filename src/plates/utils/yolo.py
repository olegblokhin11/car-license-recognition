"""Утилиты работы с bounding box координатами.

Преобразование между форматом Pascal-VOC (абсолютные ``x1,y1,x2,y2``)
и форматом YOLO (нормализованные ``x_center, y_center, width, height``).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

Number = float
BoxPascal = Sequence[float]  # (x1, y1, x2, y2)
BoxYolo = Sequence[float]  # (x_center, y_center, width, height)


def pascal_to_yolo_one(box: BoxPascal, img_w: int, img_h: int) -> BoxYolo:
    """Конвертирует один бокс Pascal VOC в YOLO-формат.

    Args:
        box: ``(x1, y1, x2, y2)`` — абсолютные координаты.
        img_w: ширина изображения в пикселях.
        img_h: высота изображения в пикселях.

    Returns:
        ``(x_center, y_center, width, height)`` — все значения в [0, 1].

    Raises:
        ValueError: если изображение вырождено (нулевой размер),
            либо если координаты образуют пустой прямоугольник.
    """
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Некорректный размер изображения: {img_w}x{img_h}")

    x1, y1, x2, y2 = box
    if x2 < x1 or y2 < y1:
        raise ValueError(f"Пустой/инвертированный бокс: {box}")

    x_center = (x1 + x2) / 2.0 / img_w
    y_center = (y1 + y2) / 2.0 / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return (x_center, y_center, width, height)


def pascal_to_yolo(boxes: Iterable[BoxPascal], img_w: int, img_h: int) -> list[BoxYolo]:
    """Конвертирует список боксов Pascal VOC в список боксов YOLO."""
    return [pascal_to_yolo_one(box, img_w, img_h) for box in boxes]


def yolo_to_pascal_one(box: BoxYolo, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Конвертирует один YOLO-бокс обратно в абсолютный Pascal VOC.

    Возвращает целые пиксельные координаты ``(x1, y1, x2, y2)``.
    """
    xc, yc, w, h = box
    x1 = round((xc - w / 2) * img_w)
    y1 = round((yc - h / 2) * img_h)
    x2 = round((xc + w / 2) * img_w)
    y2 = round((yc + h / 2) * img_h)
    return x1, y1, x2, y2


def format_label_line(box: BoxYolo, class_id: int = 0, precision: int = 6) -> str:
    """Форматирует YOLO-бокс в строку файла аннотации.

    Выходит строка вида ``"<class_id> <xc> <yc> <w> <h>\\n"``.
    """
    xc, yc, w, h = box
    return f"{class_id} {xc:.{precision}f} {yc:.{precision}f} {w:.{precision}f} {h:.{precision}f}\n"
