"""Загрузка и базовый анализ сырой разметки.

Исходный формат — CSV с колонками
``image_name, x_1, y_1, x_2, y_2, plate`` (Pascal-VOC боксы,
каждая строка — один номер на изображении).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ANNOTATION_COLUMNS = ["image_name", "x_1", "y_1", "x_2", "y_2", "plate"]


def load_annotations(path: str | Path) -> pd.DataFrame:
    """Загружает файл аннотаций в DataFrame.

    Args:
        path: путь к ``*.txt`` / ``*.csv`` с заголовком.

    Returns:
        DataFrame c колонками, типизированными: координаты → ``float``,
        ``plate`` → ``str``.

    Raises:
        ValueError: если отсутствуют обязательные колонки.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл аннотаций не найден: {path}")

    df = pd.read_csv(path, sep=",")
    missing = [c for c in ANNOTATION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"В файле аннотаций отсутствуют колонки: {missing}. "
            f"Реальные колонки: {list(df.columns)}"
        )

    for col in ["x_1", "y_1", "x_2", "y_2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["plate"] = df["plate"].astype(str)

    n_boxes = len(df)
    unique_images = df["image_name"].nunique()
    n_invalid = int(df[["x_1", "y_1", "x_2", "y_2"]].isna().any(axis=1).sum())
    return df


def dataset_statistics(df: pd.DataFrame) -> dict:
    """Базовые статистики по аннотациям: число строк, изображений, боксов.

    Returns:
        Словарь: ``total_annotations``, ``total_images``,
        ``plates_per_image`` (array), ``sample_boxes`` (float).
    """
    counts = df.groupby("image_name").size()
    box_areas = (
        (df["x_2"] - df["x_1"]) * (df["y_2"] - df["y_1"])
    )
    return {
        "total_annotations": int(len(df)),
        "total_images": int(df["image_name"].nunique()),
        "min_plates_per_image": int(counts.min() if len(counts) else 0),
        "max_plates_per_image": int(counts.max() if len(counts) else 0),
        "mean_plates_per_image": float(counts.mean() if len(counts) else 0),
        "box_area_median": float(box_areas.median()),
    }