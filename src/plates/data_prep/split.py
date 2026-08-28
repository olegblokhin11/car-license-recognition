"""Разделение аннотаций на train/val по уникальным изображениям."""

from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd


@dataclass
class SplitStats:
    """Статистика по сплиту."""

    total_images: int
    train_images: int
    val_images: int
    train_annotations: int
    val_annotations: int


def split_by_images(
    df: pd.DataFrame,
    val_fraction: float = 0.2,
    seed: int = 24,
    image_col: str = "image_name",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит датафрейм на train/val по уникальным изображениям.

    Отличие от split-po-strok: все аннотации одного изображения
    попадают в одну выборку, исключая утечку между эпохами.

    Args:
        df: датафрейм с аннотациями.
        val_fraction: доля изображений на валидацию (0..1).
        seed: seed для перемешивания.
        image_col: колонка с именем изображения.

    Returns:
        ``(train_df, val_df)``.
    """
    if not 0 <= val_fraction < 1:
        raise ValueError(f"val_fraction должен быть в [0, 1), получено {val_fraction}")

    unique = list(df[image_col].unique())
    rng = random.Random(seed)
    rng.shuffle(unique)

    val_len = int(round(val_fraction * len(unique)))
    val_images = set(unique[:val_len])
    train_images = set(unique[val_len:])

    train_df = df[df[image_col].isin(train_images)].copy()
    val_df = df[df[image_col].isin(val_images)].copy()
    return train_df, val_df


def split_statistics(
    df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame
) -> SplitStats:
    """Подсчитывает статистику сплита для логирования."""
    return SplitStats(
        total_images=int(df["image_name"].nunique()),
        train_images=int(train_df["image_name"].nunique()),
        val_images=int(val_df["image_name"].nunique()),
        train_annotations=int(len(train_df)),
        val_annotations=int(len(val_df)),
    )