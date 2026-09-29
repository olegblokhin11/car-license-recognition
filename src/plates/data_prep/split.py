"""Разделение аннотаций на train/val по уникальным изображениям."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

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

    val_len = round(val_fraction * len(unique))
    val_images = set(unique[:val_len])
    train_images = set(unique[val_len:])

    train_df = df[df[image_col].isin(train_images)].copy()
    val_df = df[df[image_col].isin(val_images)].copy()
    return train_df, val_df


def split_statistics(df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame) -> SplitStats:
    """Подсчитывает статистику сплита для логирования."""
    return SplitStats(
        total_images=int(df["image_name"].nunique()),
        train_images=int(train_df["image_name"].nunique()),
        val_images=int(val_df["image_name"].nunique()),
        train_annotations=len(train_df),
        val_annotations=len(val_df),
    )


def list_split_images(dataset_root: str | Path, split: str = "val") -> set[str]:
    """Возвращает имена файлов раздела уже собранного YOLO-датасета.

    Ожидается структура ``<dataset_root>/images/<split>/*.jpg``
    (см. ``YoloPaths`` в ``build_yolo``).

    Args:
        dataset_root: корень собранного датасета.
        split: имя раздела — ``"train"`` или ``"val"``.

    Returns:
        Множество имён файлов (basename) без путей.

    Raises:
        FileNotFoundError: если каталога раздела нет.
    """
    images_dir = Path(dataset_root) / "images" / split
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Каталог раздела не найден: {images_dir}")
    return {p.name for p in images_dir.iterdir() if p.is_file()}


def filter_annotations_by_split(
    df: pd.DataFrame,
    dataset_root: str | Path,
    split: str = "val",
    image_col: str = "image_name",
) -> pd.DataFrame:
    """Оставляет аннотации только для изображений, попавших в раздел датасета.

    Сопоставление идёт по имени файла (basename), поэтому префиксы каталогов
    в колонке ``image_name`` (например ``train/000000_0.jpg``) не мешают.

    Args:
        df: датафрейм с аннотациями.
        dataset_root: корень собранного YOLO-датасета.
        split: имя раздела — ``"train"`` или ``"val"``.
        image_col: колонка с именем изображения.

    Returns:
        Копия ``df`` со строками выбранного раздела.
    """
    names = list_split_images(dataset_root, split)
    mask = df[image_col].map(lambda name: Path(str(name)).name in names)
    return df[mask].copy()
