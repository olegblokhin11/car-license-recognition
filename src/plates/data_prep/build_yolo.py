"""Построение структуры YOLO-датасета (images/labels + data.yaml).

Считывает DataFrame с аннотациями Pascal-VOC боксами и раскладывает
по структуре:
::

    <dataset_root>/
        images/
            train/*.jpg
            val/*.jpg
        labels/
            train/*.txt
            val/*.txt
        data.yaml
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd

from ..utils.io import ensure_dir
from ..utils.yolo import format_label_line, pascal_to_yolo

DEFAULT_CLASS_NAMES = ["car_plate"]

logger = logging.getLogger(__name__)


@dataclass
class YoloPaths:
    """Набор путей внутри YOLO-датасета."""

    dataset: Path
    train_images: Path
    train_labels: Path
    val_images: Path
    val_labels: Path

    @classmethod
    def create(cls, dataset_root: str | Path) -> YoloPaths:
        dataset_root = Path(dataset_root)
        paths = cls(
            dataset=dataset_root,
            train_images=dataset_root / "images" / "train",
            train_labels=dataset_root / "labels" / "train",
            val_images=dataset_root / "images" / "val",
            val_labels=dataset_root / "labels" / "val",
        )
        for p in paths.__dict__.values():
            ensure_dir(p)
        return paths


def image_size(image_path: Path) -> tuple[int, int]:
    """Возвращает (width, height) изображения в пикселях."""
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    h, w = img.shape[:2]
    return w, h


def _resolve_source(image_name: str, image_source_dir: Path) -> Path | None:
    """Находит реальный путь изображения.

    Пробует ``image_source_dir / image_name``, затем fallback на
    ``image_source_dir / basename(image_name)`` (если в аннотации
    имя хранится с префиксом каталога).
    """
    direct = image_source_dir / image_name
    if direct.exists():
        return direct
    alt = image_source_dir / Path(image_name).name
    return alt if alt.exists() else None


def write_partition(
    df: pd.DataFrame,
    image_source_dir: str | Path,
    image_dir: str | Path,
    label_dir: str | Path,
) -> dict:
    """Пишет изображения и YOLO-аннотации для одной выборки (train или val).

    Args:
        df: аннотации для выборки (колонки image_name, x_1..y_2).
        image_source_dir: корень, где лежат исходные изображения.
        image_dir: куда класть копии изображений.
        label_dir: куда класть ``.txt`` аннотации.

    Returns:
        Словарь ``{"processed": int, "errors": list[str]}``, где ``processed`` —
        число записанных изображений, ``errors`` — сообщения о ненайденных
        исходниках.
    """
    src_root = Path(image_source_dir)
    dst_images = ensure_dir(image_dir)
    dst_labels = ensure_dir(label_dir)

    processed = 0
    errors: list[str] = []

    for image_name, group in df.groupby("image_name"):
        src = _resolve_source(image_name, src_root)
        if src is None:
            errors.append(f"image not found: {image_name}")
            continue

        dst_image = dst_images / Path(image_name).name
        if not dst_image.exists() or dst_image.resolve() != src.resolve():
            shutil.copy2(src, dst_image)

        img_w, img_h = image_size(dst_image)
        boxes = [
            [r["x_1"], r["y_1"], r["x_2"], r["y_2"]]
            for _, r in group[["x_1", "y_1", "x_2", "y_2"]].iterrows()
        ]
        yolo_boxes = pascal_to_yolo(boxes, img_w, img_h)

        label_path = dst_labels / f"{Path(image_name).stem}.txt"
        with open(label_path, "w") as f:
            for yb in yolo_boxes:
                f.write(format_label_line(yb))

        processed += 1

    return {
        "processed": processed,
        "errors": errors,
    }


def write_data_yaml(
    dataset_root: Path, class_names: list[str] | None = None, yaml_filename: str = "data.yaml"
) -> Path:
    """Создаёт ``data.yaml`` для Ultralytics."""
    class_names = class_names or DEFAULT_CLASS_NAMES
    yaml_path = dataset_root / yaml_filename
    content = (
        f"# YOLO dataset configuration\n"
        f"path: {dataset_root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"nc: {len(class_names)}  # number of classes\n"
        f"names: {class_names}  # class names\n"
    )
    yaml_path.write_text(content)
    return yaml_path


def build_yolo_dataset(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    image_source_dir: str | Path,
    dataset_root: str | Path,
    class_names: list[str] | None = None,
) -> YoloPaths:
    """Собирает полный YOLO-датасет из train/val DataFrame.

    Returns:
        ``YoloPaths``.
    """
    paths = YoloPaths.create(dataset_root)
    train_stats = write_partition(
        train_df, image_source_dir, paths.train_images, paths.train_labels
    )
    val_stats = write_partition(val_df, image_source_dir, paths.val_images, paths.val_labels)
    write_data_yaml(paths.dataset, class_names)

    logger.info(
        "YOLO-датасет собран в %s: train %d изображений, val %d изображений",
        paths.dataset,
        train_stats["processed"],
        val_stats["processed"],
    )
    for partition, stats in (("train", train_stats), ("val", val_stats)):
        errors = stats["errors"]
        if errors:
            logger.warning(
                "%s: не найдено исходников для %d изображений, например: %s",
                partition,
                len(errors),
                ", ".join(errors[:5]),
            )
    return paths
