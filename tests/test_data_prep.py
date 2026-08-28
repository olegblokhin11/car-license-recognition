"""Интеграционный тесты стадии подготовки данных."""

import cv2
import numpy as np
import pandas as pd
import pytest

from plates.data_prep import build_yolo_dataset, load_annotations, split_by_images
from plates.utils.yolo import (
    format_label_line,
    pascal_to_yolo_one,
    yolo_to_pascal_one,
)


def _make_image(path, w=100, h=80):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _make_fixture(tmp_path):
    """Создаёт маленький файл аннотаций + два изображения."""
    img_root = tmp_path / "imgs"
    for name in ("a", "b"):
        _make_image(img_root / f"{name}.jpg")

    annot = tmp_path / "annot.txt"
    annot.write_text(
        "image_name,x_1,y_1,x_2,y_2,plate\n"
        "a.jpg,0,0,30,40,ABC123\n"
        "a.jpg,60,20,90,70,XZ789\n"
        "b.jpg,0,0,40,20,DEF456\n"
    )
    return img_root, annot


def test_load_and_split(tmp_path):
    img_root, annot = _make_fixture(tmp_path)
    df = load_annotations(annot)
    assert len(df) == 3
    assert df["image_name"].unique().tolist() == ["a.jpg", "b.jpg"]

    train_df, val_df = split_by_images(df, val_fraction=0.5, seed=1)
    t_imgs = set(train_df["image_name"])
    v_imgs = set(val_df["image_name"])
    assert t_imgs.isdisjoint(v_imgs)


def test_build_dataset_roundtrip(tmp_path):
    img_root, annot = _make_fixture(tmp_path)
    df = load_annotations(annot)
    train_df, val_df = split_by_images(df, val_fraction=0.0, seed=1)
    ds = build_yolo_dataset(
        train_df, val_df, image_source_dir=img_root, dataset_root=tmp_path / "ds"
    )
    train_imgs = list(ds.train_images.glob("*.jpg"))
    assert len(train_imgs) == 2

    label = (ds.train_labels / "a.txt").read_text().strip().splitlines()
    assert len(label) == 2  # у a.jpg пара номеров
    for line in label:
        parts = line.split()
        assert len(parts) == 5


def test_yolo_convert_roundtrip():
    yolo = pascal_to_yolo_one([0, 0, 50, 20], img_w=100, img_h=80)
    back = yolo_to_pascal_one(yolo, img_w=100, img_h=80)
    assert back[0] == 0 and back[1] == 0
    assert back[2] == 50 and back[3] == 20


def test_format_label_line():
    line = format_label_line([0.25, 0.25, 0.5, 0.5], class_id=0)
    assert line == "0 0.250000 0.250000 0.500000 0.500000\n"