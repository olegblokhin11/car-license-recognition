#!/usr/bin/env python3
"""CLI: подготовка YOLO-датасета детекции номеров.

Пример:
    python scripts/prepare_data.py --data-root data/raw \\
        --annotation data/raw/annotation/train_annot.txt \\
        --out dataset_for_yolo --val-fraction 0.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Позволяет import src.plates как пакет без установки.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plates.data_prep import (
    build_yolo_dataset,
    dataset_statistics,
    load_annotations,
    split_by_images,
    split_statistics,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", dest="image_dir", type=str, default="data/raw/images", help="images root")
    p.add_argument(
        "--annotation",
        type=str,
        default="data/raw/annotation/train_annot.txt",
    )
    p.add_argument("--out", type=str, default="dataset_for_yolo")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=24)
    return p


def main() -> int:
    args = build_parser().parse_args()

    df = load_annotations(args.annotation)
    print(f"Загружено аннотаций: {len(df)}")
    stats = dataset_statistics(df)
    print("Статистика:", stats)

    train_df, val_df = split_by_images(
        df, val_fraction=args.val_fraction, seed=args.seed
    )
    sp = split_statistics(df, train_df, val_df)
    print(
        f"Сплит: train {sp.train_images} img / {sp.train_annotations} ann, "
        f"val {sp.val_images} img / {sp.val_annotations} ann"
    )

    build_yolo_dataset(
        train_df, val_df, image_source_dir=args.image_dir, dataset_root=args.out
    )
    print(f"YOLO-датасет собран в: {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())