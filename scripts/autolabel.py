#!/usr/bin/env python3
"""CLI: доразметка данных через GroundingDINO и расширение YOLO-датасета.

Пример:
    python scripts/autolabel.py \\
        --images data/raw/images/not_annotated \\
        --existing dataset_for_yolo \\
        --out dataset_for_yolo_extended \\
        --conf 0.35
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plates.autolabel import GroundingLabeler, label_images


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=str, required=True, help="папка с неразмеченными фото")
    p.add_argument("--existing", type=str, required=True, help="уже размеченный датасет")
    p.add_argument("--out", type=str, required=True, help="куда писать расширенный")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--text-threshold", type=float, default=0.3)
    p.add_argument("--device", type=str, default="auto")
    return p


def main() -> int:
    args = build_parser().parse_args()
    labeler = GroundingLabeler(
        conf_threshold=args.conf,
        text_threshold=args.text_threshold,
        device=args.device,
    )
    stats = label_images(
        args.images,
        args.existing,
        args.out,
        labeler=labeler,
        conf_threshold=args.conf,
        text_threshold=args.text_threshold,
    )
    print(
        f"Обработано изображений: {stats.processed_images}, "
        f"найдено номеров: {stats.found_plates}"
    )
    for e in stats.errors[:20]:
        print("  ОШИБКА:", e)
    print(f"Датасет → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())