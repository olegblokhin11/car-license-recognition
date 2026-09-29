#!/usr/bin/env python3
"""CLI: обучение YOLO-детекции номеров.

Пример:
    python scripts/train_detector.py \\
        --data dataset_for_yolo_extended_v2/data.yaml \\
        --cfg configs/detection/default_ext_v2.yaml \\
        --model yolo26m_ocr.yaml --weights yolo26m.pt --device 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plates.detection import train


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=str, required=True, help="data.yaml датасета")
    p.add_argument("--cfg", type=str, default=None, help="глобальный cfg обучения")
    p.add_argument("--model", type=str, default="yolo26m_ocr.yaml")
    p.add_argument("--weights", type=str, default="yolo26m.pt")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--project", type=str, default=None)
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    kwargs = {}
    if args.epochs:
        kwargs["epochs"] = args.epochs
    if args.imgsz:
        kwargs["imgsz"] = args.imgsz

    train(
        data_yaml=args.data,
        cfg_yaml=args.cfg,
        model_yaml=args.model,
        pretrained_weights=args.weights,
        device=args.device,
        project=args.project,
        name=args.name,
        **kwargs,
    )
    print("Обучение завершено.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
