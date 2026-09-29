#!/usr/bin/env python3
"""CLI: инференс детекции номеров на изображениях -> CSV предсказаний.

Пример:
    python scripts/predict_detector.py \\
        --weights assets/yolo26m.pt --images data/raw/images/test \\
        --out out/predictions.txt --conf 0.25 --imgsz 960
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plates.detection import YoloDetector, predict_directory


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=str, required=True, help="обученная модель .pt")
    p.add_argument("--images", type=str, required=True, help="папка с изображениями")
    p.add_argument("--out", type=str, required=True, help="куда писать CSV предсказаний")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--prefix", type=str, default="test/", help="префикс image_name")
    return p


def main() -> int:
    args = build_parser().parse_args()
    detector = YoloDetector(args.weights)
    df = predict_directory(
        detector,
        args.images,
        image_prefix=args.prefix,
        conf=args.conf,
        imgsz=args.imgsz,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Сохранено {len(df)} детекций в {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
