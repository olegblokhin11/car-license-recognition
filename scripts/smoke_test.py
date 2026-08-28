#!/usr/bin/env python3
"""Smoke-тест всего пайплайна (GPU) — детекция + OCR на малой выборке.

Запуск:
    python scripts/smoke_test.py --images data/raw/images/test --limit 20

Требует установленных тяжёлых зависимостей и наличия весов моделей.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plates.detection import YoloDetector
from plates.ocr import build_recognizer, postprocess_plate


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=str, default="assets/base_ext_v2/best.pt")
    p.add_argument("--images", type=str, required=True)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--n", type=int, default=20, help="сколько изображений прогнать")
    return p


def main() -> int:
    args = build_parser().parse_args()
    images = sorted(Path(args.images).glob("*.jpg"))[: args.n]
    print(f"Сэмпл: {len(images)} изображений")

    # 1) Детекция
    t0 = time.time()
    detector = YoloDetector(args.weights)
    total_dets = 0
    for img in images:
        dets = detector.predict(img, conf=args.conf, imgsz=args.imgsz)
        total_dets += len(dets)
    det_time = time.time() - t0
    print(f"Детекция: {total_dets} боксов за {det_time:.1f}s "
          f"({det_time/len(images):.2f}s/img)")

    # 2) OCR
    t0 = time.time()
    recognizer = build_recognizer("fast_plate_ocr")
    plates = []
    for img in images:
        dets = detector.predict(img, conf=args.conf, imgsz=args.imgsz)
        import cv2

        cv_img = cv2.imread(str(img))
        for d in dets:
            roi = cv_img[d.y1:d.y2, d.x1:d.x2]
            if roi.size == 0:
                continue
            raw = recognizer.recognize(roi)
            plates.append(postprocess_plate(raw))
    ocr_time = time.time() - t0
    print(f"OCR: {len(plates)} распознано за {ocr_time:.2f}s")
    print("Примеры:", plates[:10])
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())