#!/usr/bin/env python3
"""CLI: полный инференс (детекция + OCR) на папке изображений.

Собирает пайплайн: YOLO-детекция боксов -> обрезка ROI -> распознавание номеров.

Пример:
    python scripts/infer.py \\
        --weights assets/yolo26m.pt --images data/raw/images/test \\
        --out out/inference_results.txt --conf 0.25 --imgsz 960
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2

from plates.ocr import build_recognizer, postprocess_plate


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--images", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--ocr-model", type=str, default="cct-s-v2-global-model")
    return p


def main() -> int:
    from plates.detection import YoloDetector, predict_directory

    args = build_parser().parse_args()

    # 1) Детекция
    detector = YoloDetector(args.weights)
    dets_df = predict_directory(
        detector, args.images, image_prefix="test/", conf=args.conf, imgsz=args.imgsz
    )
    print(f"Детекций: {len(dets_df)}")

    # 2) OCR по боксам
    recognizer = build_recognizer("fast_plate_ocr", args.ocr_model)
    rows = []
    for image_name, group in dets_df.groupby("image_name"):
        # имя без префикса каталога
        rel = image_name.replace("test/", "")
        img_path = Path(args.images) / rel
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        for _, row in group.iterrows():
            x1, y1, x2, y2 = (
                int(row["x_1"]),
                int(row["y_1"]),
                int(row["x_2"]),
                int(row["y_2"]),
            )
            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            raw = recognizer.recognize(roi_rgb)
            rows.append({"image_name": image_name, "plate": postprocess_plate(raw)})

    import pandas as pd

    out_df = pd.DataFrame(rows, columns=["image_name", "plate"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Результат → {out_path}: {len(out_df)} записей")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
