#!/usr/bin/env python3
"""Быстрая оценка полного пайплайна (детекция + OCR) на размеченной выборке.

Использует ground-truth боксы из аннотации, затем:
1. прогоняет детектор по изображениям;
2. для задетектированных боксов сопоставляет с GT по IoU;
3. считает CRR/CER по распознанным номерам.

Пример:
    python scripts/eval_pipeline.py --weights assets/base_ext_v2/best.pt \\
        --annot data/raw/annotation/train_annot.txt \\
        --images data/raw/images --n 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import pandas as pd

from plates.data_prep import load_annotations
from plates.detection import YoloDetector
from plates.evaluation.metrics import summarize_crr
from plates.ocr import build_recognizer, postprocess_plate


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=str, default="assets/base_ext_v2/best.pt")
    p.add_argument("--annot", type=str, default="data/raw/annotation/train_annot.txt")
    p.add_argument("--images", type=str, default="data/raw/images")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=960)
    return p


def main() -> int:
    args = build_parser().parse_args()

    df = load_annotations(args.annot)
    images_dir = Path(args.images)
    detector = YoloDetector(args.weights)
    recognizer = build_recognizer("fast_plate_ocr")

    # уникальные изображения (до n)
    image_names = sorted(df["image_name"].unique())[: args.n]

    rows = []
    for image_name in image_names:
        path = images_dir / image_name
        if not path.exists():
            continue
        # GT боксы
        gt_rows = df[df["image_name"] == image_name]
        gt_boxes = [
            (r.x_1, r.y_1, r.x_2, r.y_2)
            for _, r in gt_rows.iterrows()
        ]
        gt_texts = [r.plate for _, r in gt_rows.iterrows()]

        # детекция
        dets = detector.predict(path, conf=args.conf, imgsz=args.imgsz)
        img = cv2.imread(str(path))

        matched_gt = set()
        for d in dets:
            # сопоставляем с ближайшим GT
            best_iou, best_idx = 0.0, -1
            for i, gb in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                v = iou((d.x1, d.y1, d.x2, d.y2), gb)
                if v > best_iou:
                    best_iou, best_idx = v, i
            if best_iou >= 0.5 and best_idx >= 0:
                matched_gt.add(best_idx)
                roi = img[d.y1:d.y2, d.x1:d.x2]
                if roi.size == 0:
                    pred_text = ""
                else:
                    raw = recognizer.recognize(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                    pred_text = postprocess_plate(raw)
                rows.append(
                    {"image_name": image_name, "gt": gt_texts[best_idx], "pred": pred_text}
                )
    res_df = pd.DataFrame(rows)
    if len(res_df) == 0:
        print("Нет совпадений детекции с GT.")
        return 1

    result = summarize_crr(res_df["gt"], res_df["pred"])
    print(f"Оценено изображений: {len(image_names)}")
    print(f"Совпадений детекция-GT (IoU>=0.5): {len(res_df)}")
    print(f"Точных совпадений: {result.exact_matches}/{result.total}"
          f" ({result.exact_match_rate:.1f}%)")
    print(f"CER: {result.cer:.4f} | CRR: {result.crr:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())