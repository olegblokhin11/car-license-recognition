#!/usr/bin/env python3
"""CLI: распознавание номеров OCR по предсказаниям детектора (или GT).

Читает CSV с колонками (image_name, x_1, y_1, x_2, y_2[, plate[, conf]]),
распознаёт каждый бокс и пишет результат CSV.

Пример:
    python scripts/run_ocr.py \\
        --predictions out/predictions.txt \\
        --images data/raw/images \\
        --out out/recognition_results.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from plates.ocr import build_recognizer, recognize_images


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=str, required=True, help="CSV с боксами")
    p.add_argument("--images", type=str, required=True, help="папка с изображениями")
    p.add_argument("--out", type=str, required=True, help="выходной CSV")
    p.add_argument("--ocr-model", type=str, default="cct-s-v2-global-model")
    p.add_argument("--expand", type=int, default=0)
    p.add_argument("--enhance", action="store_true", help="повышать контраст ROI")
    return p


def main() -> int:
    args = build_parser().parse_args()

    df = pd.read_csv(args.predictions)
    image_names = df["image_name"].tolist()
    boxes = df[["x_1", "y_1", "x_2", "y_2"]].values.tolist()

    recognizer = build_recognizer("fast_plate_ocr", args.ocr_model)
    preds = recognize_images(
        args.images,
        image_names,
        [tuple(b) for b in boxes],
        recognizer,
        expand_pixels=args.expand,
        enhance=bool(args.enhance),
    )

    out_df = pd.DataFrame(
        [
            {
                "image_name": p.image_name,
                "x_1": p.x_1,
                "y_1": p.y_1,
                "x_2": p.x_2,
                "y_2": p.y_2,
                "plate": p.plate,
            }
            for p in preds
        ]
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    recognized = (out_df["plate"].str.len() > 0).sum()
    print(f"Сохранено {len(out_df)} записей в {out_path}; распознано: {recognized}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
