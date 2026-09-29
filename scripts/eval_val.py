#!/usr/bin/env python3
"""CLI: оценка трёх фронтов на выборке датасета — детекция, OCR, пайплайн.

Считает:

1. **Детекция** — mAP50 / mAP50-95 / precision / recall через ``ultralytics``
   на разделе собранного датасета (по умолчанию ``val``);
2. **OCR только** — по эталонным боксам (потолок качества распознавания);
3. **Полный пайплайн** — детекция + OCR, две версии CRR:
   ``end-to-end`` (пропуск детекции = ошибка) и ``conditional``
   (только сматченные боксы).

Фронты 1 и 2 можно отключить (``--skip-detection``, ``--skip-ocr-only``) —
например, чтобы посчитать только CRR пайплайна:

    python scripts/eval_val.py --skip-detection --skip-ocr-only --split val

Пример полного прогона:
    python scripts/eval_val.py --weights assets/base_ext_v2/best.pt \\
        --dataset dataset_for_yolo_extended_v2 --split val \\
        --json out/eval_val.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plates.data_prep import filter_annotations_by_split, load_annotations
from plates.evaluation import (
    build_eval_images,
    crr_to_dict,
    evaluate_ocr_only,
    evaluate_pipeline,
    pipeline_metrics_to_dict,
)
from plates.ocr import build_recognizer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--weights", type=str, default="assets/base_ext_v2/best.pt")
    p.add_argument("--annot", type=str, default="data/raw/annotation/train_annot.txt")
    p.add_argument("--images", type=str, default="data/raw/images")
    p.add_argument(
        "--dataset", type=str, default="dataset_for_yolo_extended_v2", help="корень YOLO-датасета"
    )
    p.add_argument("--split", type=str, default="val", choices=["val", "train"])
    p.add_argument(
        "--limit", type=int, default=None, help="ограничить число изображений (по умолчанию все)"
    )
    # фронт 1 — детекция
    p.add_argument("--det-imgsz", type=int, default=640, help="разрешение для mAP")
    p.add_argument("--det-batch", type=int, default=16)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--skip-detection", action="store_true", help="не считать mAP")
    p.add_argument("--project", type=str, default="runs/detect", help="куда писать артефакты val")
    p.add_argument("--name", type=str, default="eval_val")
    # фронты 2 и 3 — OCR и пайплайн
    p.add_argument(
        "--skip-ocr-only",
        action="store_true",
        help="не считать OCR по эталонным боксам",
    )
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=960, help="разрешение инференса детектора")
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--ocr-model", type=str, default="cct-s-v2-global-model")
    p.add_argument(
        "--json",
        dest="json_path",
        type=str,
        default=None,
        help="куда сохранить метрики",
    )
    return p


def run_detection_val(args: argparse.Namespace) -> dict | None:
    """Считает mAP/P/R на разделе датасета через ultralytics."""
    data_yaml = Path(args.dataset) / "data.yaml"
    if not data_yaml.exists():
        print(f"[детекция] пропущено: не найден {data_yaml}")
        return None
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[детекция] пропущено: не установлен ultralytics (extra 'detection')")
        return None

    model = YOLO(args.weights)
    result = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.det_imgsz,
        batch=args.det_batch,
        device=args.device,
        plots=False,
        verbose=False,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )
    return {
        "imgsz": args.det_imgsz,
        "mAP50": round(float(result.box.map50), 4),
        "mAP50-95": round(float(result.box.map), 4),
        "precision": round(float(result.box.mp), 4),
        "recall": round(float(result.box.mr), 4),
    }


def main() -> int:
    args = build_parser().parse_args()

    df = load_annotations(args.annot)
    split_df = filter_annotations_by_split(df, args.dataset, args.split)
    images = build_eval_images(split_df, args.images)
    if args.limit:
        images = images[: args.limit]
    if not images:
        print("Выборка пуста — проверьте --dataset/--split/--annot/--images.")
        return 1

    n_gt = sum(len(item.plates) for item in images)
    print(f"Выборка: {args.split} ({args.dataset}) — {len(images)} изображений, {n_gt} номеров\n")

    detection = None if args.skip_detection else run_detection_val(args)

    recognizer = build_recognizer("fast_plate_ocr", args.ocr_model)
    ocr_only = None if args.skip_ocr_only else evaluate_ocr_only(images, recognizer)

    from plates.detection import YoloDetector

    detector = YoloDetector(args.weights)
    pipeline = evaluate_pipeline(
        images, detector, recognizer, conf=args.conf, imgsz=args.imgsz, iou_thr=args.iou_thr
    )

    print("\n=== Итоги ===")
    if detection:
        print(
            f"Детекция (imgsz {detection['imgsz']}): "
            f"mAP50 {detection['mAP50']} | mAP50-95 {detection['mAP50-95']} | "
            f"P {detection['precision']} | R {detection['recall']}"
        )
    if ocr_only:
        print(
            f"OCR по эталонным боксам: CRR {ocr_only.crr:.4f} | CER {ocr_only.cer:.4f} | "
            f"точных {ocr_only.exact_matches}/{ocr_only.total} "
            f"({ocr_only.exact_match_rate:.1f}%)"
        )
    print(
        f"Пайплайн end-to-end: CRR {pipeline.end_to_end.crr:.4f} | CER "
        f"{pipeline.end_to_end.cer:.4f} | точных "
        f"{pipeline.end_to_end.exact_matches}/{pipeline.end_to_end.total} "
        f"({pipeline.end_to_end.exact_match_rate:.1f}%)"
    )
    print(
        f"Пайплайн conditional: CRR {pipeline.conditional.crr:.4f} | CER "
        f"{pipeline.conditional.cer:.4f} | точных "
        f"{pipeline.conditional.exact_matches}/{pipeline.conditional.total} "
        f"({pipeline.conditional.exact_match_rate:.1f}%)"
    )
    print(
        f"Детекция @IoU {args.iou_thr}, conf {args.conf}: сматчено "
        f"{pipeline.n_matched} | пропущено {pipeline.n_missed} | лишних "
        f"{pipeline.n_false_positives} | recall {pipeline.detection_recall:.2%}"
    )

    if args.json_path:
        out = Path(args.json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "weights": args.weights,
                "dataset": args.dataset,
                "split": args.split,
                "annot": args.annot,
                "images": args.images,
                "conf": args.conf,
                "pipeline_imgsz": args.imgsz,
                "iou_thr": args.iou_thr,
                "ocr_model": args.ocr_model,
                "n_images": len(images),
                "n_plates": n_gt,
                "skipped": [
                    name
                    for name, skipped in (
                        ("detection", args.skip_detection),
                        ("ocr_only", args.skip_ocr_only),
                    )
                    if skipped
                ],
            },
            "detection": detection,
            "ocr_only": crr_to_dict(ocr_only) if ocr_only else None,
            "pipeline": pipeline_metrics_to_dict(pipeline),
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nМетрики сохранены: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
