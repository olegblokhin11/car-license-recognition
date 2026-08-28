"""Авторазметка недостающих изображений через GroundingDINO (detr).

Использует zero-shot детектор ``IDEA-Research/grounding-dino-tiny``
с текстовым промптом ``["license plate"]``, чтобы доразметить
изображения, у которых нет ручной аннотации, и расширить датасет.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..data_prep.build_yolo import image_size
from ..utils.io import ensure_dir
from ..utils.yolo import format_label_line, pascal_to_yolo


@dataclass
class GroundingLabelStats:
    """Статистика авторазметки."""

    processed_images: int
    found_plates: int
    errors: list[str]
    threshold: float


class GroundingLabeler:
    """Размечает изображения нулевым шотом через GroundingDINO."""

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        text_prompt: str = "license plate",
        device: str = "auto",
        conf_threshold: float = 0.35,
        text_threshold: float = 0.3,
    ):
        self.model_id = model_id
        self.text_prompt = text_prompt
        self.conf_threshold = conf_threshold
        self.text_threshold = text_threshold
        self._device = device
        self._loaded = False

    def _lazy_load(self) -> None:
        if self._loaded:
            return
        import torch
        from accelerate import Accelerator
        from PIL import Image
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        dev = Accelerator().device if self._device == "auto" else self._device
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        ).to(dev)
        self._Image = Image
        self._device = dev
        self._loaded = True

    def annotate_image(self, image_path: str | Path) -> list[list[float]]:
        """Возвращает список боксов ``[x1, y1, x2, y2]`` для одного изображения."""
        self._lazy_load()
        image = self._Image.open(str(image_path)).convert("RGB")
        inputs = self._processor(
            images=image,
            text=[[self.text_prompt]],
            return_tensors="pt",
        ).to(self._device)

        import torch

        with torch.no_grad():
            outputs = self._model(**inputs)

        result = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.conf_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        boxes: list[list[float]] = []
        for box, score in zip(result["boxes"], result["scores"]):
            if float(score) >= self.conf_threshold:
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                boxes.append([x1, y1, x2, y2])
        return boxes


def label_images(
    new_images_dir: str | Path,
    yolo_annotated_dataset: str | Path,
    output_dataset: str | Path,
    labeler: GroundingLabeler | None = None,
    conf_threshold: float = 0.35,
    text_threshold: float = 0.3,
) -> GroundingLabelStats:
    """Доразмечен копию существующего YOLO-датасета и добавляет новые изображения.

    Args:
        new_images_dir: папка с изображениями без ручной разметки.
        yolo_annotated_dataset: существующий (уже размеченный) датасет.
        output_dataset: куда писать объединённый датасет.
        labeler: инстанс GroundingLabeler; если нет — создаётся.
        conf_threshold: порог уверенности боксов от DINO.

    Returns:
        ``GroundingLabelStats``.
    """
    src = Path(yolo_annotated_dataset)
    dst = Path(output_dataset)
    ensure_dir(dst / "images" / "train")
    ensure_dir(dst / "images" / "val")
    ensure_dir(dst / "labels" / "train")
    ensure_dir(dst / "labels" / "val")

    # 1) Копируем уже размеченные изображения/аннотации.
    import shutil

    for split in ("train", "val"):
        for sub in ("images", "labels"):
            src_dir = src / sub / split
            dst_dir = dst / sub / split
            if src_dir.exists():
                for f in src_dir.iterdir():
                    if f.is_file():
                        shutil.copy2(f, dst_dir / f.name)

    # 2) Размечаем новые.
    labeler = labeler or GroundingLabeler(
        conf_threshold=conf_threshold, text_threshold=text_threshold
    )
    new_dir = Path(new_images_dir)
    new_imgs = sorted(new_dir.glob("*.jpg")) + sorted(new_dir.glob("*.png"))

    errors: list[str] = []
    found_plates = 0
    for img_path in new_imgs:
        try:
            boxes = labeler.annotate_image(img_path)
        except Exception as exc:  # noqa: BLE001 - продолжаем дальше по папке
            errors.append(f"{img_path.name}: {exc}")
            continue

        if boxes:
            found_plates += len(boxes)
            img_w, img_h = image_size(img_path)
            yolo_boxes = pascal_to_yolo(boxes, img_w, img_h)
            label_path = dst / "labels" / "train" / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                for yb in yolo_boxes:
                    f.write(format_label_line(yb))
            shutil.copy2(img_path, dst / "images" / "train" / img_path.name)

    # 3) data.yaml.
    content = (
        f"# YOLO dataset configuration\n"
        f"path: {dst.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"nc: 1\nnames: ['car_plate']\n"
    )
    (dst / "data.yaml").write_text(content)

    return GroundingLabelStats(
        processed_images=len(new_imgs),
        found_plates=found_plates,
        errors=errors,
        threshold=conf_threshold,
    )