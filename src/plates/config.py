"""Централизованная конфигурация проекта (пути и значения по умолчанию).

В исходном эксперименте пути были «зашиты» строками вида ``'data/raw'``.
Здесь все пути вынесены в один класс и переопределяются переменными
окружения ``PLATES_*`` либо полями при создании экземпляра.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass
class ProjectConfig:
    """Конфигурация проекта: пути и ключевые гиперпараметры."""

    # --- пути ---
    data_root: str = field(
        default_factory=lambda: _env("PLATES_DATA_ROOT", "data/raw")
    )
    annotation_file: str = field(
        default_factory=lambda: _env(
            "PLATES_ANNOTATION", "data/raw/annotation/train_annot.txt"
        )
    )
    images_dir: str = field(
        default_factory=lambda: _env("PLATES_IMAGES", "data/raw/images")
    )
    yolo_root: str = field(
        default_factory=lambda: _env("PLATES_YOLO_ROOT", "datasets")
    )
    assets_dir: str = field(
        default_factory=lambda: _env("PLATES_ASSETS", "assets")
    )
    runs_dir: str = field(default_factory=lambda: _env("PLATES_RUNS", "runs"))
    output_dir: str = field(default_factory=lambda: _env("PLATES_OUTPUT", "out"))

    # --- параметры ---
    seed: int = 24
    split_val_fraction: float = 0.2
    class_names: list = field(
        default_factory=lambda: ["car_plate"]
    )
    detection_imgsz: int = 960
    detection_conf: float = 0.25
    ocr_model: str = "cct-s-v2-global-model"
    ocr_expand_pixels: int = 0
    ocr_enhance: bool = False

    # --- файлы моделей (в assets) ---
    model_yaml: str = "yolo26m_ocr.yaml"
    pretrained_weights: str = "yolo26m.pt"

    def project_root(self) -> Path:
        """Путь к корню проекта (2 уровня выше этого файла: src/plates)."""
        return Path(__file__).resolve().parents[2]

    def resolved(self) -> dict:
        """Возвращает dict с абсолютными путями (для удобного логирования)."""
        root = self.project_root()

        out: dict = {"project_root": str(root)}
        for f in fields(self):
            if f.name.endswith(("_dir", "_file", "_root")):
                val = getattr(self, f.name)
                p = Path(val)
                out[f.name] = str(p if p.is_absolute() else root / p)
        return out


def default_config() -> ProjectConfig:
    """Создаёт ``ProjectConfig`` со значениями по умолчанию."""
    return ProjectConfig()