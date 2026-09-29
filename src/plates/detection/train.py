"""Тренировка YOLO-детекции номеров.

Инкапсулирует типовой вызов Ultralytics: загрузка модели из yaml +
pretrained весов, запуск ``model.train(...)`` с конфигом.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # ultralytics — опциональная зависимость (extra "detection")
    from ultralytics import YOLO


def build_model(
    model_yaml: str | Path,
    pretrained_weights: str | Path | None = None,
) -> YOLO:
    """Загружает модель YOLO из yaml-спеки и, опционально, весов.

    Args:
        model_yaml: путь к ``yolo26m_ocr.yaml``-спеке.
        pretrained_weights: путь к ``.pt`` для инициализации.

    Returns:
        экземпляр ultralytics YOLO.
    """
    from ultralytics import YOLO  # ленивый импорт

    if pretrained_weights:
        model = YOLO(str(model_yaml)).load(str(pretrained_weights))
    else:
        model = YOLO(str(model_yaml))
    return model


def train(
    data_yaml: str | Path,
    cfg_yaml: str | Path | None = None,
    model_yaml: str | Path = "yolo26m_ocr.yaml",
    pretrained_weights: str | Path | None = "yolo26m.pt",
    device: int | str | None = None,
    project: str | None = None,
    name: str | None = None,
    **overrides,
):
    """Тренирует модель детекции.

    Args:
        data_yaml: путь к ``data.yaml`` датасета.
        cfg_yaml: глобальный конфиг тренировки (``default*.yaml``).
        model_yaml: спека модели.
        pretrained_weights: .pt для инициализации.
        device: CUDA-индекс (напр. 0) или строку ``"cpu"``.
        project, name: куда писать результат.
        **overrides: произвольные аргументы тренировки будут переопределять
            значения из конфига.

    Returns:
        Результат ``model.train(...)``.
    """
    model = build_model(model_yaml, pretrained_weights)

    train_kwargs = {}
    if cfg_yaml:
        train_kwargs["cfg"] = str(cfg_yaml)
    if device is not None:
        train_kwargs["device"] = device
    if project:
        train_kwargs["project"] = project
    if name:
        train_kwargs["name"] = name
    train_kwargs.update(overrides)

    return model.train(data=str(data_yaml), **train_kwargs)
