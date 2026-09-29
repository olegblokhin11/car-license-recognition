"""Базовые утилиты пакета (io, файловые операции, воспроизводимость)."""

from __future__ import annotations

import os
import random
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Фиксирует seed для воспроизводимости numpy / stdlib / torch.

    torch импортируется лениво, чтобы модуль оставался импортируемым
    без установленного torch.
    """
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except Exception:
        pass


def set_cublas_workspace() -> None:
    """Включает детерминированную конфигурацию cuBLAS workspace.

    Это УВЕЛИЧИВАЕТ совместимость с ``torch.use_deterministic_algorithms``,
    но может замедлить обучение. На практике для YOLO оставляйте False.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def ensure_dir(path: str | Path) -> Path:
    """Создаёт директорию (включая родителей), если её нет."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def copy_file(src: str | Path, dst: str | Path) -> None:
    """Копирует один файл, создавая родительские директории цели."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise FileNotFoundError(f"Исходный файл не найден: {src}")
    shutil.copy2(src, dst)


def copy_many(files: Sequence[str | Path], dst_dir: str | Path) -> tuple[int, list[Path]]:
    """Копирует много файлов в одну директорию.

    Returns:
        (количество скопированных, список целевых путей).
    """
    dst_dir = ensure_dir(dst_dir)
    copied: list[Path] = []
    for f in files:
        f = Path(f)
        target = dst_dir / f.name
        shutil.copy2(f, target)
        copied.append(target)
    return len(copied), copied
