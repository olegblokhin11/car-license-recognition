"""Импорт публичных модулей пакета в отдельных процессах.

Цикл импортов в Python зависит от порядка: например, ``import plates.ocr``
падал, а ``import plates.evaluation`` на том же коде проходил (первый импорт
запускает ``evaluation/__init__``, который тянет ``pipeline_eval``, а тот —
``ocr.pipeline``, ещё не доинициализированный). Обычные тесты такой баг
пропускают: в одном процессе порядок импортов задан коллекцией pytest и
фиксирован. Поэтому здесь каждый модуль импортируется в свежем интерпретаторе.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

# Точки входа, которые импортируют скрипты и тесты.
MODULES = [
    "plates",
    "plates.ocr",
    "plates.ocr.pipeline",
    "plates.evaluation",
    "plates.evaluation.pipeline_eval",
    "plates.data_prep",
    "plates.detection",
    "plates.autolabel",
    "plates.utils.yolo",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_standalone(module: str) -> None:
    """Модуль импортируется первым, до любых других модулей пакета."""
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SRC)},
        check=False,
    )
    assert result.returncode == 0, f"{module} не импортируется:\n{result.stderr}"
