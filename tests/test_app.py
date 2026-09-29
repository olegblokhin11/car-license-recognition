"""Smoke-тесты Streamlit-демо через ``streamlit.testing``.

Пропускаются, если Streamlit не установлен: демо — необязательная часть,
и в облегчённом окружении его может не быть.

Веса в тестах не нужны: детектор загружается только когда подано изображение,
поэтому вместо настоящего ``best.pt`` достаточно пустого файла-заглушки.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="для тестов демо нужен streamlit")

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def _make_assets(tmp_path: Path) -> Path:
    """Создаёт каталог весов с файлом-заглушкой ``<эксперимент>/best.pt``."""
    experiment = tmp_path / "exp"
    experiment.mkdir()
    (experiment / "best.pt").write_bytes(b"placeholder")
    return tmp_path


def test_app_renders_sidebar_and_waits_for_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Приложение собирается, показывает веса и параметры и ждёт изображение."""
    monkeypatch.setenv("PLATES_ASSETS", str(_make_assets(tmp_path)))

    app = AppTest.from_file(str(APP), default_timeout=120)
    app.run()

    assert not app.exception
    assert app.title[0].value == "Распознавание автомобильных номеров"
    assert app.sidebar.selectbox[0].options == ["exp/best.pt"]
    assert any("Порог уверенности" in slider.label for slider in app.sidebar.slider)
    assert any("Выберите изображение" in info.value for info in app.info)


def test_app_reports_missing_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Без весов приложение объясняет, что делать, вместо падения."""
    monkeypatch.setenv("PLATES_ASSETS", str(tmp_path))

    app = AppTest.from_file(str(APP), default_timeout=120)
    app.run()

    assert not app.exception
    assert any("Не найдено весов" in error.value for error in app.sidebar.error)
