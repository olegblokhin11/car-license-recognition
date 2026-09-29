#!/usr/bin/env python3
"""Streamlit-демо: детекция и распознавание номеров на одном изображении.

Запуск:
    streamlit run app.py

Нужны локальные веса детектора вида ``assets/<эксперимент>/best.pt``: в
репозиторий они не входят, обучение описано в README («Быстрый старт»).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from plates.config import ProjectConfig, default_config
from plates.demo import (
    DemoResult,
    PlateResult,
    discover_weights,
    draw_detections,
    encode_png,
    plates_to_rows,
    run_pipeline,
    temporary_image,
)
from plates.detection.predict import YoloDetector
from plates.ocr.recognizer import PlateRecognizer, build_recognizer

UPLOAD_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
NO_SAMPLE = "— не выбран —"


@st.cache_resource(show_spinner=False)
def get_detector(weights: str) -> YoloDetector:
    """Загружает детектор один раз на процесс."""
    return YoloDetector(weights)


@st.cache_resource(show_spinner=False)
def get_recognizer(model_name: str) -> PlateRecognizer:
    """Загружает OCR-модель один раз на процесс."""
    return build_recognizer(model_name=model_name)


@st.cache_resource(show_spinner=False)
def cuda_available() -> bool:
    """Проверяет доступность GPU (torch импортируется лениво).

    Если torch не установлен, страница всё равно должна открыться — просто
    без подсказки про GPU; сам инференс тогда сообщит об ошибке при запуске.
    """
    try:
        import torch
    except ImportError:
        return False

    return torch.cuda.is_available()


@st.cache_data(show_spinner=False, max_entries=8)
def cached_run(
    image_bytes: bytes,
    suffix: str,
    weights: str,
    ocr_model: str,
    conf: float,
    imgsz: int,
    expand_pixels: int,
    enhance: bool,
    chinese_hack: bool,
) -> DemoResult:
    """Прогоняет пайплайн и кэширует результат по изображению и параметрам."""
    with temporary_image(image_bytes, suffix=suffix or ".jpg") as path:
        return run_pipeline(
            path,
            get_detector(weights),
            get_recognizer(ocr_model),
            conf=conf,
            imgsz=imgsz,
            expand_pixels=expand_pixels,
            enhance=enhance,
            use_chinese_hack=chinese_hack,
        )


def render_sidebar(config: ProjectConfig) -> dict | None:
    """Отрисовывает панель параметров и возвращает выбранные значения."""
    st.sidebar.header("Модель")

    weights_list = discover_weights(config.assets_dir)
    if not weights_list:
        st.sidebar.error(
            f"Не найдено весов `{config.assets_dir}/<эксперимент>/best.pt`.\n\n"
            "Обучите детектор (README, «Быстрый старт») или положите готовый "
            "`best.pt` в подкаталог `assets/`."
        )
        return None

    labels = {str(path): f"{path.parent.name}/best.pt" for path in weights_list}
    weights = st.sidebar.selectbox("Веса детектора", list(labels), format_func=labels.get)
    ocr_model = st.sidebar.text_input("OCR-модель", value=config.ocr_model)

    device = "GPU (CUDA)" if cuda_available() else "CPU"
    st.sidebar.caption(
        f"Инференс детектора: {device}. OCR — ONNX Runtime, провайдер выбирается сам."
    )

    st.sidebar.header("Параметры")
    conf = st.sidebar.slider("Порог уверенности детектора", 0.05, 0.95, config.detection_conf, 0.05)
    imgsz = st.sidebar.select_slider(
        "Разрешение инференса (imgsz)", [640, 960, 1280], config.detection_imgsz
    )
    expand_pixels = st.sidebar.slider("Расширение ROI, px", 0, 30, config.ocr_expand_pixels, 1)
    enhance = st.sidebar.checkbox("Повышать контраст ROI", value=config.ocr_enhance)
    chinese_hack = st.sidebar.checkbox("Постобработка границ «粤…港»", value=True)

    return {
        "weights": weights,
        "ocr_model": ocr_model,
        "conf": conf,
        "imgsz": imgsz,
        "expand_pixels": expand_pixels,
        "enhance": enhance,
        "chinese_hack": chinese_hack,
    }


def pick_image(config: ProjectConfig) -> tuple[bytes, str] | None:
    """Возвращает изображение: загруженное в браузере или пример из test/."""
    uploaded = st.file_uploader("Изображение с номером", type=UPLOAD_TYPES)
    if uploaded is not None:
        return uploaded.getvalue(), uploaded.name

    sample_dir = Path(config.images_dir) / "test"
    samples = sorted(sample_dir.glob("*.jpg")) if sample_dir.is_dir() else []
    if not samples:
        st.info("Загрузите изображение: локальных примеров `data/raw/images/test` нет.")
        return None

    st.caption(f"Либо пример из `{sample_dir}` ({len(samples)} изображений в наличии)")
    names = [NO_SAMPLE, *[path.name for path in samples[:100]]]
    choice = st.selectbox("Пример", names, key="sample")
    if choice == NO_SAMPLE:
        return None
    path = sample_dir / choice
    return path.read_bytes(), path.name


def decode_image(image_bytes: bytes) -> np.ndarray | None:
    """Декодирует байты в BGR-изображение (для отрисовки боксов)."""
    buffer = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def render_plate_card(image_bgr: np.ndarray, index: int, plate: PlateResult) -> None:
    """Показывает один найденный номер: вырезку, текст, уверенность и бокс."""
    roi = image_bgr[max(0, plate.y1) : plate.y2, max(0, plate.x1) : plate.x2]
    left, right = st.columns([1, 2])
    if roi.size:
        left.image(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), use_container_width=True)
    right.markdown(f"**№{index}: `{plate.plate or '—'}`**")
    right.write(
        f"уверенность детектора: `{plate.confidence:.3f}` · "
        f"бокс: `({plate.x1}, {plate.y1}, {plate.x2}, {plate.y2})` · "
        f"OCR до постобработки: `{plate.raw or '—'}`"
    )


def render_results(image_bgr: np.ndarray, file_name: str, result: DemoResult) -> None:
    """Показывает метрики, изображение с боксами, карточки номеров и таблицу."""
    annotated = draw_detections(image_bgr, result.plates)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Найдено номеров", len(result.plates))
    col2.metric("Средняя уверенность", f"{result.mean_confidence:.3f}")
    col3.metric("Детекция", f"{result.detection_ms:.0f} мс")
    col4.metric("OCR", f"{result.ocr_ms:.0f} мс")

    left, right = st.columns(2)
    left.image(
        image_bgr,
        caption=f"Исходное изображение {file_name}",
        channels="BGR",
        use_container_width=True,
    )
    right.image(
        annotated, caption="Детекция (номер и уверенность в подписи)", use_container_width=True
    )

    if not result.plates:
        st.warning(
            "Номеров не найдено. Попробуйте снизить порог уверенности детектора "
            "или увеличить разрешение инференса."
        )
        return

    stem = Path(file_name).stem
    rows = plates_to_rows(result.plates)
    download_left, download_right = st.columns(2)
    download_left.download_button(
        "Скачать изображение с боксами",
        data=encode_png(annotated),
        file_name=f"{stem}_boxes.png",
        mime="image/png",
        use_container_width=True,
    )
    download_right.download_button(
        "Скачать результаты (CSV)",
        data=pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{stem}_plates.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.subheader("Распознанные номера")
    for index, plate in enumerate(result.plates, start=1):
        render_plate_card(image_bgr, index, plate)

    with st.expander("Таблица результатов"):
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "«OCR (сырое)» — что вернула модель, «номер» — после постобработки. "
            "В подписях на изображении сам номер не печатается: cv2.putText "
            "не умеет китайские символы."
        )


def main() -> None:
    """Собирает страницу демо."""
    st.set_page_config(page_title="Распознавание номеров", page_icon="🚗", layout="wide")

    config = default_config()
    st.title("Распознавание автомобильных номеров")
    st.caption(
        "Гонконг / Китай / Макао · детекция YOLO26 → OCR fast_plate_ocr → постобработка. "
        "Веса берутся локально из `assets/`."
    )

    params = render_sidebar(config)
    if params is None:
        return

    picked = pick_image(config)
    if picked is None:
        st.info("Выберите изображение, чтобы запустить распознавание.")
        return

    image_bytes, file_name = picked
    image_bgr = decode_image(image_bytes)
    if image_bgr is None:
        st.error(f"Не удалось прочитать изображение {file_name}.")
        return

    with st.spinner("Детекция и распознавание…"):
        try:
            result = cached_run(
                image_bytes,
                Path(file_name).suffix,
                params["weights"],
                params["ocr_model"],
                params["conf"],
                params["imgsz"],
                params["expand_pixels"],
                params["enhance"],
                params["chinese_hack"],
            )
        except Exception as error:
            st.error(f"Ошибка инференса: {error}")
            return

    render_results(image_bgr, file_name, result)
    st.caption(
        "Тайминги — со времени прогона этой пары «изображение + параметры» (результат кэшируется)."
    )


if __name__ == "__main__":
    main()
