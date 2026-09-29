"""Метрики оценки качества распознавания номеров.

CER (Character Error Rate) и CRR (Character Recognition Rate) —
стандартные метрики для OCR. CER считается как расстояние Левенштейна
между предсказанием и ground-truth, нормированное на длину ground-truth.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

_INVALID_CHARS = re.compile(r"[\W_]", re.UNICODE)


def clean_text(text) -> str:
    """Очищает сырой текст номера для сравнения.

    - приводит к верхнему регистру;
    - удаляет всё, кроме букв/цифр (латиница, кириллица, китайские символы).
    Handles ``None`` и ``NaN`` (pandas).
    """
    if text is None:
        return ""
    if str(text) != text:  # NaN и т.п.
        return ""
    return _INVALID_CHARS.sub("", str(text)).upper()


def levenshtein_distance(a: str, b: str) -> int:
    """Расстояние Левенштейна между двумя строками."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def character_error_rate(ground_truth: str, prediction: str) -> float:
    """CER для двух строк (терминальные строки очищаются внутри).

    Возвращает ``levenshtein(gt, pred) / len(gt)`` в диапазоне [0, 1].
    Если ``gt`` пустой, а ``pred`` нет — 1.0; если оба пустые — 0.0.
    """
    gt = clean_text(ground_truth)
    pred = clean_text(prediction)

    if not gt:
        return 0.0 if not pred else 1.0
    if not pred:
        return 1.0
    dist = levenshtein_distance(gt, pred)
    return min(dist / len(gt), 1.0)


@dataclass
class CRRResult:
    """Сводка метрик по набору пар ground-truth / prediction."""

    total: int = 0
    exact_matches: int = 0
    total_cer: float = 0.0
    cer: float = 0.0
    crr: float = 0.0
    errors: list[dict] = field(default_factory=list)

    @property
    def exact_match_rate(self) -> float:
        return 100.0 * self.exact_matches / self.total if self.total else 0.0


def summarize_crr(
    ground_truths: Sequence, predictions: Sequence, show_top_errors: int = 10
) -> CRRResult:
    """Подсчитывает CER/CRR/точность для списка пар.

    Args:
        ground_truths: сырые тексты эталонных номеров.
        predictions: сырые тексты предсказанных номеров.
        show_top_errors: сколько худших ошибок записать в ``errors``.

    Returns:
        ``CRRResult``.
    """
    result = CRRResult()

    for gt_raw, pred_raw in zip(ground_truths, predictions, strict=True):
        cerr = character_error_rate(gt_raw, pred_raw)
        result.total += 1
        result.total_cer += cerr
        if cerr == 0:
            result.exact_matches += 1
        elif len(result.errors) < show_top_errors:
            result.errors.append(
                {
                    "gt": clean_text(gt_raw),
                    "pred": clean_text(pred_raw),
                    "cer": round(cerr, 4),
                }
            )

    if result.total > 0:
        result.cer = result.total_cer / result.total
        result.crr = 1.0 - result.cer
    return result
