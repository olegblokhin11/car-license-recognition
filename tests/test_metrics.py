"""Тесты метрик CER/CRR."""

import pytest

from plates.evaluation.metrics import (
    character_error_rate,
    clean_text,
    levenshtein_distance,
    summarize_crr,
)


def test_clean_text():
    assert clean_text("粤ZJA50港") == "粤ZJA50港"
    assert clean_text("abc 123") == "ABC123"
    assert clean_text(None) == ""
    assert clean_text("b_a-c") == "BAC"


def test_levenshtein():
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("abc", "abc") == 0
    assert levenshtein_distance("", "abc") == 3


def test_cer_perfect():
    assert character_error_rate("ABC123", "ABC123") == 0.0


def test_cer_one_substitution():
    assert character_error_rate("ABCD", "ABXD") == pytest.approx(0.25)


def test_cer_empty_gt():
    assert character_error_rate("", "XX") == 1.0
    assert character_error_rate("", "") == 0.0


def test_summarize():
    res = summarize_crr(
        ["ABC123", "XYZ789", "abc123"],
        ["ABC123", "XQZ789", "ABC123"],
    )
    assert res.total == 3
    # ABC123==ABC123 (0), XYZ789 vs XQZ789 (1/6), abc123 vs ABC123 (0 после upper)
    assert res.exact_matches == 2
    assert res.cer == pytest.approx((0 + 1 / 6 + 0) / 3)
