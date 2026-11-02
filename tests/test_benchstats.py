"""Estatística dos benchmarks (`tools/benchstats.py`)."""

from __future__ import annotations

import math

import pytest

from benchstats import Stats, compare, summarize, t975


def test_t975_table_and_limits() -> None:
    assert t975(1) == pytest.approx(12.706)
    assert t975(10) == pytest.approx(2.228)
    assert t975(1e9) == pytest.approx(1.96, abs=1e-3)
    assert 2.228 > t975(11) > 2.179  # interpolado entre 10 e 12


def test_summarize() -> None:
    s = summarize([1.0, 2.0, 3.0])
    assert (s.mean, s.n) == (2.0, 3)
    assert s.sd == pytest.approx(1.0)
    assert s.half_width == pytest.approx(4.303 / math.sqrt(3))
    assert math.isinf(summarize([5.0]).half_width)


@pytest.mark.parametrize(
    ("new", "status"),
    [
        (Stats(0.50, 0.01, 10), "ok"),
        (Stats(1.50, 0.01, 10), "mais lento"),
        (Stats(1.10, 0.30, 3), "inconclusivo"),
    ],
)
def test_compare(new: Stats, status: str) -> None:
    ratio, lo, hi, got = compare(new, Stats(1.0, 0.01, 30))
    assert lo <= ratio <= hi
    assert got == status
