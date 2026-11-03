"""Fase 6c: comparadores próprios por coluna (``comparators={...}``): funções do Polars (com PerformanceWarning),
comparadores do datacompolars e o dc.expr, que roda no C++ e tem de dar o mesmo resultado que a expressão
equivalente do Polars."""

from __future__ import annotations

import datetime
import math
import random
import warnings
from typing import Any

import polars as pl
import pytest

import nini_datacomp as dc
from nini_datacomp.expr import left, lit, right

L = pl.DataFrame(
    {
        "id": list(range(8)),
        "v": [1.0, 2.0, None, 4.0, math.nan, 0.0, 100.0, None],
        "n": [1, 2, 3, 4, 5, 6, 7, None],
        "s": ["  Ana ", "JOSÉ", "Straße", None, "ΟΔΟΣ", "x y", "", "z"],
        "b": [True, False, None, True, False, True, False, True],
    }
)
R = pl.DataFrame(
    {
        "id": list(range(8)),
        "v": [1.004, 2.5, None, None, math.nan, -0.0, 101.0, None],
        "n": [1, 3, 3, 4, 6, 6, 9, None],
        "s": ["ana", "josé", "STRASSE", None, "οδος", "xy", None, "Z"],
        "b": [True, True, None, True, False, True, False, False],
    }
)


def _quiet(**kwargs: Any) -> dc.ComparisonResult:
    with warnings.catch_warnings():
        warnings.simplefilter("error", dc.PerformanceWarning)  # o dc.expr não pode avisar
        return dc.compare(L, R, "id", **kwargs)


def _mismatched_ids(res: dc.ComparisonResult, column: str) -> list[int]:
    return res.columns[column].samples["id"].to_list()


def test_expr_numeric() -> None:
    res = _quiet(comparators={"v": (left - right).abs() <= 0.01, "n": (left - right).abs() <= 1})
    # v: 1 (0,5), 3 (só um nulo), 4 (NaN - NaN é NaN, e NaN <= 0,01 é falso, como no Polars), 6 (diferença 1);
    # -0.0 x 0.0 dá diferença 0
    assert _mismatched_ids(res, "v") == [1, 3, 4, 6]
    assert res.columns["n"].mismatches == 1  # só o 6 (7 x 9)
    # tolerância relativa, com divisão (real) e inteiro virando real
    rel = _quiet(comparators={"n": (left - right).abs() / right.abs() <= 0.25})
    assert _mismatched_ids(rel, "n") == [1]  # 2 x 3: 0,33; 5 x 6: 0,17; 7 x 9: 0,22


def test_expr_text() -> None:
    res = _quiet(comparators={"s": left.strip().lower() == right.strip().lower()})
    # Straße.lower() = straße != strasse; "x y" != "xy"; "" x nulo; z == Z; ΟΔΟΣ -> οδος (sigma final)
    assert _mismatched_ids(res, "s") == [2, 5, 6]
    up = _quiet(comparators={"s": left.upper().remove_whitespace() == right.upper().remove_whitespace()})
    assert _mismatched_ids(up, "s") == [6]  # STRASSE == STRASSE; XY == XY; só o "" x nulo
    length = _quiet(comparators={"s": left.strip().len() == right.len()})
    assert 0 not in _mismatched_ids(length, "s")


def test_expr_bool_and_logic() -> None:
    res = _quiet(comparators={"b": (left == right) | (left & ~right)})
    assert _mismatched_ids(res, "b") == [1]  # False x True; o 7 (True x False) passa pelo left & ~right
    const = _quiet(comparators={"b": (left == lit(True)) == (right == True)})  # noqa: E712
    assert const.columns["b"].mismatches == 2


@pytest.mark.parametrize(
    ("expr", "polars"),
    [
        ((left - right).abs() <= 0.5, lambda a, b: (a - b).abs() <= 0.5),
        (left * 2 >= right + 1, lambda a, b: a * 2 >= b + 1),
        (left / right < 1.5, lambda a, b: a / b < 1.5),
        (-left != right, lambda a, b: -a != b),
        ((left > right) & (left < 100) | (right == 0), lambda a, b: ((a > b) & (a < 100)) | (b == 0)),
    ],
)
@pytest.mark.parametrize("column", ["f", "i"])
def test_expr_matches_polars(expr: dc.expr.Expr, polars: Any, column: str) -> None:
    """O dc.expr dá o mesmo que a função equivalente no Polars, em valores sortidos (com nulos e, nos reais, NaN)."""
    rnd = random.Random(11)
    n = 3000

    def values(seed_shift: float) -> list[Any]:
        out: list[Any] = []
        for _ in range(n):
            r = rnd.random()
            if r < 0.05:
                out.append(None)
            elif column == "f" and r < 0.08:
                out.append(math.nan)
            elif column == "f":
                out.append(round(rnd.uniform(-5, 5) + seed_shift, 2))
            else:
                out.append(rnd.randint(-5, 5))
        return out

    a = pl.DataFrame({"id": range(n), column: values(0)})
    b = pl.DataFrame({"id": range(n), column: values(0.3)})
    native = dc.compare(a, b, "id", comparators={column: expr}, samples=0, window_rows=700)
    with pytest.warns(dc.PerformanceWarning):
        slow = dc.compare(a, b, "id", comparators={column: polars}, samples=0, window_rows=700)
    assert native.columns[column].mismatches == slow.columns[column].mismatches
    assert native.rows.mismatched == slow.rows.mismatched


def test_polars_function_and_datacompolars_comparator() -> None:
    class Upper:  # como um comparador do datacompolars: nomes de coluna e kwargs
        def compare(self, col_l: str, col_r: str, **kwargs: Any) -> pl.Expr:
            assert set(kwargs) == {"abs_tol", "rel_tol", "ignore_spaces", "ignore_case", "is_float"}
            return pl.col(col_l).str.to_uppercase() == pl.col(col_r).str.to_uppercase()

    with pytest.warns(dc.PerformanceWarning, match="s"):
        res = dc.compare(L, R, "id", comparators={"s": Upper()})
    assert _mismatched_ids(res, "s") == [0, 5, 6]
    with pytest.warns(dc.PerformanceWarning):
        fn = dc.compare(L, R, "id", comparators={"s": lambda a, b: a.str.to_uppercase() == b.str.to_uppercase()})
    assert _mismatched_ids(fn, "s") == [0, 5, 6]


def test_comparator_on_lists_with_polars() -> None:
    a = pl.DataFrame({"id": [1, 2, 3], "l": [[1, 2], [3], None]})
    b = pl.DataFrame({"id": [1, 2, 3], "l": [[2, 1], [4], None]})
    with pytest.warns(dc.PerformanceWarning):
        res = dc.compare(a, b, "id", comparators={"l": lambda x, y: x.list.sort() == y.list.sort()})
    assert _mismatched_ids(res, "l") == [2]


def test_comparator_rules() -> None:
    # na chave: erro fatal no resultado
    res = _quiet(comparators={"id": left == right})
    assert res.aborted
    assert "join key" in res.errors[0]
    # coluna que não é comparada: aviso
    res = _quiet(comparators={"nada": left == right})
    assert res.warnings == ("comparators names columns that are not compared: ['nada']",)
    # nomes em minúsculas, como as outras opções
    assert _quiet(comparators={"N": (left - right).abs() <= 10}).columns["n"].mismatches == 0
    with pytest.raises(ValueError, match="não são comparadores"):
        dc.compare(L, R, "id", comparators={"v": 3})
    # o comparador decide sobre a coluna: a tolerância e o ignore_case não se aplicam a ela
    res = _quiet(comparators={"s": left == right}, ignore_case=True, abs_tol=1000.0)
    assert res.columns["s"].mismatches == 7  # todos menos o 3 (nulo/nulo)
    assert res.columns["v"].mismatches == 1  # o 3 (só um nulo); a tolerância vale nas outras colunas


def test_expr_type_errors() -> None:
    with pytest.raises(TypeError, match="booleano"):
        _quiet(comparators={"v": left - right})
    with pytest.raises(TypeError, match="não é texto"):
        _quiet(comparators={"v": left.lower() == right.lower()})
    with pytest.raises(TypeError, match="comparação de texto com inteiro"):
        _quiet(comparators={"s": left == 1})
    with pytest.raises(TypeError, match="and/or/not"):
        bool(left == right)
    dates = pl.DataFrame({"id": [1], "d": [datetime.date(2024, 1, 1)]})
    with pytest.raises(TypeError, match="não lê colunas Date"):
        dc.compare(dates, dates, "id", comparators={"d": left == right})
    assert repr((left - right).abs() <= 0.01) == "((left - right).abs() <= 0.01)"
