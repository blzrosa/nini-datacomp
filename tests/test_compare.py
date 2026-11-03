"""Comparação de uma coluna, par a par (`core/compare.py`, ``dc_compare_column``): conferida contra a própria expressão
do datacompolars 0.4.4 (o `NumericComparator` e o `eq_missing`) rodando no Polars instalado."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from itertools import pairwise
from typing import Any

import numpy as np
import polars as pl
import pytest

from nini_datacomp.core.arrow import import_column, import_frame
from nini_datacomp.core.compare import Tolerance, compare_column, compare_frames, compare_pairs, tolerance_for


def _datacompolars(left: pl.Series, right: pl.Series, abs_tol: float, rel_tol: float) -> list[bool]:
    """Diverge? -- a expressão do datacompolars 0.4.4 (comparators.NumericComparator + regra de nulos do engine)."""
    df = pl.DataFrame({"l": left, "r": right})
    a, b = pl.col("l"), pl.col("r")
    exact = a.eq_missing(b)
    if left.dtype.is_float():
        exact = exact | (a.is_nan() & b.is_nan())
    m = exact
    if abs_tol or rel_tol:
        m = exact | ((a - b).abs() <= abs_tol + rel_tol * b.abs()).fill_null(False)
    return df.select((~m.fill_null(False)).alias("d"))["d"].to_list()


def _run(left: pl.Series, right: pl.Series, abs_tol: float = 0.0, rel_tol: float = 0.0) -> list[bool]:
    idx = np.arange(left.len(), dtype=np.uint32)
    tol = Tolerance(abs_tol, rel_tol) if (abs_tol or rel_tol) else None
    differs, n = compare_column(import_column(left), import_column(right), idx, idx, tolerance=tol, threads=4)
    assert n == int(differs.sum())
    return [bool(x) for x in differs]


def _pair(dtype: Any, n: int, seed: int) -> tuple[pl.Series, pl.Series]:
    rng = np.random.default_rng(seed)
    if dtype in (pl.Float32, pl.Float64):
        pool = [0.0, -0.0, float("nan"), float("inf"), -float("inf"), 1.0, 1.25, 1.5, 2.0, -3.5, 1e30]
        lv = [pool[i] for i in rng.integers(0, len(pool), n)]
        rv = [pool[i] for i in rng.integers(0, len(pool), n)]
    else:
        np_type = NUMPY[dtype]
        info = np.iinfo(np_type)
        draw = np.int64 if info.min < 0 else np.uint64
        lv = [int(x) for x in rng.integers(info.min, info.max, n, endpoint=True, dtype=draw)]
        steps = rng.integers(-3, 4, n)
        near = [min(max(x + int(d), int(info.min)), int(info.max)) for x, d in zip(lv, steps, strict=True)]
        far = [int(x) for x in rng.integers(info.min, info.max, n, endpoint=True, dtype=draw)]
        pick = rng.integers(0, 3, n)
        rv = [lv[i] if pick[i] == 0 else near[i] if pick[i] == 1 else far[i] for i in range(n)]
    nl, nr = rng.random(n) < 0.05, rng.random(n) < 0.05
    left = pl.Series("x", [None if nl[i] else v for i, v in enumerate(lv)], dtype=dtype)
    right = pl.Series("x", [None if nr[i] else v for i, v in enumerate(rv)], dtype=dtype)
    return left, right


NUMPY: dict[Any, Any] = {
    pl.Int8: np.int8, pl.Int16: np.int16, pl.Int32: np.int32, pl.Int64: np.int64,
    pl.UInt8: np.uint8, pl.UInt16: np.uint16, pl.UInt32: np.uint32, pl.UInt64: np.uint64,
}  # fmt: skip
TYPES = [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64]


@pytest.mark.parametrize("tol", [(0.0, 0.0), (2.0, 0.0), (0.0, 0.01), (0.5, 0.125)])
@pytest.mark.parametrize("dtype", TYPES, ids=str)
def test_numeric_matches_datacompolars(dtype: Any, tol: tuple[float, float]) -> None:
    left, right = _pair(dtype, 3_000, seed=TYPES.index(dtype))
    assert _run(left, right, *tol) == _datacompolars(left, right, *tol)


def test_decimal_matches_datacompolars() -> None:
    lv = [Decimal("1.00"), Decimal("-2.50"), Decimal("1000000000.00"), None, Decimal("5.00"), Decimal("0.01")]
    rv = [Decimal("1.50"), Decimal("-2.00"), Decimal("1000000000.01"), None, None, Decimal("-0.01")]
    left = pl.Series("d", lv, dtype=pl.Decimal(20, 2))
    right = pl.Series("d", rv, dtype=pl.Decimal(20, 2))
    try:
        import_column(left)
    except Exception:  # Polars 1.30: o Decimal não sai pela interface C do Arrow (veja test_arrow)
        pytest.skip(f"Decimal não exportado pelo Polars {pl.__version__}")
    for tol in ((0.0, 0.0), (0.5, 0.0), (0.4, 0.0), (0.0, 0.5)):
        assert _run(left, right, *tol) == _datacompolars(left, right, *tol), tol


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (pl.Series(["a", None, "ação", "x" * 20, ""]), pl.Series(["a", None, "acao", "x" * 20, None])),
        (pl.Series([True, False, None, True]), pl.Series([True, True, None, None])),
        (pl.Series([1, 2, 3]).cast(pl.Date), pl.Series([1, 5, None]).cast(pl.Date)),
        (pl.Series([1, 2, 3]).cast(pl.Datetime("us")), pl.Series([1, 2, 4]).cast(pl.Datetime("us"))),
    ],
)
def test_exact_types(left: pl.Series, right: pl.Series) -> None:
    want = [not x for x in pl.DataFrame({"l": left, "r": right}).select(pl.col("l").eq_missing(pl.col("r")))["l"]]
    assert _run(left, right) == want


def test_tolerance_only_for_numeric() -> None:
    assert tolerance_for(pl.Float64(), 0.5, 0) == Tolerance(0.5, 0.0)
    assert tolerance_for(pl.Int32(), 0, 0.1) == Tolerance(0.0, 0.1)
    assert tolerance_for(pl.Float64(), 0, 0) is None  # sem tolerância: exata
    assert tolerance_for(pl.Date(), 1.0, 0) is None  # datas: sempre exatas (como no datacompolars)
    assert tolerance_for(pl.String(), 1.0, 0) is None
    with pytest.raises(ValueError, match="negativos"):
        tolerance_for(pl.Float64(), -1.0, 0)


def test_subset_and_accumulate() -> None:
    left, right = import_column(pl.Series([1, 2, 3, 4])), import_column(pl.Series([1, 9, 3, 8]))
    idx = np.arange(4, dtype=np.uint32)
    acc = np.zeros(4, np.uint8)
    _, n = compare_column(left, right, idx, idx, subset=np.array([0, 1], np.uint32), differs=acc)
    assert (acc.tolist(), n) == ([0, 1, 0, 0], 1)  # só os pares do subset
    _, n = compare_column(left, right, idx, idx, subset=np.array([3], np.uint32), differs=acc)
    assert (acc.tolist(), n) == ([0, 1, 0, 1], 1)  # acumula, nunca apaga
    with pytest.raises(ValueError, match="tipos diferentes"):
        compare_column(left, import_column(pl.Series(["a"] * 4)), idx, idx)


def test_compare_frames_matches_compare_column() -> None:
    """`compare_frames` (todas as colunas numa chamada, `dc_compare_rows`) dá o mesmo que `compare_column` coluna a
    coluna, em todos os tipos, com nulos, NaN e -0.0."""
    rng = np.random.default_rng(3)
    n = 70_000  # mais de um bloco
    pick = rng.integers(0, 4, n)

    def side(shift: int) -> pl.DataFrame:
        ints = (pick + shift * (rng.random(n) < 0.1)).tolist()
        floats = [[0.0, -0.0, float("nan"), 1.5][(v + shift) % 4] for v in pick.tolist()]
        return pl.DataFrame(
            {
                "i": [None if v == 3 else v for v in ints],
                "u": pl.Series(ints, dtype=pl.UInt16),
                "f": floats,
                "f32": pl.Series(floats, dtype=pl.Float32),
                "s": [None if v == 0 else "texto bem maior que doze " * v for v in ints],
                "b": [v % 2 == 0 for v in ints],
                "d": [date(2026, 1, 1 + v % 28) for v in ints],
            }
        )

    left, right = side(0), side(1)
    differs = compare_frames(left, right, threads=4)
    rows = np.arange(n, dtype=np.uint32)
    for j, c in enumerate(left.columns):
        want, count = compare_column(import_column(left[c]), import_column(right[c]), rows, rows)
        assert np.array_equal(differs[j], want), c
        assert int(differs[j].sum()) == count
    assert compare_frames(left.head(0), right.head(0)).shape == (left.width, 0)
    with pytest.raises(ValueError, match="mesmas colunas"):
        compare_frames(left, right.head(5))


def test_compare_pairs_reads_the_window_pieces() -> None:
    """`compare_pairs` (`dc_compare_pairs`) lê as colunas direto dos pedaços de memória da janela (os mesmos do hash
    de linha), com pedaços de tamanhos diferentes dos dois lados e os pares fora de ordem: mesmo resultado que
    `compare_column` nas colunas inteiras."""
    rng = np.random.default_rng(5)

    def frame(n: int) -> pl.DataFrame:
        v = rng.integers(0, 3, n)
        return pl.DataFrame(
            {"i": [None if x == 2 else int(x) for x in v], "s": ["um texto maior que doze " * int(x) for x in v]}
        )

    left, right = frame(9_000), frame(8_000)
    lcuts, rcuts = [0, 3_000, 3_500, 9_000], [0, 8_000]

    def pieces(df: pl.DataFrame, cuts: list[int], frames: bool) -> list[tuple[int, Any]]:
        out: list[tuple[int, Any]] = []
        for a, b in pairwise(cuts):
            part = df.slice(a, b - a)
            got = import_frame(part) if frames else None
            out.append((a, got if got is not None else [import_column(part[c]) for c in part.columns]))
        return out

    li = rng.integers(0, left.height, 70_000).astype(np.uint32)
    ri = rng.integers(0, right.height, 70_000).astype(np.uint32)
    subset = np.sort(rng.choice(li.size, 5_000, replace=False))
    for frames in (True, False):
        lp, rp = pieces(left, lcuts, frames), pieces(right, rcuts, not frames)
        whole = compare_pairs(lp, rp, li, ri, columns=2, threads=4)
        some = compare_pairs(lp, rp, li, ri, columns=2, subset=subset)
        for j, c in enumerate(left.columns):
            want, _ = compare_column(import_column(left[c]), import_column(right[c]), li, ri)
            assert np.array_equal(whole[j], want), c
            assert np.array_equal(some[j], want[subset]), c
    with pytest.raises(ValueError, match="fora da janela"):
        compare_pairs(pieces(left, lcuts, True), pieces(right, rcuts, True), li + 9_000, ri, columns=2)
