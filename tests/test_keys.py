"""Alinhamento das chaves (`core.keys`, kernel ``dc_align_keys``): pares e contagens contra uma referência em Python e
contra os gabaritos do datacompolars."""

from __future__ import annotations

import json
import math
import random
import struct
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any

import polars as pl
import pytest

import make_golden
from cases import CASES
from nini_datacomp import UnsupportedTypeError
from nini_datacomp.core.arrow import import_column, import_frame
from nini_datacomp.core.keys import Alignment, align_frames, align_keys, key_kind

STRATEGIES = ("auto", "merge", "hash")


def _norm(x: Any) -> Any:
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if x == 0.0:
            return 0.0  # -0.0 == 0.0
    return x


def _reference(left: pl.DataFrame, right: pl.DataFrame) -> tuple[set[tuple[int, int]], int, int]:
    """Pares (linha esquerda, linha direita), excedentes à esquerda e à direita."""
    lk = [tuple(_norm(x) for x in row) for row in left.iter_rows()]
    rk = [tuple(_norm(x) for x in row) for row in right.iter_rows()]
    pos = {k: j for j, k in enumerate(rk)}
    pairs = {(i, pos[k]) for i, k in enumerate(lk) if k in pos}
    return pairs, len(lk) - len(set(lk)), len(rk) - len(set(rk))


def _align(left: pl.DataFrame, right: pl.DataFrame, strategy: str = "auto") -> Alignment:
    return align_keys(
        [import_column(left[c]) for c in left.columns],
        [import_column(right[c]) for c in right.columns],
        strategy,  # type: ignore[arg-type]
    )


def check(left: pl.DataFrame, right: pl.DataFrame, strategies: Sequence[str] = STRATEGIES) -> Alignment:
    pairs, dl, dr = _reference(left, right)
    a = None
    for s in strategies:  # devolve o resultado da última (nos padrões, "hash")
        a = _align(left, right, s)
        assert (a.dup_left, a.dup_right) == (dl, dr), s
        if dl or dr:
            assert a.left.size == 0
            assert a.has_duplicates
            continue
        assert set(zip(a.left.tolist(), a.right.tolist(), strict=True)) == pairs, s
        assert a.common == len(pairs)
        assert a.only_left == left.height - len(pairs)
        assert a.only_right == right.height - len(pairs)
    assert a is not None
    return a


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("ordered", [True, False])
def test_int64_keys(seed: int, ordered: bool) -> None:
    rng = random.Random(seed)
    universe = rng.sample(range(-(10**12), 10**12), rng.randint(0, 400))
    lk = rng.sample(universe, rng.randint(0, len(universe)))
    rk = rng.sample(universe, rng.randint(0, len(universe)))
    if ordered:
        lk.sort()
        rk.sort()
    a = check(pl.DataFrame({"id": lk}, schema={"id": pl.Int64}), pl.DataFrame({"id": rk}, schema={"id": pl.Int64}))
    if ordered and lk and rk:
        assert _align(pl.DataFrame({"id": lk}), pl.DataFrame({"id": rk})).strategy == "merge"
    assert a.left.dtype.name == "uint32"


@pytest.mark.parametrize("dtype", [pl.Int8, pl.Int16, pl.Int32, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64])
def test_integer_widths(dtype: pl.DataType) -> None:
    lk = pl.Series("k", [0, 1, 2, 3, 5, 8, 13, 21, 100], dtype=dtype)
    rk = pl.Series("k", [1, 3, 4, 13, 22, 100, 101], dtype=dtype)
    a = check(lk.to_frame(), rk.to_frame())
    assert a.common == 4


def test_null_keys_match_each_other() -> None:
    a = check(pl.DataFrame({"id": [None, 1, 2, 3]}), pl.DataFrame({"id": [1, None, 2, 4]}))
    assert (a.common, a.only_left, a.only_right) == (3, 1, 1)


@pytest.mark.parametrize("ordered", [True, False])
def test_text_keys(ordered: bool) -> None:
    words = ["", "a", "ção", "doze bytes!!", "treze bytes!!", "um texto bem maior que doze bytes", "A-01", "A-02"]
    words += [f"chave-{i:05d}-com-texto-longo" for i in range(300)]
    rng = random.Random(3)
    lk, rk = rng.sample(words, 200), rng.sample(words, 250)
    if ordered:
        lk.sort()
        rk.sort()
    check(pl.DataFrame({"t": lk}), pl.DataFrame({"t": rk}))
    auto = _align(pl.DataFrame({"t": lk}), pl.DataFrame({"t": rk}))  # check devolve a última estratégia do laço (hash)
    assert auto.strategy == ("merge" if ordered else "hash")


def test_text_slice_keeps_offset() -> None:
    s = pl.Series("t", [f"valor número {i} com texto longo" for i in range(100)])
    check(s.slice(10, 50).to_frame(), s.slice(30, 50).to_frame())


@pytest.mark.parametrize("ordered", [True, False])
def test_composite_with_nulls(ordered: bool) -> None:
    rng = random.Random(5)
    universe = list(
        {(rng.choice([None, 1, 2, 3, 4]), rng.choice([None, "x", "y", "um texto comprido demais"])) for _ in range(300)}
    )
    lk, rk = rng.sample(universe, 15), rng.sample(universe, 12)
    left = pl.DataFrame({"a": [x[0] for x in lk], "b": [x[1] for x in lk]}, schema={"a": pl.Int32, "b": pl.String})
    right = pl.DataFrame({"a": [x[0] for x in rk], "b": [x[1] for x in rk]}, schema={"a": pl.Int32, "b": pl.String})
    if ordered:
        left, right = left.sort("a", "b"), right.sort("a", "b")
    check(left, right)


def test_duplicates_are_counted_and_no_pairs() -> None:
    a = check(pl.DataFrame({"id": [1, 2, 2, 3, 3, 3]}), pl.DataFrame({"id": [2, 3, 4]}))
    assert a.dup_left == 3
    check(pl.DataFrame({"id": [3, 1, 2, 3, 2, 3]}), pl.DataFrame({"id": [2, 3, 4]}))
    check(pl.DataFrame({"id": [2, 3, 4]}), pl.DataFrame({"id": [3, 1, 2, 3, 2, 3]}))


def test_float_keys() -> None:
    nan2 = struct.unpack("<d", struct.pack("<Q", 0xFFF8000000000001))[0]
    a = check(
        pl.DataFrame({"f": [0.0, float("nan"), 1.5, float("inf")]}),
        pl.DataFrame({"f": [-0.0, nan2, 2.5, float("inf")]}),
    )
    assert a.common == 3
    floats = pl.DataFrame({"f": [0.0, 1.0]})
    assert _align(floats, floats).strategy == "hash"  # reais não entram no merge, mesmo em ordem


def test_bool_keys_with_offset() -> None:
    s = pl.Series("b", [True, False, None, True, False, True])
    check(s.slice(1, 3).to_frame(), s.slice(2, 3).to_frame())


def test_temporal_keys() -> None:
    left = pl.DataFrame({"d": [date(2026, 1, d) for d in (1, 2, 3)], "t": [datetime(2026, 1, 1, h) for h in (1, 2, 3)]})
    right = pl.DataFrame(
        {"d": [date(2026, 1, d) for d in (2, 3, 4)], "t": [datetime(2026, 1, 1, h) for h in (2, 9, 4)]}
    )
    a = check(left, right)
    assert a.common == 1


def test_key_kind() -> None:
    assert key_kind(b"l") == key_kind(b"tdD") == key_kind(b"tsu:UTC") == 1
    assert key_kind(b"L") == 2
    assert key_kind(b"vu") == 3
    assert key_kind(b"b") == 4
    assert key_kind(b"d:10,2") == 5
    assert key_kind(b"g") == 6


def test_errors() -> None:
    with pytest.raises(ValueError, match="tipos diferentes"):
        _align(pl.DataFrame({"id": [1]}), pl.DataFrame({"id": ["1"]}))
    with pytest.raises(ValueError, match="mesmas colunas"):
        align_keys([], [])
    with pytest.raises(ValueError, match="estratégia"):
        _align(pl.DataFrame({"id": [1]}), pl.DataFrame({"id": [1]}), "outra")


def test_categorical_is_rejected() -> None:
    with pytest.raises(UnsupportedTypeError, match="dicionário"):
        import_column(pl.Series("c", ["a", "b"], dtype=pl.Categorical))


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_golden_row_counts(case: Any) -> None:
    """As contagens de linhas de cada caso de borda batem com o gabarito do datacompolars."""
    golden = json.loads((make_golden.GOLDEN / f"{case.name}.json").read_text(encoding="utf-8"))
    left, right = case.build()
    keys = case.settings["join_columns"]
    for strategy in STRATEGIES:
        a = _align(left.select(keys), right.select(keys), strategy)
        if golden["aborted"]:
            assert a.has_duplicates
            continue
        rows = golden["rows"]
        assert (a.common, a.only_left, a.only_right) == (rows["common"], rows["left_only"], rows["right_only"])


@pytest.mark.parametrize("strategy", ["auto", "merge", "hash"])
def test_align_frames_matches_align_keys(strategy: Any) -> None:
    """`align_frames` (a chave de cada lado numa exportação só) dá o mesmo que `align_keys` (coluna a coluna)."""
    rng = random.Random(9)
    left = pl.DataFrame({"a": [rng.randint(0, 50) for _ in range(500)], "b": [rng.choice("xyz") for _ in range(500)]})
    right = pl.DataFrame({"a": [rng.randint(0, 50) for _ in range(400)], "b": [rng.choice("xyz") for _ in range(400)]})
    left, right = left.unique(maintain_order=True), right.unique(maintain_order=True)
    lf, rf = import_frame(left), import_frame(right)
    assert lf is not None
    assert rf is not None
    got = align_frames(lf, rf, strategy)
    want = align_keys(
        [import_column(left[c]) for c in left.columns], [import_column(right[c]) for c in right.columns], strategy
    )
    assert (got.common, got.only_left, got.only_right) == (want.common, want.only_left, want.only_right)
    assert got.strategy == want.strategy
    pairs = sorted(zip(got.left.tolist(), got.right.tolist(), strict=True))
    assert pairs == sorted(zip(want.left.tolist(), want.right.tolist(), strict=True))
    bad = import_frame(right.with_columns(pl.col("a").cast(pl.Int32)))
    assert bad is not None
    with pytest.raises(ValueError, match="tipos diferentes"):
        align_frames(lf, bad)
