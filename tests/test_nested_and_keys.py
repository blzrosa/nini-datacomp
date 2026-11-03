"""Fase 6b: colunas List, Array e Struct (comparadas pelo valor inteiro; ``unordered_lists`` ignora a ordem) e
``only_common_keys`` (o mesmo resultado, lendo por inteiro só as linhas na faixa de chaves que os dois lados têm)."""

from __future__ import annotations

import random
from typing import Any

import polars as pl
import pytest

import nini_datacomp as dc


def _summary(res: dc.ComparisonResult) -> tuple[Any, ...]:
    r = res.rows
    cols = tuple(sorted((c, res.columns[c].mismatches) for c in res.columns))
    return (r.left_total, r.right_total, r.common, r.left_only, r.right_only, r.mismatched, cols, res.errors)


# ----------------------------------------------------------------------------- List, Array, Struct
LEFT = pl.DataFrame(
    {
        "id": [1, 2, 3, 4, 5, 6],
        "l": [[1, 2], [2, 1], None, [], [3, None], [1, 1, 2]],
        "s": [
            {"a": 1, "b": "x"}, {"a": 2, "b": "y"}, None, {"a": None, "b": "z"}, {"a": 5, "b": None},
            {"a": 6, "b": "w"},
        ],
    }
)  # fmt: skip


def test_lists_keep_order_by_default() -> None:
    right = LEFT.with_columns(l=pl.Series([[2, 1], [2, 1], None, [], [None, 3], [1, 2, 2]]))
    res = dc.compare(LEFT, right, "id")
    assert res.columns["l"].mismatches == 3  # ids 1, 5 e 6: a ordem (e a multiplicidade em 6) conta
    assert res.columns["s"].mismatches == 0
    samples = res.columns["l"].samples
    assert samples["id"].to_list() == [1, 5, 6]
    assert samples["l (left)"].to_list() == [[1, 2], [3, None], [1, 1, 2]]  # o valor de verdade, não o hash
    unordered = dc.compare(LEFT, right, "id", unordered_lists=["L"])  # nomes em minúsculas, como as outras opções
    assert unordered.columns["l"].mismatches == 1  # só o 6: [1, 1, 2] e [1, 2, 2] continuam diferentes
    assert not unordered.warnings


def test_nulls_and_empty_lists() -> None:
    right = LEFT.with_columns(l=pl.Series([[1, 2], [2, 1], [], None, [3, None], [1, 1, 2]]))
    res = dc.compare(LEFT, right, "id")
    assert res.columns["l"].mismatches == 2  # nulo != [] (ids 3 e 4); nulo == nulo
    assert dc.compare(LEFT, LEFT, "id").is_match


def test_structs() -> None:
    right = LEFT.with_columns(
        s=pl.Series(
            [
                {"a": 1, "b": "x"}, {"a": 2, "b": "Y"}, None, {"a": None, "b": "z"}, {"a": None, "b": None},
                {"a": 6, "b": "w"},
            ]
        )
    )  # fmt: skip
    res = dc.compare(LEFT, right, "id", ignore_case=True, abs_tol=10.0)
    # o ignore_case e a tolerância não entram nos campos de um Struct: comparado pelo valor exato
    assert res.columns["s"].mismatches == 2
    assert res.columns["s"].samples["id"].to_list() == [2, 5]


def test_nested_inside_nested_and_arrays() -> None:
    left = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "ll": [[[1], [2, 3]], [[4]], None],
            "sl": [{"v": [1, 2]}, {"v": None}, {"v": []}],
            "arr": pl.Series([[1, 2], [3, 4], [5, 6]], dtype=pl.Array(pl.Int64, 2)),
        }
    )
    right = left.with_columns(
        ll=pl.Series([[[1], [2, 3]], [[5]], None]),
        arr=pl.Series([[1, 2], [4, 3], [5, 6]], dtype=pl.Array(pl.Int64, 2)),
    )
    res = dc.compare(left, right, "id")
    assert res.columns["ll"].mismatches == 1
    assert res.columns["sl"].mismatches == 0
    assert res.columns["arr"].mismatches == 1


@pytest.mark.parametrize("window_rows", [None, 2])
@pytest.mark.parametrize("verify_hash", [False, True])
@pytest.mark.parametrize("samples", [0, 5])
def test_nested_same_result_in_any_mode(window_rows: int | None, verify_hash: bool, samples: int) -> None:
    right = LEFT.with_columns(l=pl.Series([[1, 2], [1, 2], None, [], [3, None], [1, 1, 2]]))
    res = dc.compare(LEFT, right, "id", window_rows=window_rows, verify_hash=verify_hash, samples=samples)
    assert res.rows.mismatched == 1
    assert res.columns["l"].mismatches == 1


def test_unordered_lists_warning() -> None:
    res = dc.compare(LEFT, LEFT, "id", unordered_lists=["s", "nada"])
    assert res.warnings == ("unordered_lists names columns that are not compared lists: ['nada', 's']",)


def test_nested_from_parquet(tmp_path: Any) -> None:
    path = tmp_path / "l.parquet"
    LEFT.write_parquet(path, row_group_size=2)
    right = LEFT.with_columns(l=pl.Series([[1, 2], [2, 1], None, [], [3, None], [2, 1, 1]]))
    res = dc.compare(path, right, "id", unordered_lists=["l"], window_rows=2)
    assert res.columns["l"].mismatches == 0
    assert res.is_match


# ----------------------------------------------------------------------------- only_common_keys
def _tables(seed: int, *, key: str = "int", nulls: bool = False) -> tuple[pl.DataFrame, pl.DataFrame]:
    rnd = random.Random(seed)
    left_ids = list(range(0, 400))
    right_ids = list(range(250, 700))
    rnd.shuffle(right_ids)

    def make(ids: list[int], noise: float) -> pl.DataFrame:
        vals = [i * 1.5 + (1 if rnd.random() < noise else 0) for i in ids]
        names = [f"n{i % 7}" for i in ids]
        ks: list[Any] = ids if key == "int" else [f"k{i:05d}" for i in ids]
        if nulls:
            ks = [*ks, None]
            vals = [*vals, 1.0]
            names = [*names, "nulo"]
        return pl.DataFrame({"k": ks, "v": vals, "s": names})

    return make(left_ids, 0.0), make(right_ids, 0.1)


@pytest.mark.parametrize("key", ["int", "str"])
@pytest.mark.parametrize("nulls", [False, True])
@pytest.mark.parametrize("window_rows", [None, 37])
@pytest.mark.parametrize("abs_tol", [0.0, 0.5])
def test_only_common_keys_same_result(key: str, nulls: bool, window_rows: int | None, abs_tol: float) -> None:
    left, right = _tables(7, key=key, nulls=nulls)
    base = dc.compare(left, right, "k", window_rows=window_rows, abs_tol=abs_tol)
    fast = dc.compare(left, right, "k", window_rows=window_rows, abs_tol=abs_tol, only_common_keys=True)
    assert _summary(fast) == _summary(base)
    assert fast.rows.left_only == 250
    assert fast.rows.right_only == 300
    assert fast.columns["v"].samples.equals(base.columns["v"].samples)


def test_only_common_keys_from_parquet_and_edges(tmp_path: Any) -> None:
    left, right = _tables(3)
    lp, rp = tmp_path / "l.parquet", tmp_path / "r.parquet"
    left.write_parquet(lp, row_group_size=50)
    right.sort("k").write_parquet(rp, row_group_size=50)
    base = dc.compare(lp, rp, "k", window_rows=60)
    assert _summary(dc.compare(lp, rp, "k", window_rows=60, only_common_keys=True)) == _summary(base)
    # faixas que não se tocam: nada em comum
    far = right.with_columns(pl.col("k") + 10_000)
    res = dc.compare(left, far, "k", only_common_keys=True)
    assert (res.rows.common, res.rows.left_only, res.rows.right_only) == (0, 400, 450)
    # um lado vazio
    empty = left.head(0)
    res = dc.compare(empty, right, "k", only_common_keys=True)
    assert (res.rows.left_total, res.rows.right_only) == (0, 450)
    # chave composta: a faixa vale para a 1ª coluna
    lc = left.with_columns(k2=pl.col("k") % 3)
    rc = right.with_columns(k2=pl.col("k") % 3)
    assert _summary(dc.compare(lc, rc, ["k", "k2"], only_common_keys=True)) == _summary(dc.compare(lc, rc, ["k", "k2"]))


@pytest.mark.parametrize("key", [300, 10])  # dentro e fora da faixa em comum (250..399)
def test_only_common_keys_still_finds_duplicates(key: int) -> None:
    left, right = _tables(5)
    dup = pl.concat([left, left.filter(pl.col("k") == key)])
    res = dc.compare(dup, right, "k", only_common_keys=True)
    assert res.aborted
    assert res.errors[0] == "Duplicate join keys in the left table: 1 surplus rows"
