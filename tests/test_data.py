"""Dados de teste e de benchmark (`tools/make_data.py`): esperado pela construção x datacompolars."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

import make_data
from make_data import Spec, column_names, generate, load_expected


def test_column_names_skip_the_key() -> None:
    names = column_names(300)
    assert len(names) == len(set(names)) == 300
    assert make_data.KEY not in names
    with pytest.raises(ValueError, match="cols"):
        column_names(0)


def test_spec_validation_and_files() -> None:
    with pytest.raises(ValueError, match="divergence"):
        Spec(10, 3, divergence=1.5)
    assert Spec(10_000_000, 300).rows_per_file == 100_000  # 300 colunas: um row group por arquivo
    assert Spec(10_000_000, 10).rows_per_file % 100_000 == 0
    assert Spec(10, 3, tag="x").label == "10x3_x"


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    mp = pytest.MonkeyPatch()
    mp.setattr(make_data, "CELLS_PER_FILE", 13 * 2_000)  # 12 colunas + id: arquivos de 2 mil linhas
    try:
        spec = Spec(5_000, 12, divergence=0.05, row_group_size=1_000)
        return generate(spec, tmp_path_factory.mktemp("data"), log=lambda _: None)
    finally:
        mp.undo()


def test_layout(dataset: Path) -> None:
    left = sorted(p.name for p in (dataset / "left").iterdir())
    assert left == ["part-00000.parquet", "part-00001.parquet", "part-00002.parquet"]
    ids = pl.read_parquet(dataset / "left")["id"]
    assert ids.to_list() == list(range(5_000))
    right_ids = pl.read_parquet(dataset / "right")["id"]
    assert right_ids.is_unique().all()
    exp = load_expected(dataset)
    assert exp.left_rows == 5_000
    assert exp.right_rows == right_ids.len()
    assert exp.common + exp.left_only == exp.left_rows
    assert exp.mismatched > 0
    assert sum(exp.column_mismatches.values()) >= exp.mismatched


def test_types_cycle(dataset: Path) -> None:
    schema = pl.read_parquet_schema(dataset / "left" / "part-00000.parquet")
    kinds = [schema[c] for c in column_names(12)]
    assert kinds[0].is_integer()
    assert kinds[1].is_float()
    assert kinds[2] == pl.String
    assert kinds[3] == pl.Date
    assert kinds[7] == pl.Boolean
    assert kinds[9] == pl.Datetime


def test_expected_matches_datacompolars(dataset: Path) -> None:
    from datacompolars import CompareSettings, compare  # noqa: PLC0415

    variants: list[dict[str, Any]] = [{}, {"abs_tol": 1e-12}]  # hash e columnwise
    for extra in variants:
        res = compare(str(dataset / "left"), str(dataset / "right"), CompareSettings(join_columns=["id"], **extra))
        exp = load_expected(dataset)
        assert res.rows.common == exp.common
        assert res.rows.left_only == exp.left_only
        assert res.rows.right_only == exp.right_only
        assert res.rows.mismatched == exp.mismatched
        got = {st.column: st.mismatches for st in res.column_stats}
        assert got == {c: n for c, n in exp.column_mismatches.items() if n}


def test_same_seed_same_files(dataset: Path, tmp_path: Path) -> None:
    mp = pytest.MonkeyPatch()
    mp.setattr(make_data, "CELLS_PER_FILE", 13 * 2_000)
    try:
        again = generate(Spec(5_000, 12, divergence=0.05, row_group_size=1_000), tmp_path, log=lambda _: None)
    finally:
        mp.undo()
    for side in ("left", "right"):
        a = pl.read_parquet(dataset / side)
        b = pl.read_parquet(again / side)
        assert a.equals(b), side
