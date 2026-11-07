"""Perfil ``wide84`` do make_data: 84 colunas, 48 exatas (com texto) e 36 reais."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import make_data
from make_data import Spec, column_names, generate, load_expected


def test_wide84_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(make_data, "CELLS_PER_FILE", 85 * 1_000)  # arquivos de mil linhas
    spec = Spec(2_500, 84, divergence=0.05, only_left=0.02, only_right=0.01, profile="wide84", tag="w")
    root = generate(spec, tmp_path, log=lambda _: None)
    assert root.name == "2500x84_w"
    schema = pl.read_parquet_schema(sorted((root / "left").glob("*.parquet"))[0])
    dtypes = [schema[c] for c in column_names(84)]
    assert len(dtypes) == 84
    assert sum(d.is_float() for d in dtypes) == 36
    exact = [d for d in dtypes if not d.is_float()]
    assert len(exact) == 48
    assert any(d == pl.String for d in exact)
    assert any(d == pl.Boolean for d in exact)
    assert any(d == pl.Date for d in exact)
    assert any(d == pl.Datetime for d in exact)
    assert any(d.is_integer() for d in exact)
    # as contagens esperadas batem com as geradas
    exp = load_expected(root)
    left, right = pl.read_parquet(root / "left"), pl.read_parquet(root / "right")
    assert left.height == exp.left_rows == 2_500
    assert right.height == exp.right_rows
    joined = left.join(right, on="id", how="inner")
    assert joined.height == exp.common
    assert exp.left_only == left.height - exp.common
    assert exp.right_only == right.height - exp.common
    assert exp.left_only > 0
    assert exp.right_only > 0
    assert exp.mismatched > 0
