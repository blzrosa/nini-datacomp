"""Leitura com sobra (``Plan.carry``): cada row group do lado que não é a base é decodificado uma vez, e o que passa do
fim da janela entra na frente da seguinte. O resultado é o do caminho por faixa da chave."""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path

import polars as pl
import pytest

import nini_datacomp as dc
from nini_datacomp.io import Source, open_source
from nini_datacomp.pipeline import WindowReader
from nini_datacomp.windows import Plan, plan_windows


def _write(df: pl.DataFrame, folder: Path, row_group_size: int) -> Source:
    folder.mkdir(parents=True, exist_ok=True)
    df.write_parquet(folder / "part.parquet", row_group_size=row_group_size)
    return open_source(folder)


def _tables(n: int = 12_000) -> tuple[pl.DataFrame, pl.DataFrame]:
    left = pl.DataFrame({"id": range(n), "v": [i * 2 for i in range(n)], "s": [f"x{i}" for i in range(n)]})
    right = left.filter(pl.col("id") % 97 != 5).with_columns(
        pl.when(pl.col("id") % 211 == 0).then(-1).otherwise(pl.col("v")).alias("v")
    )
    return left, right


def _windows(plan: Plan, ls: Source, rs: Source) -> tuple[list[tuple[pl.DataFrame, pl.DataFrame]], WindowReader]:
    reader = WindowReader(ls, rs, plan, ["id"], [pl.Int64()], prefetch=False)
    out = [(lf.sort("id"), rf.sort("id")) for _, lf, rf in reader]
    return out, reader


def test_carry_reads_each_row_group_once(tmp_path: Path) -> None:
    left, right = _tables()
    ls = _write(left, tmp_path / "l", 1_000)
    rs = _write(right, tmp_path / "r", 700)  # row groups de tamanhos diferentes: as fronteiras cortam os da direita
    plan = plan_windows(ls, rs, ["id"], window_rows=1_500)
    assert plan.base == "left"
    assert plan.carry[0] is None
    assert plan.carry[1] is not None
    assert plan.read_amplification == (1.0, 1.0)
    old = dataclasses.replace(plan, carry=(None, None))

    with_carry, reader = _windows(plan, ls, rs)
    without, _ = _windows(old, ls, rs)
    assert len(with_carry) == len(plan.windows) > 4
    assert reader.decoded_with_carry == (None, right.height)  # cada linha da direita decodificada uma vez
    for (lf_a, rf_a), (lf_b, rf_b) in zip(with_carry, without, strict=True):
        assert lf_a.equals(lf_b)
        assert rf_a.equals(rf_b)

    # a comparação inteira dá o mesmo que sem janelas
    whole = dc.compare(left, right, on="id")
    windowed = dc.compare(ls.files[0].parent, rs.files[0].parent, on="id", window_rows=1_500)
    assert windowed.rows == whole.rows
    assert windowed.columns["v"].mismatches == whole.columns["v"].mismatches
    assert windowed.columns["v"].samples.equals(whole.columns["v"].samples)
    assert windowed.execution.read_amplification == (1.0, 1.0)


def test_carry_when_the_base_is_the_right_side(tmp_path: Path) -> None:
    left, right = _tables()
    ls = _write(left.head(9_000), tmp_path / "l", 700)
    rs = _write(right, tmp_path / "r", 1_000)
    plan = plan_windows(ls, rs, ["id"], window_rows=1_500)
    assert plan.base == "right"
    assert plan.carry[0] is not None
    assert plan.carry[1] is None
    with_carry, reader = _windows(plan, ls, rs)
    without, _ = _windows(dataclasses.replace(plan, carry=(None, None)), ls, rs)
    assert reader.decoded_with_carry == (9_000, None)
    assert all(a[0].equals(b[0]) and a[1].equals(b[1]) for a, b in zip(with_carry, without, strict=True))


def test_carry_with_an_extra_file_listed_first(tmp_path: Path) -> None:
    """Um arquivo ``extra-*`` com chaves altas vem antes do resto na ordem dos arquivos (como nos dados do benchmark):
    a ordem dos arquivos não importa, só as faixas dos row groups não se sobreporem."""
    left, right = _tables()
    ls = _write(left, tmp_path / "l", 1_000)
    folder = tmp_path / "r"
    folder.mkdir()
    extra = pl.DataFrame({"id": range(20_000, 20_300), "v": range(300), "s": ["e"] * 300})
    extra.write_parquet(folder / "extra-00000.parquet", row_group_size=300)
    right.write_parquet(folder / "part-00000.parquet", row_group_size=700)
    rs = open_source(folder)
    assert rs.files[0].name.startswith("extra")
    plan = plan_windows(ls, rs, ["id"], window_rows=1_500)
    assert plan.carry[1] is not None
    assert plan.read_amplification == (1.0, 1.0)
    with_carry, reader = _windows(plan, ls, rs)
    without, _ = _windows(dataclasses.replace(plan, carry=(None, None)), ls, rs)
    assert reader.decoded_with_carry == (None, right.height + extra.height)
    assert all(a[0].equals(b[0]) and a[1].equals(b[1]) for a, b in zip(with_carry, without, strict=True))


def test_carry_with_rows_out_of_order_inside_row_groups(tmp_path: Path) -> None:
    """Row groups em ordem mas linhas embaralhadas dentro deles: a busca binária não vale, separa pelo valor."""
    left, right = _tables(6_000)
    rng = random.Random(7)
    pieces = []
    for start in range(0, right.height, 700):
        part = right.slice(start, 700)
        pieces.append(part.select(pl.all().gather(rng.sample(range(part.height), part.height))))
    shuffled = pl.concat(pieces)
    ls = _write(left, tmp_path / "l", 1_000)
    rs = _write(shuffled, tmp_path / "r", 700)
    plan = plan_windows(ls, rs, ["id"], window_rows=1_500)
    assert plan.carry[1] is not None
    with_carry, reader = _windows(plan, ls, rs)
    without, _ = _windows(dataclasses.replace(plan, carry=(None, None)), ls, rs)
    assert reader.decoded_with_carry[1] == shuffled.height
    assert all(a[0].equals(b[0]) and a[1].equals(b[1]) for a, b in zip(with_carry, without, strict=True))


def _carry_of(left: pl.DataFrame, right: pl.DataFrame, tmp_path: Path, keys: list[str], **kw: object) -> Plan:
    ls = _write(left, tmp_path / "l", 1_000)
    rs = _write(right, tmp_path / "r", 700)
    return plan_windows(ls, rs, keys, window_rows=1_500, **kw)  # type: ignore[arg-type]


def test_carry_falls_back(tmp_path: Path) -> None:
    left, right = _tables()
    assert _carry_of(left, right, tmp_path / "ok", ["id"]).carry != (None, None)  # a referência: serve

    as_text = (
        left.with_columns(pl.format("k{}", pl.col("id").cast(pl.String).str.zfill(6)).alias("id")),
        right.with_columns(pl.format("k{}", pl.col("id").cast(pl.String).str.zfill(6)).alias("id")),
    )
    assert _carry_of(*as_text, tmp_path / "text", ["id"]).carry == (None, None)  # chave de texto

    both = [df.with_columns((pl.col("id") % 50).alias("g")).select("g", "id", "v", "s") for df in (left, right)]
    assert _carry_of(both[0], both[1], tmp_path / "composite", ["g", "id"]).carry == (None, None)  # chave composta

    with_null = right.with_columns(pl.when(pl.col("id") == 7_000).then(None).otherwise(pl.col("id")).alias("id"))
    assert _carry_of(left, with_null, tmp_path / "nulls", ["id"]).carry == (None, None)  # chave nula

    # faixas sobrepostas, e com menos linhas que a esquerda (a direita não é a base: é o lado que poderia ter sobra)
    overlapping = pl.concat([right.slice(0, 800), right.slice(500, 5_000)])
    assert _carry_of(left, overlapping, tmp_path / "overlap", ["id"]).carry == (None, None)  # faixas sobrepostas

    shuffled = right.sample(fraction=1.0, shuffle=True, seed=3)
    assert _carry_of(left, shuffled, tmp_path / "shuffled", ["id"]).carry == (None, None)  # fora de ordem


def test_carry_not_with_column_groups(tmp_path: Path) -> None:
    n, rg = 20_000, 5_000
    df = pl.DataFrame({"id": range(n), **{f"s{j}": [f"{i:06d}" + "x" * 500 for i in range(n)] for j in range(4)}})
    left = _write(df, tmp_path / "l", rg)
    right = _write(df.filter(pl.col("id") % 3 != 0), tmp_path / "r", 3_000)
    plan = plan_windows(left, right, ["id"], memory_limit=64 * 1024**2, allocator="tuned")
    if plan.column_groups == 1:
        pytest.skip("o plano coube sem grupos de colunas")
    assert plan.carry == (None, None)
