"""Fontes (`io.py`) e o leitor de rodapé Parquet em C++ (``dc_parquet_row_groups``)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nini_datacomp.io import key_stats, open_source, row_groups


def test_frames() -> None:
    df = pl.DataFrame({"id": [1, 2]})
    s = open_source(df)
    assert s.frame is df
    assert not s.is_parquet
    assert open_source(df.lazy()).lazy.collect().equals(df)


def test_folder_glob_and_file(tmp_path: Path) -> None:
    df = pl.DataFrame({"id": range(10)})
    (tmp_path / "sub").mkdir()
    (tmp_path / "_tmp").mkdir()
    df.slice(0, 5).write_parquet(tmp_path / "a.parquet")
    df.slice(5, 5).write_parquet(tmp_path / "sub" / "b.parquet")
    df.write_parquet(tmp_path / "_tmp" / "lixo.parquet")  # pasta oculta: fica de fora
    (tmp_path / ".oculto.parquet").write_bytes(b"x")
    folder = open_source(tmp_path)
    assert [f.name for f in folder.files] == ["a.parquet", "b.parquet"]
    assert folder.lazy.collect().equals(df)
    assert [f.name for f in open_source(str(tmp_path / "*.parquet")).files] == ["a.parquet"]
    assert open_source(tmp_path / "a.parquet").lazy.collect().height == 5


def test_scan_rows_slices_before_steps(tmp_path: Path) -> None:
    """`scan(files, rows=(a, n))`: as linhas a..a+n do arquivo (o `slice` vem antes dos passos) que passam no filtro,
    não as n primeiras já filtradas."""
    path = tmp_path / "a.parquet"
    pl.DataFrame({"id": range(100)}).write_parquet(path, row_group_size=10)
    src = open_source(path).filter(pl.col("id") % 2 == 0)
    got = src.scan([path], rows=(10, 20)).collect()
    assert got["id"].to_list() == list(range(10, 30, 2))
    assert src.scan([path], rows=None).collect()["id"].to_list() == list(range(0, 100, 2))
    assert src.scan([path]).collect()["id"].to_list() == list(range(0, 100, 2))


def test_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        open_source(tmp_path / "nada")
    with pytest.raises(FileNotFoundError):
        open_source(tmp_path)  # pasta sem Parquet
    with pytest.raises(FileNotFoundError):
        open_source(str(tmp_path / "*.parquet"))
    (tmp_path / "x.unknown").write_text("a\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="formato desconhecido"):
        open_source(tmp_path / "x.unknown")
    with pytest.raises(TypeError):
        open_source(123)  # type: ignore[arg-type]


@pytest.mark.parametrize(("n", "rg"), [(1, 10), (10_000, 1_000), (10_000, 3_333), (50_000, 50_000)])
def test_row_groups_from_footer(tmp_path: Path, n: int, rg: int) -> None:
    path = tmp_path / "x.parquet"
    df = pl.DataFrame({"id": range(n), "s": [f"t{i}" for i in range(n)], "f": [i / 3 for i in range(n)]})
    df.write_parquet(path, row_group_size=rg, statistics=True)
    groups = row_groups(path)
    assert sum(groups) == n  # o leitor conta todas as linhas
    assert all(g > 0 for g in groups)
    # o tamanho de cada row group é escolha de quem grava (o Polars 1.30 não corta exatamente em 3.333): só o total vale


def test_key_stats_text(tmp_path: Path) -> None:
    n = 3_000
    values: list[str | None] = [f"k{(i * 7919) % n:05d}" for i in range(n)]
    values[10] = "açaí-ünico"  # acento: o mínimo/máximo é comparado em bytes UTF-8
    values[1_500] = None  # um row group com nulo
    df = pl.DataFrame({"s": values, "i": range(n)}, schema={"s": pl.String, "i": pl.Int64})
    path = tmp_path / "t.parquet"
    df.write_parquet(path, row_group_size=1_000, statistics=True)
    stats = key_stats(path, "s", text=True)
    sizes = row_groups(path)
    assert len(stats) == len(sizes) >= 3
    start = 0
    for g, size in zip(stats, sizes, strict=True):
        part = df["s"].slice(start, size)
        start += size
        assert g.rows == size
        assert g.min == part.min()
        assert g.max == part.max()
        assert g.nulls == part.null_count()
    assert any(g.nulls for g in stats)
    assert any(isinstance(g.min, str) and not g.min.isascii() for g in stats) or any(
        isinstance(g.max, str) and not g.max.isascii() for g in stats
    )
    # coluna inteira com `text=True` e coluna de texto sem `text`: sem mínimo e máximo
    assert all(g.min is None and g.max is None for g in key_stats(path, "i", text=True))
    assert all(g.min is None and g.max is None for g in key_stats(path, "s"))
    # sem estatísticas gravadas
    bare = tmp_path / "bare.parquet"
    df.write_parquet(bare, row_group_size=1_000, statistics=False)
    assert all(g.min is None and g.max is None and g.nulls is None for g in key_stats(bare, "s", text=True))


def test_row_groups_wide_file(tmp_path: Path) -> None:
    path = tmp_path / "wide.parquet"
    pl.DataFrame({f"c{i}": range(3_000) for i in range(300)}).write_parquet(path, row_group_size=1_000)
    assert sum(row_groups(path)) == 3_000


def test_row_groups_rejects_non_parquet(tmp_path: Path) -> None:
    bad = tmp_path / "bad.parquet"
    bad.write_bytes(b"isto nao e parquet nenhum")
    with pytest.raises(ValueError, match="PAR1"):
        row_groups(bad)
    broken = tmp_path / "broken.parquet"
    broken.write_bytes(b"PAR1" + b"\x15\x00\x00\x00" * 4 + (12).to_bytes(4, "little") + b"PAR1")
    with pytest.raises(ValueError, match=r"ilegível|inválido"):
        row_groups(broken)
