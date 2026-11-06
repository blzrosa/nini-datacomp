"""Leitor Parquet em C++ (`core/parquet.read_native`, `ReadOptions(reader="native")`): o resultado tem de ser igual
ao do `pl.read_parquet`, e tudo fora do suporte devolve None (o chamador cai no Polars)."""

from __future__ import annotations

import gc
import random
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import nini_datacomp as dc
from nini_datacomp.core.memory import pool_stats, release_memory

N_ROWS = 30_000
RG_ROWS = 8_000  # 4 row groups (o último com 6 000 linhas)
N_RG = 4
CODECS = ["uncompressed", "snappy", "zstd", "gzip"]


def _values(dtype: str) -> list[Any]:
    rng = random.Random(11)
    out: list[Any] = []
    for i in range(N_ROWS):
        if i % 7 == 3:
            out.append(None)
        elif dtype == "int32":
            out.append(rng.randint(-(2**31), 2**31 - 1))
        elif dtype == "int64":
            out.append(rng.randint(-(2**62), 2**62))
        elif dtype in ("float32", "float64"):
            out.append(rng.uniform(-1e6, 1e6) if i % 11 else float("nan"))
        elif dtype == "bool":
            out.append(rng.random() < 0.4)
        elif dtype == "string":  # parte repetida (dicionário), parte única, com acentos
            out.append(f"grupo-{i % 40}" if i % 3 else f"único-{i}-{rng.random():.6f}-ção")
        elif dtype == "date":
            out.append(date(1960, 1, 1) + timedelta(days=rng.randint(0, 40_000)))
        else:  # datetime_ms | datetime_us | datetime_ns
            out.append(datetime(2000, 1, 1) + timedelta(seconds=rng.randint(0, 2_000_000_000), milliseconds=5))
    return out


DTYPES: dict[str, Any] = {
    "int32": pl.Int32,
    "int64": pl.Int64,
    "float32": pl.Float32,
    "float64": pl.Float64,
    "bool": pl.Boolean,
    "string": pl.String,
    "date": pl.Date,
    "datetime_ms": pl.Datetime("ms"),
    "datetime_us": pl.Datetime("us"),
    "datetime_ns": pl.Datetime("ns"),
}


def _frame(dtype: str) -> pl.DataFrame:
    return pl.DataFrame({"id": pl.int_range(N_ROWS, eager=True), "v": pl.Series(_values(dtype), dtype=DTYPES[dtype])})


def _write(df: pl.DataFrame, path: Path, codec: str = "uncompressed", **kw: Any) -> Path:
    kw.setdefault("row_group_size", RG_ROWS)
    df.write_parquet(path, compression=codec, statistics=True, **kw)  # pyright: ignore[reportArgumentType]
    return path


def _read_native(path: Path, row_groups: list[int], columns: list[str] | None = None) -> pl.DataFrame | None:
    from nini_datacomp.core.parquet import read_native  # noqa: PLC0415

    return read_native(path, row_groups, columns)


@pytest.mark.parametrize("codec", CODECS)
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_roundtrip(tmp_path: Path, dtype: str, codec: str) -> None:
    path = _write(_frame(dtype), tmp_path / "t.parquet", codec, data_page_size=4096)
    got = _read_native(path, list(range(N_RG)))
    assert got is not None
    assert_frame_equal(got, pl.read_parquet(path))


@pytest.mark.parametrize("codec", ["snappy", "zstd"])
def test_snappy_long_runs(tmp_path: Path, codec: str) -> None:
    rng = random.Random(5)
    repeated = ["ab" * 300 + str(i % 3) for i in range(2_000)]  # cópias com offset menor que o tamanho (2)
    runs = ["x" * 5_000 + str(i) for i in range(300)] + [""] * 1_700  # literais e cópias longos
    noisy = [bytes(rng.getrandbits(8) for _ in range(300)).hex() for _ in range(2_000)]  # 1,2 MB sem repetição
    df = pl.DataFrame({"repeated": repeated, "runs": runs, "noisy": noisy})
    path = _write(df, tmp_path / "t.parquet", codec, row_group_size=2_000, data_page_size=4 * 1024**2)
    got = _read_native(path, [0])
    assert got is not None
    assert_frame_equal(got, pl.read_parquet(path))


def test_dictionary_and_plain_pages(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rng = random.Random(3)
    values = [f"valor-{rng.randrange(10**9)}" for _ in range(60_000)]  # cardinalidade alta: o dicionário estoura
    table = pa.table({"s": values, "i": list(range(60_000))})
    path = tmp_path / "t.parquet"
    pq.write_table(table, path, row_group_size=60_000, dictionary_pagesize_limit=64 * 1024, data_page_size=32 * 1024)
    got = _read_native(path, [0])
    assert got is not None
    assert_frame_equal(got, pl.read_parquet(path))


def test_page_v2(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    df = pl.concat([_frame("int64").rename({"v": "i"}), _frame("string").select(s="v")], how="horizontal")
    for codec in ("none", "snappy", "zstd"):
        path = tmp_path / f"v2_{codec}.parquet"
        pq.write_table(
            pa.table(df.to_dict(as_series=False)),
            path,
            row_group_size=RG_ROWS,
            data_page_version="2.0",
            compression=codec,
            data_page_size=8 * 1024,
        )
        got = _read_native(path, list(range(N_RG)))
        assert got is not None
        assert_frame_equal(got, pl.read_parquet(path))


def test_row_group_subset(tmp_path: Path) -> None:
    df = _frame("int64")
    path = _write(df, tmp_path / "t.parquet", "zstd")
    got = _read_native(path, [1, 3], ["v", "id"])
    expected = pl.concat([df.slice(RG_ROWS, RG_ROWS), df.slice(3 * RG_ROWS)]).select("v", "id")
    assert got is not None
    assert_frame_equal(got, expected)


def _unsupported(kind: str, tmp_path: Path) -> tuple[Path, Path]:
    n = 500
    extra: dict[str, pl.Series] = {
        "list": pl.Series("v", [[i, i + 1] for i in range(n)]),
        "decimal": pl.Series("v", [Decimal(i) / 100 for i in range(n)], dtype=pl.Decimal(10, 2)),
        "brotli": pl.Series("v", range(n)),
        "tz": pl.Series("v", [datetime(2020, 1, 1)] * n).dt.replace_time_zone("UTC"),
        "categorical": pl.Series("v", [f"c{i % 5}" for i in range(n)], dtype=pl.Categorical),
    }
    left = pl.DataFrame({"id": pl.int_range(n, eager=True)}).with_columns(extra[kind])
    right = left
    if kind == "brotli":  # uma diferença, para o resultado não ser trivial (no tz, o Windows sem tzdata não lê valores)
        right = left.with_columns(pl.when(pl.col("id") == 7).then(None).otherwise(pl.col("v")).alias("v"))
    codec = "brotli" if kind == "brotli" else "uncompressed"
    return (
        _write(left, tmp_path / "l.parquet", codec, row_group_size=200),
        _write(right, tmp_path / "r.parquet", codec, row_group_size=200),
    )


def _comparable(result: dc.ComparisonResult) -> dict[str, Any]:
    out = result.to_dict()
    for k in ("execution", "created_at"):
        out.pop(k)
    return out


@pytest.mark.parametrize("kind", ["list", "decimal", "brotli", "tz", "categorical"])
def test_unsupported_falls_back(tmp_path: Path, kind: str) -> None:
    left, right = _unsupported(kind, tmp_path)
    assert _read_native(left, [0, 1, 2]) is None
    native = dc.compare(left, right, "id", read=dc.ReadOptions(reader="native"))
    default = dc.compare(left, right, "id", read=dc.ReadOptions(reader="polars"))
    assert _comparable(native) == _comparable(default)


def test_buffers_released(tmp_path: Path) -> None:
    df = pl.DataFrame({"i": pl.int_range(400_000, eager=True), "s": [f"texto-{i % 977}" for i in range(400_000)]})
    path = _write(df, tmp_path / "t.parquet", row_group_size=200_000)
    gc.collect()
    release_memory()
    before = pool_stats()["in_use"]
    got = _read_native(path, [0, 1])
    assert got is not None
    assert pool_stats()["in_use"] > before  # os buffers são do pool, entregues ao Polars sem cópia
    del got
    gc.collect()
    release_memory()
    assert pool_stats()["in_use"] == before


def test_corrupt_file_raises(tmp_path: Path) -> None:
    path = _write(_frame("int64"), tmp_path / "t.parquet", "snappy")
    data = path.read_bytes()
    for name, size in (("rodape", len(data) - 40), ("metade", len(data) // 2), ("curto", 6)):
        broken = tmp_path / f"{name}.parquet"
        broken.write_bytes(data[:size])
        with pytest.raises((OSError, ValueError)):
            _read_native(broken, [0])
