"""Colunas do Polars lidas pela interface C do Arrow, sem cópia e sem o pyarrow (`core.arrow`, `core.arrow_c`)."""

from __future__ import annotations

import ctypes
import gc
import subprocess
import sys
import textwrap
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import numpy as np
import polars as pl
import pytest

import nini_datacomp as dc
from nini_datacomp.core import arrow_c
from nini_datacomp.core.arrow import Column, import_column, value_width
from nini_datacomp.core.memory import POOL_MIN_BYTES, empty, pool_stats


def _bits(col: Column) -> list[bool]:
    """Validade como lista (True = presente), lida da memória do Polars."""
    if col.validity is None:
        return [True] * col.length
    nbytes = (col.bit_offset + col.length + 7) // 8
    raw = np.frombuffer(ctypes.string_at(col.validity, nbytes), np.uint8)
    bits = np.unpackbits(raw, bitorder="little")[col.bit_offset : col.bit_offset + col.length]
    return [bool(b) for b in bits]


def _text(col: Column, i: int) -> bytes:
    assert col.data is not None
    view = ctypes.string_at(col.data + 16 * i, 16)
    n = int.from_bytes(bytes(view[0:4]), "little")
    if n <= 12:
        return bytes(view[4 : 4 + n])
    buf = int.from_bytes(bytes(view[8:12]), "little")
    off = int.from_bytes(bytes(view[12:16]), "little")
    return ctypes.string_at(col.texts[buf] + off, n)


@pytest.mark.parametrize(
    ("dtype", "np_dtype"),
    [
        (pl.Int8, np.int8),
        (pl.Int16, np.int16),
        (pl.Int32, np.int32),
        (pl.Int64, np.int64),
        (pl.UInt8, np.uint8),
        (pl.UInt16, np.uint16),
        (pl.UInt32, np.uint32),
        (pl.UInt64, np.uint64),
        (pl.Float32, np.float32),
        (pl.Float64, np.float64),
    ],
)
def test_numeric_without_copy(dtype: pl.DataType, np_dtype: type) -> None:
    s = pl.Series("x", [1, None, 3, 4, None, 6], dtype=dtype)
    col = import_column(s)
    assert col.length == 6
    assert col.null_count == 2
    assert col.width == np.dtype(np_dtype).itemsize
    assert _bits(col) == [True, False, True, True, False, True]
    v = col.values(np_dtype)
    assert [v[i] for i in (0, 2, 3, 5)] == [1, 3, 4, 6]
    assert not v.flags.writeable


def test_no_nulls_has_no_validity() -> None:
    col = import_column(pl.Series("x", np.arange(10, dtype=np.int64)))
    assert col.null_count == 0
    assert col.validity is None
    assert col.values(np.int64).tolist() == list(range(10))


def test_slice_applies_offset() -> None:
    s = pl.Series("x", [None if i % 3 == 0 else i for i in range(100)], dtype=pl.Int64).slice(7, 50)
    col = import_column(s)
    assert col.length == 50
    assert col.values(np.int64)[1] == 8  # o deslocamento já está aplicado em `data`
    assert _bits(col) == [x is not None for x in s.to_list()]
    assert col.null_count == s.null_count()


def test_many_chunks_are_joined() -> None:
    s = pl.concat([pl.Series("x", [1, 2], dtype=pl.Int32), pl.Series("x", [3, None], dtype=pl.Int32)], rechunk=False)
    assert s.n_chunks() == 2
    col = import_column(s)
    assert col.length == 4
    assert col.values(np.int32)[:3].tolist() == [1, 2, 3]
    assert col.null_count == 1


def test_booleans_are_bits() -> None:
    s = pl.Series("b", [True, False, None, True, False, True, True, False, True], dtype=pl.Boolean).slice(1, 7)
    col = import_column(s)
    assert col.is_bool
    assert col.width == 0
    assert col.data is not None
    nbytes = (col.bit_offset + col.length + 7) // 8
    raw = np.frombuffer(ctypes.string_at(col.data, nbytes), np.uint8)
    bits = np.unpackbits(raw, bitorder="little")[col.bit_offset : col.bit_offset + col.length]
    got = [None if not ok else bool(b) for b, ok in zip(bits, _bits(col), strict=True)]
    assert got == s.to_list()


def test_strings_short_and_long_with_nulls() -> None:
    values = ["F", None, "", "doze bytes!!", "treze bytes!!", "um texto bem mais comprido que doze bytes", "ção ü"]
    s = pl.Series("t", values * 50).slice(3, 300)
    col = import_column(s)
    assert col.is_text
    assert col.width == 16
    assert col.null_count == s.null_count()
    valid = _bits(col)
    for i, want in enumerate(s.to_list()):
        assert valid[i] == (want is not None)
        if want is not None:
            assert _text(col, i).decode() == want
    assert len(col.texts) == len(col.text_sizes)


@pytest.mark.parametrize(
    ("s", "width"),
    [
        (pl.Series("d", [date(2026, 1, 1), None]), 4),
        (pl.Series("dt", [datetime(2026, 1, 1, 12), None]), 8),
        (pl.Series("dtz", [datetime(2026, 1, 1, 12), None]).dt.replace_time_zone("America/Sao_Paulo"), 8),
        (pl.Series("du", [timedelta(seconds=5), None]), 8),
        (pl.Series("tm", [time(12, 30), None]), 8),
        (pl.Series("dec", [Decimal("1.25"), None], dtype=pl.Decimal(10, 2)), 16),
    ],
)
def test_temporal_and_decimal(s: pl.Series, width: int) -> None:
    try:
        col = import_column(s)
    except dc.UnsupportedTypeError:
        if s.dtype == pl.Decimal:  # Polars antigos podem não exportar Decimal de jeito nenhum: o erro tem de ser claro
            pytest.skip(f"Decimal não exportado pelo Polars {pl.__version__}")
        raise
    assert col.dtype == s.dtype
    assert col.width == width
    assert col.null_count == 1
    assert _bits(col) == [True, False]


def test_value_width() -> None:
    assert value_width(b"l") == 8
    assert value_width(b"tsu:UTC") == 8
    assert value_width(b"tDn") == 8
    assert value_width(b"d:10,2") == 16
    assert value_width(b"d:40,2,256") == 32
    assert value_width(b"vu") is None
    assert value_width(b"+l") is None


def test_nested_is_rejected() -> None:
    with pytest.raises(dc.UnsupportedTypeError, match="não é suportada"):
        import_column(pl.Series("l", [[1, 2], [3]]))


def test_empty_series() -> None:
    col = import_column(pl.Series("x", [], dtype=pl.Int64))
    assert col.length == 0
    assert col.null_count == 0
    assert col.values(np.int64).size == 0


def test_column_keeps_memory_alive() -> None:
    s = pl.Series("x", np.arange(1_000_000, dtype=np.int64))
    col = import_column(s)
    del s
    gc.collect()
    assert int(col.values(np.int64)[-1]) == 999_999


def test_export_keeps_buffers_until_series_dies() -> None:
    gc.collect()
    before = pool_stats()["in_use"]
    n = POOL_MIN_BYTES // 8
    values = empty(n, np.int64)
    values[:] = np.arange(n)
    validity = np.full(-(-n // 64), np.uint64(0xFFFFFFFFFFFFFFFF))
    validity[0] = np.uint64(0xFFFFFFFFFFFFFFFE)  # linha 0 nula
    s = pl.Series(values=arrow_c.export_array(b"l", n, [validity, values], 1))  # pyright: ignore[reportArgumentType]
    del values
    gc.collect()
    assert pool_stats()["in_use"] > before  # o Polars ainda segura o bloco
    assert s.null_count() == 1
    assert s[0] is None
    assert s.sum() == n * (n - 1) // 2
    del s
    gc.collect()
    assert pool_stats()["in_use"] == before  # o release do Polars devolveu o bloco
    assert not arrow_c._live


def test_works_without_pyarrow() -> None:
    """Ler colunas de todos os tipos não importa o pyarrow (instalado ou não)."""
    code = textwrap.dedent(
        """
        import sys
        from datetime import date
        import polars as pl
        from nini_datacomp.core.arrow import import_column
        df = pl.DataFrame({
            "i": [1, None, 3], "f": [1.5, 2.5, None], "s": ["a", None, "um texto comprido demais"],
            "b": [True, None, False], "d": [date(2026, 1, 1), None, date(2026, 1, 3)],
        })
        cols = [import_column(df[c]) for c in df.columns]
        assert [c.null_count for c in cols] == [1, 1, 1, 1, 1]
        assert "pyarrow" not in sys.modules, "algo importou o pyarrow"
        print("ok")
        """
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"
