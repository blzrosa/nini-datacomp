"""Leitor Parquet em C++ (``cpp/parquet_read.cpp``), entregue ao Polars pela interface C do Arrow, sem pyarrow e sem
cópia: ``read_native`` devolve o DataFrame dos row groups pedidos ou None quando o arquivo está fora do suporte (o
chamador usa o Polars). Arquivo corrompido ou ilegível levanta ``ValueError``."""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from .._native import lib
from .arrow_c import ArrowArray, ArrowSchema, wrap_filled
from .blocks import resolve_threads

__all__ = ["read_native"]

_UNSUPPORTED = -1
_ERROR_CAP = 4096


def _native_dtype(dtype: pl.DataType) -> bool:
    """Tipos que o leitor sabe produzir (Datetime só sem fuso)."""
    if isinstance(dtype, pl.Datetime):
        return dtype.time_zone is None and dtype.time_unit in ("ms", "us", "ns")
    return dtype in (pl.Int32, pl.Int64, pl.Float32, pl.Float64, pl.Boolean, pl.String, pl.Date)


def read_native(path: Path, row_groups: Sequence[int], columns: Sequence[str] | None) -> pl.DataFrame | None:
    """As ``columns`` (None: todas) dos ``row_groups`` de ``path``, na ordem dos row groups, ou None se algo está fora
    do suporte do leitor (aninhados, Decimal, INT96, brotli/lz4, fuso, Categorical, UInt...) ou se os tipos que ele
    produziria divergem dos que o ``pl.read_parquet`` devolveria."""
    if not row_groups:
        return None
    try:
        schema = pl.read_parquet_schema(path)
    except pl.exceptions.PolarsError as e:  # arquivo truncado ou que não é Parquet
        raise ValueError(f"{Path(path).name}: {e}") from e
    names = list(schema) if columns is None else list(columns)
    if not names or any(n not in schema or not _native_dtype(schema[n]) for n in names):
        return None
    n_rg, n_col = len(row_groups), len(names)
    raw_path = str(path).encode("utf-8")
    raw_names = [n.encode("utf-8") for n in names]
    blob = b"".join(raw_names)
    lens = (ctypes.c_int64 * n_col)(*[len(b) for b in raw_names])
    groups = (ctypes.c_int32 * n_rg)(*row_groups)
    schemas = (ArrowSchema * (n_rg * n_col))()
    arrays = (ArrowArray * (n_rg * n_col))()
    rc = int(
        lib().dc_parquet_read(
            raw_path,
            len(raw_path),
            ctypes.addressof(groups),
            n_rg,
            blob,
            ctypes.addressof(lens),
            n_col,
            resolve_threads(),
            ctypes.addressof(schemas),
            ctypes.addressof(arrays),
        )
    )
    if rc == _UNSUPPORTED:
        return None
    if rc < 0:
        buf = ctypes.create_string_buffer(_ERROR_CAP)
        n = int(lib().dc_parquet_error(buf, _ERROR_CAP))
        raise ValueError(f"{Path(path).name}: {buf.raw[:n].decode('utf-8', 'replace') or 'erro na leitura do Parquet'}")
    columns_out: list[pl.Series] = []
    for c, name in enumerate(names):
        pieces = [
            pl.Series(values=wrap_filled(schemas[r * n_col + c], arrays[r * n_col + c]))  # pyright: ignore[reportArgumentType]
            for r in range(n_rg)
        ]
        series = pieces[0] if n_rg == 1 else pl.concat(pieces, rechunk=False)
        series = series.alias(name)
        if series.dtype != schema[name]:
            return None
        columns_out.append(series)
    return pl.DataFrame(columns_out)
