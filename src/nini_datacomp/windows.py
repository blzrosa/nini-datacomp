"""Planejamento das janelas: faixas da chave que cabem no orçamento de memória (``memory_limit``).

Cada janela é a faixa ``lo <= chave < hi`` da tupla inteira da chave (ordem lexicográfica), de qualquer tipo ordenável:
inteiro, texto, data, chave composta. Como a mesma chave cai sempre na mesma janela dos dois lados, as contagens das
janelas somam. As linhas com nulo em alguma coluna da chave vão todas para a primeira janela (nulo casa com nulo).

Como o plano é feito:

1. uma leitura só das colunas da chave de cada lado (bem mais barata que a tabela);
2. o custo por linha na memória (todas as colunas), medido pelo Polars numa amostra: é a memória que a janela ocupa
   depois de lida, não o tamanho comprimido no disco;
3. linhas por janela = ``memory_limit`` x 0,8 / (2 janelas em memória x (bytes por linha dos dois lados + o trabalho
   dos kernels)): a janela k+1 é lida enquanto a k é comparada;
4. as fronteiras são a chave a cada N linhas do lado maior, em ordem. Se esse lado está gravado em ordem no Parquet,
   cada fronteira vai para o começo do row group mais próximo (rodapé lido em C++, `io.row_groups`): nenhum row group
   é decodificado duas vezes.

Substitui as janelas do datacompolars, que exigiam a 1ª coluna da chave inteira (com chave de texto, a tabela inteira
ia numa janela só).
"""

from __future__ import annotations

import ctypes
import os
import re
import sys
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import reduce
from itertools import pairwise
from typing import Any

import polars as pl

from .io import Source, row_groups

__all__ = ["MIN_MEMORY", "Plan", "Window", "parse_memory_limit", "plan_windows"]

AUTO_FRACTION = 0.25  # "auto": 1/4 da memória livre...
AUTO_CAP = 2 * 1024**3  # ... até 2 GiB
MIN_MEMORY = 64 * 1024**2
SAFETY = 0.8  # folga para o que a estimativa não vê (buffers do Polars, fragmentação)
KERNEL_BYTES_PER_ROW = 32  # pares (8), hashes dos dois lados (16) e tabelas, por linha da janela
SAMPLE_ROWS = 50_000
_UNITS = {
    "": 1, "b": 1,
    "kb": 1000, "mb": 1000**2, "gb": 1000**3, "tb": 1000**4,
    "kib": 1024, "mib": 1024**2, "gib": 1024**3, "tib": 1024**4,
}  # fmt: skip


# ----------------------------------------------------------------------------- orçamento
def _available_memory() -> int:
    """Memória livre do sistema em bytes (sem psutil); 8 GiB se não der para saber."""
    if sys.platform == "win32":

        class MemoryStatus(ctypes.Structure):
            _fields_ = (
                ("length", ctypes.c_uint32),
                ("load", ctypes.c_uint32),
                ("total_phys", ctypes.c_uint64),
                ("avail_phys", ctypes.c_uint64),
                ("total_page", ctypes.c_uint64),
                ("avail_page", ctypes.c_uint64),
                ("total_virtual", ctypes.c_uint64),
                ("avail_virtual", ctypes.c_uint64),
                ("avail_ext", ctypes.c_uint64),
            )

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        windll = getattr(ctypes, "windll", None)  # só existe no Windows (getattr para o pyright de outros sistemas)
        if windll is not None and windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.avail_phys)
        return 8 * 1024**3
    for pages in ("SC_AVPHYS_PAGES", "SC_PHYS_PAGES"):  # livre (Linux); total (macOS não tem a livre)
        found = _sysconf_bytes(pages)
        if found:
            return found
    return 8 * 1024**3


def _sysconf_bytes(pages: str) -> int | None:
    sysconf = getattr(os, "sysconf", None)  # não existe no Windows
    if sysconf is None:
        return None
    try:
        return int(sysconf(pages)) * int(sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError):
        return None


def parse_memory_limit(value: int | str) -> int:
    """Bytes de ``memory_limit``: inteiro, ``"auto"`` (1/4 da memória livre, até 2 GiB) ou texto com unidade
    (``"512MB"``, ``"2GB"``, ``"1.5GiB"``; KB/MB/GB são de 1000, KiB/MiB/GiB de 1024). Mínimo de 64 MiB."""
    if isinstance(value, bool):
        raise TypeError("memory_limit deve ser um inteiro (bytes) ou um texto como '2GB'")
    if isinstance(value, int):
        nbytes = value
    elif value.strip().lower() == "auto":
        nbytes = min(AUTO_CAP, int(_available_memory() * AUTO_FRACTION))
    else:
        m = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]*)\s*", value)
        unit = m.group(2).lower() if m else None
        if m is None or unit not in _UNITS:
            raise ValueError(f"memory_limit inválido: {value!r} (ex.: 'auto', '512MB', '2GB', '1.5GiB')")
        nbytes = int(float(m.group(1)) * _UNITS[unit])
    return max(MIN_MEMORY, nbytes)


# ----------------------------------------------------------------------------- janelas
def _lit(value: Any, dtype: pl.DataType) -> pl.Expr:
    return pl.lit(value, dtype=dtype)


def _less(keys: Sequence[str], values: Sequence[Any], dtypes: Sequence[pl.DataType]) -> pl.Expr:
    """``tupla(keys) < values`` em ordem lexicográfica (as colunas não nulas)."""
    c = pl.col(keys[0])
    v = _lit(values[0], dtypes[0])
    if len(keys) == 1:
        return c < v
    return (c < v) | ((c == v) & _less(keys[1:], values[1:], dtypes[1:]))


def _show(bound: tuple[Any, ...]) -> str:
    return repr(bound[0]) if len(bound) == 1 else repr(bound)


@dataclass(frozen=True)
class Window:
    index: int
    lo: tuple[Any, ...] | None = None  # inclusivo (None = sem limite)
    hi: tuple[Any, ...] | None = None  # exclusivo
    with_nulls: bool = False  # inclui as linhas com nulo em alguma coluna da chave (só a primeira janela)

    @property
    def is_full(self) -> bool:
        return self.lo is None and self.hi is None and self.with_nulls

    def predicate(self, keys: Sequence[str], dtypes: Sequence[pl.DataType]) -> pl.Expr | None:
        """O filtro da janela (None = todas as linhas)."""
        if self.is_full:
            return None
        cols = [pl.col(k) for k in keys]
        inside = reduce(lambda a, b: a & b, [c.is_not_null() for c in cols])
        first = pl.col(keys[0])
        if self.lo is not None:
            inside = inside & ~_less(keys, self.lo, dtypes)
        if self.hi is not None:
            inside = inside & _less(keys, self.hi, dtypes)
        if self.with_nulls:
            return inside | reduce(lambda a, b: a | b, [c.is_null() for c in cols])
        # redundantes, mas simples: o Polars usa as estatísticas dos row groups para pular os de fora da faixa
        if self.lo is not None:
            inside = inside & (first >= _lit(self.lo[0], dtypes[0]))
        if self.hi is not None:
            inside = inside & (first <= _lit(self.hi[0], dtypes[0]))
        return inside

    def describe(self) -> str:
        if self.is_full:
            return "todas as linhas"
        lo = "-inf" if self.lo is None else _show(self.lo)
        hi = "+inf" if self.hi is None else _show(self.hi)
        return f"[{lo}, {hi})" + (" + chaves nulas" if self.with_nulls else "")


@dataclass(frozen=True)
class Plan:
    windows: list[Window]
    rows_per_window: int
    memory_limit: int
    bytes_per_row: tuple[float, float]  # esquerda, direita (todas as colunas, na memória)
    rows: tuple[int, int]  # linhas da esquerda e da direita
    base: str  # o lado de onde saíram as fronteiras ("left" ou "right")
    aligned_to_row_groups: bool = False
    notes: list[str] = field(default_factory=list)


def _key_frame(source: Source, keys: Sequence[str]) -> pl.DataFrame:
    """Só as colunas da chave (Categorical e Enum como texto: a ordem e a igualdade são as do texto)."""
    schema = source.lazy.collect_schema()
    exprs = [pl.col(k).cast(pl.String) if isinstance(schema[k], (pl.Categorical, pl.Enum)) else pl.col(k) for k in keys]
    if source.frame is not None:
        return source.frame.select(exprs)
    return source.lazy.select(exprs).collect(engine="streaming")


def _bytes_per_row(source: Source, sample_rows: int) -> float:
    """Bytes por linha na memória (todas as colunas), por uma amostra do começo da fonte."""
    frame = source.frame if source.frame is not None else source.lazy.head(sample_rows).collect()
    return frame.estimated_size() / frame.height if frame.height else 0.0


def _row_group_starts(source: Source) -> list[int]:
    """Posição (na ordem de leitura) do começo de cada row group, somando os arquivos."""
    starts, pos = [], 0
    for path in source.files:
        for n in row_groups(path):
            starts.append(pos)
            pos += n
    return starts


def _snap(cuts: list[int], starts: list[int], rows_per_window: int) -> list[int] | None:
    """Cada corte vai para o começo de row group mais próximo; None se os row groups forem maiores que a janela."""
    if len(starts) < 2 or max(b - a for a, b in pairwise(starts)) > rows_per_window:
        return None
    inner = starts[1:]  # o começo do primeiro row group (0) não é fronteira
    out: list[int] = []
    for c in cuts:
        i = bisect_left(inner, c)
        near = [inner[j] for j in (i - 1, i) if 0 <= j < len(inner)]
        best = min(near, key=lambda s: abs(s - c))
        if not out or best > out[-1]:
            out.append(best)
    return out


def plan_windows(
    left: Source,
    right: Source,
    keys: Sequence[str],
    memory_limit: int | str = "auto",
    *,
    window_rows: int | None = None,
    sample_rows: int = SAMPLE_ROWS,
) -> Plan:
    """As janelas para comparar ``left`` e ``right`` pela chave ``keys`` dentro de ``memory_limit``.

    ``window_rows`` (avançado) fixa as linhas por janela em vez de calculá-las pelo orçamento.
    """
    if not keys:
        raise ValueError("a chave precisa de ao menos uma coluna")
    budget = parse_memory_limit(memory_limit)
    kl, kr = _key_frame(left, keys), _key_frame(right, keys)
    bpr = (_bytes_per_row(left, sample_rows), _bytes_per_row(right, sample_rows))
    if window_rows is not None:
        if window_rows < 1:
            raise ValueError("window_rows deve ser >= 1")
        rows_per_window = window_rows
    else:
        rows_per_window = max(1, int(budget * SAFETY / (2 * (bpr[0] + bpr[1] + 2 * KERNEL_BYTES_PER_ROW))))
    rows = (kl.height, kr.height)
    base_name, base_keys, base_source = ("left", kl, left) if rows[0] >= rows[1] else ("right", kr, right)
    plan = Plan([Window(0, with_nulls=True)], rows_per_window, budget, bpr, rows, base_name)
    if max(rows) <= rows_per_window:
        return plan

    sorted_keys = base_keys.sort(list(keys), nulls_last=False)
    nulls = sorted_keys.height - sorted_keys.drop_nulls().height
    non_null = sorted_keys.drop_nulls()
    cuts = list(range(rows_per_window, non_null.height, rows_per_window))
    if not cuts:
        return plan

    aligned = False
    if base_source.is_parquet and nulls == 0 and sorted_keys.equals(base_keys):  # gravado em ordem, sem chave nula
        snapped = _snap(cuts, _row_group_starts(base_source), rows_per_window)
        if snapped:
            cuts, aligned = snapped, True
    bounds: list[tuple[Any, ...]] = []
    for row in non_null[cuts].iter_rows():
        if not bounds or row != bounds[-1]:  # chave repetida: a mesma fronteira uma vez só
            bounds.append(tuple(row))

    windows = [Window(0, None, bounds[0], with_nulls=True)]
    windows += [Window(i + 1, bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    windows.append(Window(len(bounds), bounds[-1], None))
    notes = [] if aligned or not base_source.is_parquet else ["fronteiras fora do começo dos row groups"]
    return Plan(windows, rows_per_window, budget, bpr, rows, base_name, aligned, notes)
