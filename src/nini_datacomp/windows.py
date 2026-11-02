"""Planejamento das janelas: faixas da chave que cabem no orçamento de memória (``memory_limit``).

Cada janela é a faixa ``lo <= chave < hi`` da tupla inteira da chave (ordem lexicográfica), de qualquer tipo ordenável:
inteiro, texto, data, chave composta. Como a mesma chave cai sempre na mesma janela dos dois lados, as contagens das
janelas somam. As linhas com nulo em alguma coluna da chave vão todas para a primeira janela (nulo casa com nulo).

Como o plano é feito, sem a memória crescer com a tabela:

1. uma passada só pelas colunas da chave de cada lado, sem guardá-las: linhas e se há chave nula;
2. o custo por linha na memória (todas as colunas), medido pelo Polars numa amostra: é a memória que a janela ocupa
   depois de lida, não o tamanho comprimido no disco;
3. linhas por janela = ``memory_limit`` x 0,8 / (4 x bytes por linha dos dois lados + o trabalho dos kernels). O 4 é
   o medido na fase 1c: a janela em uso, a seguinte sendo lida e as cópias temporárias do Polars;
4. fronteiras: se um lado é Parquet gravado em ordem (a chave do começo de cada row group, lida do arquivo, está em
   ordem), cada janela começa num row group e nenhum é decodificado duas vezes. Senão, os quantis de uma amostra
   regular de até 200 mil chaves.

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
SAFETY = 0.8  # folga para o que a estimativa não vê (fragmentação, o próprio processo)
# Cópias de uma janela vivas no pico: a janela em uso, a seguinte sendo lida e as cópias temporárias que o Polars faz
# ao decodificar e filtrar. Medido na fase 1c (docs/benchmarks/fase1.md): o pico ficou perto de 4x a maior janela.
IN_FLIGHT = 4
KERNEL_BYTES_PER_ROW = 32  # pares (8), hashes dos dois lados (16) e tabelas, por linha da janela
SAMPLE_ROWS = 50_000  # linhas lidas para medir os bytes por linha
SAMPLE_KEYS = 200_000  # chaves da amostra que define as fronteiras quando a fonte não está em ordem
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
    with_nulls: bool = False  # inclui as linhas com nulo em alguma coluna da chave (só a primeira, se houver nulas)

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
    # por lado (esquerda, direita): para cada janela, os índices (em Source.files) dos arquivos Parquet cuja faixa de
    # chaves encosta nela; None = ler a fonte inteira (não é Parquet, ou um arquivo só)
    files: tuple[list[tuple[int, ...]] | None, list[tuple[int, ...]] | None] = (None, None)
    column_groups: int = 1  # em quantos grupos de colunas ler cada janela (1 = todas de uma vez)
    column_bytes: dict[str, float] = field(default_factory=dict)  # bytes por linha de cada coluna, os dois lados

    def split_columns(self, columns: Sequence[str]) -> list[list[str]]:
        """``columns`` em até `column_groups` grupos seguidos (na mesma ordem), de tamanhos parecidos em bytes."""
        groups = max(1, min(self.column_groups, len(columns)))
        if groups == 1:
            return [list(columns)]
        sizes = [self.column_bytes.get(c, 0.0) or 1.0 for c in columns]
        target = sum(sizes) / groups
        out: list[list[str]] = [[]]
        acc = 0.0
        for i, (c, size) in enumerate(zip(columns, sizes, strict=True)):
            # novo grupo quando este passa da sua parte dos bytes, ou quando as colunas que faltam só dão para um
            # grupo cada (assim saem exatamente `groups` grupos)
            if (
                out[-1]
                and len(out) < groups
                and (acc + size / 2 > target * len(out) or len(columns) - i == groups - len(out))
            ):
                out.append([])
            out[-1].append(c)
            acc += size
        return out

    def files_for(self, index: int) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
        left, right = self.files
        return (None if left is None else left[index], None if right is None else right[index])


def _key_lazy(source: Source, keys: Sequence[str]) -> pl.LazyFrame:
    """Só as colunas da chave, sem ler (Categorical e Enum como texto: a ordem e a igualdade são as do texto)."""
    schema = source.lazy.collect_schema()
    exprs = [pl.col(k).cast(pl.String) if isinstance(schema[k], (pl.Categorical, pl.Enum)) else pl.col(k) for k in keys]
    return source.lazy.select(exprs)


def _any_null(keys: Sequence[str]) -> pl.Expr:
    return pl.any_horizontal([pl.col(k).is_null() for k in keys])


def _stats(kl: pl.LazyFrame, keys: Sequence[str]) -> tuple[int, bool]:
    """(linhas, se há chave nula) numa passada pelas colunas da chave, sem guardá-las."""
    row = kl.select(pl.len().alias("n"), _any_null(keys).any().alias("nulls")).collect(engine="streaming").row(0)
    return int(row[0]), bool(row[1])


def _rows_at(kl: pl.LazyFrame, keep: pl.Expr) -> pl.DataFrame:
    """As chaves das linhas escolhidas por ``keep`` (sobre a coluna ``__i``, a posição na ordem de leitura)."""
    out = kl.with_row_index("__i").filter(keep).collect(engine="streaming")
    return out.sort("__i")


def _bytes_per_row(source: Source, sample_rows: int) -> tuple[float, dict[str, float]]:
    """Bytes por linha na memória (todas as colunas, e de cada coluna), por uma amostra do começo da fonte."""
    frame = source.frame if source.frame is not None else source.lazy.head(sample_rows).collect()
    if not frame.height:
        return 0.0, {}
    per_column = {c: frame[c].estimated_size() / frame.height for c in frame.columns}
    return frame.estimated_size() / frame.height, per_column


def _row_group_starts(source: Source) -> list[int]:
    """Posição (na ordem de leitura) do começo de cada row group, somando os arquivos."""
    starts, pos = [], 0
    for path in source.files:
        for n in row_groups(path):
            starts.append(pos)
            pos += n
    return starts


def _largest_row_group(*sources: Source) -> int:
    """O maior row group dos Parquet entre ``sources`` (0 se nenhum é Parquet)."""
    return max((n for src in sources for path in src.files for n in row_groups(path)), default=0)


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


def _aligned_bounds(
    source: Source, kl: pl.LazyFrame, keys: Sequence[str], n: int, rows_per_window: int
) -> list[tuple[Any, ...]] | None:
    """Fronteiras no começo de row groups, se a fonte é Parquet gravado em ordem pela chave (senão None).

    Lê só a chave da primeira linha de cada row group: se essas chaves estão em ordem, cada janela começa num row group
    e o Polars não decodifica nenhum duas vezes. (Uma fonte fora de ordem com os começos em ordem ainda dá janelas
    corretas, só menos parelhas: a ordem afeta o tamanho das janelas, nunca o resultado.)
    """
    if not source.is_parquet:
        return None
    starts = _row_group_starts(source)
    snapped = _snap(list(range(rows_per_window, n, rows_per_window)), starts, rows_per_window)
    if not snapped:
        return None
    firsts = _rows_at(kl, pl.col("__i").is_in(starts))
    keyed = firsts.select(keys)
    if keyed.height != len(starts) or not keyed.equals(keyed.sort(list(keys))):
        return None
    at = {int(i): row for i, row in zip(firsts["__i"], keyed.iter_rows(), strict=True)}
    return [tuple(at[s]) for s in snapped]


def _sampled_bounds(kl: pl.LazyFrame, keys: Sequence[str], n: int, rows_per_window: int) -> list[tuple[Any, ...]]:
    """Fronteiras pelos quantis de uma amostra regular das chaves não nulas (no máximo ~SAMPLE_KEYS chaves)."""
    step = max(1, n // SAMPLE_KEYS)
    sample = _rows_at(kl, (pl.col("__i") % step == 0) & ~_any_null(keys)).select(keys).sort(list(keys))
    positions = [round(c / step) for c in range(rows_per_window, n, rows_per_window)]
    return [tuple(row) for row in sample[[p for p in positions if p < sample.height]].iter_rows()]


def _window_size(
    left: Source, right: Source, bpr: tuple[float, float], budget: int, *, window_rows: int | None, columns: int
) -> tuple[int, int, list[str]]:
    """Linhas por janela (pelo orçamento, ou ``window_rows``), em quantos grupos de colunas ler cada janela, e as
    notas do plano sobre isso."""
    if window_rows is not None:
        if window_rows < 1:
            raise ValueError("window_rows deve ser >= 1")
        return window_rows, 1, []
    per_row = IN_FLIGHT * (bpr[0] + bpr[1]) + 2 * KERNEL_BYTES_PER_ROW
    rows_per_window = max(1, int(budget * SAFETY / per_row))
    floor = _largest_row_group(left, right)
    if rows_per_window >= floor:
        return rows_per_window, 1, []
    # O Polars decodifica o row group inteiro mesmo para uma parte dele: janelas menores que um row group só repetem
    # o trabalho (medido na fase 2: 10M x 300 com janelas de ~52 mil linhas e row groups de 100 mil, cada um lido duas
    # vezes). Então a janela fica com um row group, e a memória cabe lendo as colunas em grupos (o hash de linha
    # continua de um grupo para o outro: mesmo resultado).
    groups = -(-floor // rows_per_window)
    notes = [
        f"o memory_limit pede janelas de {rows_per_window:,} linhas, menos que um row group ({floor:,}): cada janela "
        f"leva um row group, lido em {min(groups, max(columns, 1))} grupos de colunas"
    ]
    if groups > columns:
        notes.append("nem uma coluna por vez cabe no memory_limit: o pico vai passar dele")
    return floor, max(1, min(groups, columns)), notes


FileRange = tuple[Any, Any, bool]  # (menor e maior valor da 1ª coluna da chave, se tem chave nula) de um arquivo


def _file_ranges(source: Source, keys: Sequence[str]) -> list[FileRange] | None:
    """A faixa da 1ª coluna da chave em cada arquivo Parquet da fonte (uma passada só pela chave), ou None se a fonte
    não tem vários arquivos."""
    if len(source.files) < 2:
        return None
    schema = source.lazy.collect_schema()
    first = pl.col(keys[0])
    if isinstance(schema[keys[0]], (pl.Categorical, pl.Enum)):
        first = first.cast(pl.String)
    aggs = [first.min().alias("lo"), first.max().alias("hi"), _any_null(keys).any().alias("nulls")]
    frames = pl.collect_all([source.scan([f]).select(aggs) for f in source.files])
    return [(df["lo"][0], df["hi"][0], bool(df["nulls"][0])) for df in frames]


def _touches(r: FileRange, w: Window) -> bool:
    """O arquivo pode ter linhas da janela? (Só pela 1ª coluna da chave: conservador, nunca deixa uma de fora.)"""
    lo, hi, nulls = r
    if w.with_nulls and nulls:
        return True
    if lo is None:  # só chaves nulas
        return False
    if w.lo is not None and hi < w.lo[0]:
        return False
    return not (w.hi is not None and lo > w.hi[0])


def _window_files(ranges: list[FileRange] | None, windows: list[Window]) -> list[tuple[int, ...]] | None:
    if ranges is None:
        return None
    return [tuple(i for i, r in enumerate(ranges) if _touches(r, w)) for w in windows]


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

    ``window_rows`` (avançado) fixa as linhas por janela em vez de calculá-las pelo orçamento. A memória do
    planejamento não cresce com a tabela: só as contagens, a chave do começo de cada row group e uma amostra limitada.
    """
    if not keys:
        raise ValueError("a chave precisa de ao menos uma coluna")
    budget = parse_memory_limit(memory_limit)
    kl, kr = _key_lazy(left, keys), _key_lazy(right, keys)
    (nl, null_l), (nr, null_r) = _stats(kl, keys), _stats(kr, keys)
    (bl, cl), (br, cr) = _bytes_per_row(left, sample_rows), _bytes_per_row(right, sample_rows)
    bpr = (bl, br)
    column_bytes = {c: cl.get(c, 0.0) + cr.get(c, 0.0) for c in {*cl, *cr} if c not in keys}
    rows_per_window, groups, notes = _window_size(
        left, right, bpr, budget, window_rows=window_rows, columns=len(column_bytes)
    )
    rows = (nl, nr)
    extra = {"column_groups": groups, "column_bytes": column_bytes}
    single = Plan([Window(0, with_nulls=True)], rows_per_window, budget, bpr, rows, "left", notes=notes, **extra)
    if max(rows) <= rows_per_window:
        return single

    # a base é o lado com mais linhas, preferindo um Parquet em ordem sem chave nula (fronteiras nos row groups)
    sides = sorted([("left", left, kl, nl, null_l), ("right", right, kr, nr, null_r)], key=lambda s: -s[3])
    bounds: list[tuple[Any, ...]] | None = None
    base_name, aligned = sides[0][0], False
    for name, source, lazy, n, nulls in sides:
        if n > rows_per_window and not nulls:
            bounds = _aligned_bounds(source, lazy, keys, n, rows_per_window)
            if bounds:
                base_name, aligned = name, True
                break
    if not bounds:
        _, _, lazy, n, _ = sides[0]
        bounds = _sampled_bounds(lazy, keys, n, rows_per_window)
    unique: list[tuple[Any, ...]] = []
    for b in bounds:
        if not unique or b != unique[-1]:  # chave repetida: a mesma fronteira uma vez só
            unique.append(b)
    if not unique:
        return single

    # a cláusula "ou chave nula" impede o Polars de pular row groups pelas estatísticas: só entra se houver chave nula
    windows = [Window(0, None, unique[0], with_nulls=null_l or null_r)]
    windows += [Window(i + 1, unique[i], unique[i + 1]) for i in range(len(unique) - 1)]
    windows.append(Window(len(unique), unique[-1], None))
    if not aligned:
        notes.append("fronteiras por amostra das chaves (fora do começo dos row groups)")
    files = (_window_files(_file_ranges(left, keys), windows), _window_files(_file_ranges(right, keys), windows))
    return Plan(windows, rows_per_window, budget, bpr, rows, base_name, aligned, notes, files, **extra)
