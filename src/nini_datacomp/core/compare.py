"""Comparação de uma coluna, par a par (kernel ``dc_compare_column``, em ``cpp/compare.cpp``).

Igualdade exata (o ``eq_missing`` do Polars, com NaN = NaN) ou com tolerância nas colunas numéricas:
``|l - r| <= abs_tol + rel_tol * |r|``, com a aritmética do Polars (inteiros no próprio tipo, Float32 em float,
Decimal pelo inteiro de 128 bits). Nulo dos dois lados = igual; de um lado só = diferente. É a semântica do
datacompolars 0.4.4.

Os pares são os de `align_keys` (linha da esquerda, linha da direita); ``subset`` restringe a alguns deles (as
posições nos vetores de pares), como os que o hash de linha marcou como divergentes. Roda em blocos, em paralelo.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from .._native import lib, ptr
from .arrow import Column, FrameColumns, import_column, import_frame
from .blocks import Block, map_blocks, split
from .keys import descriptor, key_kind

__all__ = ["Tolerance", "compare_column", "compare_frames", "compare_pairs", "tolerance_for"]


@dataclass(frozen=True)
class Tolerance:
    abs_tol: float = 0.0
    rel_tol: float = 0.0

    @property
    def active(self) -> bool:
        return self.abs_tol > 0 or self.rel_tol > 0


def tolerance_for(dtype: pl.DataType, abs_tol: float, rel_tol: float) -> Tolerance | None:
    """A tolerância de uma coluna do tipo ``dtype`` (None: comparação exata). Como no datacompolars, só os tipos
    numéricos (inteiros, reais, Decimal) têm tolerância; datas, horas, texto e booleanos são sempre exatos."""
    if abs_tol < 0 or rel_tol < 0:
        raise ValueError("abs_tol e rel_tol não podem ser negativos")
    tol = Tolerance(float(abs_tol), float(rel_tol))
    return tol if tol.active and dtype.is_numeric() else None


def _scale(dtype: pl.DataType) -> float:
    scale = getattr(dtype, "scale", None)
    return float(10 ** int(scale)) if isinstance(dtype, pl.Decimal) and scale else 1.0


def compare_column(
    left: Column,
    right: Column,
    left_rows: np.ndarray,
    right_rows: np.ndarray,
    *,
    subset: np.ndarray | None = None,
    tolerance: Tolerance | None = None,
    differs: np.ndarray | None = None,
    threads: int | None = None,
) -> tuple[np.ndarray, int]:
    """``(differs, quantos)``: ``differs[p]`` = 1 nos pares ``(left_rows[p], right_rows[p])`` em que a coluna diverge.

    ``differs`` (uint8, um por par) pode vir de fora para acumular várias colunas: só recebe 1s, nunca apaga.
    """
    if key_kind(left.format) != key_kind(right.format) or left.width != right.width:
        raise ValueError(f"coluna {left.name!r}: tipos diferentes nos dois lados ({left.dtype} x {right.dtype})")
    li, ri = np.ascontiguousarray(left_rows, np.uint32), np.ascontiguousarray(right_rows, np.uint32)
    if li.size != ri.size:
        raise ValueError("left_rows e right_rows precisam ter o mesmo tamanho")
    npairs = int(li.size)
    if differs is None:
        differs = np.zeros(npairs, np.uint8)
    elif differs.dtype != np.uint8 or differs.shape != (npairs,) or not differs.flags.c_contiguous:
        raise ValueError(f"differs precisa ser um vetor uint8 contíguo de {npairs:,} posições")
    sub = None if subset is None else np.ascontiguousarray(subset, np.uint32)
    todo = npairs if sub is None else int(sub.size)
    if todo == 0:
        return differs, 0
    if npairs and (int(li.max()) >= left.length or int(ri.max()) >= right.length):
        raise ValueError("índice de linha fora da coluna")
    if sub is not None and int(sub.max()) >= npairs:
        raise ValueError("subset aponta para um par que não existe")
    ldesc, lkeep = descriptor([left])
    rdesc, rkeep = descriptor([right])
    tol = tolerance if tolerance is not None and tolerance.active else None
    scale = _scale(left.dtype)
    dc = lib()
    sub_ptr = 0 if sub is None else ptr(sub)

    fixed = (ptr(ldesc), ptr(rdesc), ptr(li), ptr(ri), sub_ptr)
    params = (int(tol is not None), tol.abs_tol if tol else 0.0, tol.rel_tol if tol else 0.0, scale, ptr(differs))

    def run(b: Block) -> int:
        return int(dc.dc_compare_column(*fixed, b.start, b.start + b.rows, *params))

    count = sum(map_blocks(run, split(todo), threads))
    del lkeep, rkeep
    return differs, count


def compare_frames(left: pl.DataFrame, right: pl.DataFrame, *, threads: int | None = None) -> np.ndarray:
    """Igualdade exata, coluna a coluna, da linha i de ``left`` com a linha i de ``right`` (as linhas dos pares já
    reunidas, mesmas colunas e tipos nos dois). Devolve ``differs`` (uint8, colunas x linhas): 1 onde diverge.

    Os dois lados numa exportação só do Polars cada e uma chamada ao C++ (``dc_compare_rows``), em vez de uma
    `compare_column` por coluna; se um lado não exporta assim, coluna a coluna.
    """
    k, n = left.width, left.height
    if right.width != k or right.height != n:
        raise ValueError("left e right precisam das mesmas colunas e linhas")
    differs = np.zeros((k, n), np.uint8)
    if k == 0 or n == 0:
        return differs
    lf, rf = import_frame(left), import_frame(right)
    if lf is not None and rf is not None:
        if not np.array_equal(lf.desc[:, :2], rf.desc[:, :2]):  # tipo e largura de cada coluna
            raise ValueError("left e right têm colunas de tipos diferentes")
        dc = lib()
        fixed = (k, ptr(lf.desc), ptr(rf.desc), n)

        def run(b: Block) -> int:
            return int(dc.dc_compare_rows(*fixed, b.start, b.start + b.rows, ptr(differs)))

        map_blocks(run, split(n), threads)
        del lf, rf
        return differs
    rows = np.arange(n, dtype=np.uint32)
    for j in range(k):
        compare_column(
            import_column(left.to_series(j)), import_column(right.to_series(j)), rows, rows, differs=differs[j],
            threads=threads,
        )  # fmt: skip
    return differs


Pieces = Sequence[tuple[int, Sequence[Column] | FrameColumns]]  # (primeira linha, colunas do pedaço), como no hash


def _stack(pieces: Pieces, k: int) -> tuple[np.ndarray, np.ndarray, list[object]]:
    """(primeira linha de cada pedaço mais o total, descritores de todos os pedaços em sequência, o que os mantém
    vivos)."""
    starts: list[int] = []
    descs: list[np.ndarray] = [np.zeros((0, 6), np.int64)]
    keep: list[object] = []
    end = 0
    for start, part in pieces:
        if isinstance(part, FrameColumns):
            desc, rows = part.desc, part.length
            keep.append(part)
        else:
            desc, texts = descriptor(part)
            keep.append(texts)
            rows = part[0].length if part else 0
        if desc.shape[0] != k:
            raise ValueError(f"um pedaço tem {desc.shape[0]} colunas, não {k}")
        starts.append(start)
        descs.append(desc)
        end = start + rows
    starts.append(end)
    return np.asarray(starts, np.int64), np.ascontiguousarray(np.concatenate(descs), np.int64), keep


def compare_pairs(
    left: Pieces,
    right: Pieces,
    left_rows: np.ndarray,
    right_rows: np.ndarray,
    *,
    columns: int,
    subset: np.ndarray | None = None,
    threads: int | None = None,
) -> np.ndarray:
    """Igualdade exata das ``columns`` colunas nos pares ``(left_rows[p], right_rows[p])`` (só os de ``subset``, se
    vier), lendo as colunas direto dos pedaços da janela (os mesmos que o hash de linha leu: `_hash_side`), sem reunir
    as linhas no Polars antes. Devolve ``differs`` (uint8, colunas x pares de ``subset``): 1 onde diverge."""
    li, ri = np.ascontiguousarray(left_rows, np.uint32), np.ascontiguousarray(right_rows, np.uint32)
    sub = None if subset is None else np.ascontiguousarray(subset, np.uint32)
    n = int(li.size if sub is None else sub.size)
    differs = np.zeros((columns, n), np.uint8)
    if columns == 0 or n == 0:
        return differs
    lstarts, ldesc, lkeep = _stack(left, columns)
    rstarts, rdesc, rkeep = _stack(right, columns)
    if not np.array_equal(ldesc[:columns, :2], rdesc[:columns, :2]):  # tipo e largura de cada coluna
        raise ValueError("left e right têm colunas de tipos diferentes")
    p = li if sub is None else li[sub]
    q = ri if sub is None else ri[sub]
    if int(p.max()) >= int(lstarts[-1]) or int(q.max()) >= int(rstarts[-1]):
        raise ValueError("índice de linha fora da janela")
    dc = lib()
    fixed = (
        columns, len(left), ptr(lstarts), ptr(ldesc), len(right), ptr(rstarts), ptr(rdesc), ptr(li), ptr(ri),
        0 if sub is None else ptr(sub), n,
    )  # fmt: skip

    def run(b: Block) -> int:
        return int(dc.dc_compare_pairs(*fixed, b.start, b.start + b.rows, ptr(differs)))

    map_blocks(run, split(n), threads)
    del lkeep, rkeep
    return differs
