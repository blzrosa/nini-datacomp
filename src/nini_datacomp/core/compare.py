"""Comparação de uma coluna, par a par (kernel ``dc_compare_column``, em ``cpp/compare.cpp``).

Igualdade exata (o ``eq_missing`` do Polars, com NaN = NaN) ou com tolerância nas colunas numéricas:
``|l - r| <= abs_tol + rel_tol * |r|``, com a aritmética do Polars (inteiros no próprio tipo, Float32 em float,
Decimal pelo inteiro de 128 bits). Nulo dos dois lados = igual; de um lado só = diferente. É a semântica do
datacompolars 0.4.4.

Os pares são os de `align_keys` (linha da esquerda, linha da direita); ``subset`` restringe a alguns deles (as
posições nos vetores de pares), como os que o hash de linha marcou como divergentes. Roda em blocos, em paralelo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .._native import lib, ptr
from .arrow import Column
from .blocks import Block, map_blocks, split
from .keys import descriptor, key_kind

__all__ = ["Tolerance", "compare_column", "tolerance_for"]


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
