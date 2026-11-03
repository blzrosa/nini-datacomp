"""Comparadores próprios por coluna (``compare(..., comparators={"coluna": ...})``).

Cada comparador vira um `Comparator`: dado o valor dos dois lados nos pares de uma janela, diz em quais pares a
coluna diverge. A regra de nulos é a do datacompolars e vale para qualquer comparador: nulo/nulo é igual, nulo de um
lado só é diferente, e um resultado nulo do comparador conta como diferente.

Formas aceitas:

* uma função ``f(esquerda, direita) -> pl.Expr`` (booleana, True = iguais), sobre duas expressões do Polars;
* uma expressão do `dc.expr` (``left``/``right``), que roda no C++, sem aviso;
* um comparador do datacompolars (um objeto com ``compare(col_l, col_r, **kwargs) -> pl.Expr``), que recebe
  ``abs_tol``, ``rel_tol``, ``ignore_spaces``, ``ignore_case`` e ``is_float`` como lá.

A função e o comparador do datacompolars rodam no Polars, só nos pares da coluna, e avisam com
`PerformanceWarning`: são mais lentos que o caminho dos kernels C++.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
import polars as pl

__all__ = ["Comparator", "PolarsComparator", "as_comparator", "is_comparator"]

Rule = Callable[[pl.Expr, pl.Expr], pl.Expr]
_L, _R = "__dc_l", "__dc_r"


class Comparator(Protocol):
    """Diz em que pares uma coluna diverge."""

    def differs(self, left: pl.Series, right: pl.Series, li: np.ndarray, ri: np.ndarray) -> np.ndarray:
        """1 (uint8) nos pares ``(li[i], ri[i])`` em que os valores divergem, 0 nos outros."""
        ...


@dataclass(frozen=True)
class PolarsComparator:
    """Um comparador que roda no Polars: ``rule(esquerda, direita)`` dá True onde os valores são iguais."""

    rule: Rule

    def differs(self, left: pl.Series, right: pl.Series, li: np.ndarray, ri: np.ndarray) -> np.ndarray:
        if not li.size:
            return np.zeros(0, np.uint8)
        frame = pl.DataFrame([left.gather(pl.Series(li)).alias(_L), right.gather(pl.Series(ri)).alias(_R)])
        a, b = pl.col(_L), pl.col(_R)
        equal = (a.is_null() & b.is_null()) | (a.is_not_null() & b.is_not_null() & self.rule(a, b).fill_null(False))
        out = frame.select(equal.not_().alias("d")).to_series()
        if out.dtype != pl.Boolean or out.len() != li.size:
            raise TypeError("o comparador precisa devolver uma expressão booleana com uma linha por par")
        return out.to_numpy().astype(np.uint8)


def is_comparator(obj: object) -> bool:
    """Uma das formas aceitas (dc.expr, função de duas expressões, ou objeto com ``compare``)."""
    return callable(getattr(obj, "comparator", None)) or callable(getattr(obj, "compare", None)) or callable(obj)


def as_comparator(
    obj: Any,
    dtype: pl.DataType,
    *,
    abs_tol: float = 0.0,
    rel_tol: float = 0.0,
    ignore_spaces: bool = False,
    ignore_case: bool = False,
) -> Comparator:
    """O `Comparator` de uma das formas aceitas, para uma coluna do tipo ``dtype``."""
    native = getattr(obj, "comparator", None)  # dc.expr: já sabe se comparar (no C++)
    if callable(native):
        return cast("Comparator", native(dtype))
    compare = getattr(obj, "compare", None)
    if callable(compare):  # comparador do datacompolars: trabalha com nomes de coluna
        kwargs = {
            "abs_tol": abs_tol,
            "rel_tol": rel_tol,
            "ignore_spaces": ignore_spaces,
            "ignore_case": ignore_case,
            "is_float": dtype.is_float(),
        }
        return PolarsComparator(lambda a, b: cast("pl.Expr", compare(_L, _R, **kwargs)))
    if callable(obj):
        return PolarsComparator(cast("Rule", obj))
    raise TypeError(f"comparador não reconhecido: {obj!r} (use f(esquerda, direita) -> pl.Expr ou dc.expr)")
