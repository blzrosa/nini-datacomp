"""dc.expr: comparadores próprios que rodam nos kernels C++, sem passar pelo Polars.

Uma expressão booleana sobre ``left`` e ``right`` (o valor da coluna em cada lado), escrita como no Polars::

    import nini_datacomp as dc
    from nini_datacomp.expr import left, right

    dc.compare(a, b, on="id", comparators={
        "nome": left.strip().lower() == right.strip().lower(),
        "valor": (left - right).abs() <= 0.01 * right.abs(),
        "codigo": left.remove_whitespace() == right.remove_whitespace(),
    })

O compilador escolhe as instruções pelo tipo da coluna, e o C++ (cpp/expr.cpp) roda o programa por blocos de
pares. Regra de nulos: nulo/nulo é igual, nulo de um lado só é diferente; a expressão só vê pares sem nulos.

O que há: ``+ - * /``, ``-x``, ``abs()``; ``== != < <= > >=``; ``& | ~``; no texto, ``lower()``, ``upper()``,
``strip()``, ``remove_whitespace()`` e ``len()``; números, booleanos e textos como constantes (ou ``lit(v)``).
Colunas: inteiros (até 64 bits com sinal, sem UInt64), reais, booleanos e texto. Inteiros fazem conta em 64 bits;
``/`` dá real; reais são comparados como no Polars (NaN == NaN, -0.0 == 0.0); Float32 faz conta em double. Texto:
``lower``/``upper`` com o mapeamento completo do Unicode ("ß".upper() == "SS") e ``strip`` com os espaços do
Unicode, como o Polars. Para o resto (datas, listas...), uma função do Polars como comparador.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from ._native import lib, ptr
from .core.arrow import import_column
from .core.blocks import resolve_threads
from .core.keys import descriptor

__all__ = ["Expr", "NativeComparator", "left", "lit", "right"]

# tipos (os mesmos do cpp/expr.cpp)
_I, _F, _B, _S = 0, 1, 2, 3
_NAMES = {_I: "inteiro", _F: "real", _B: "booleano", _S: "texto"}
# instruções
_LOAD_L, _LOAD_R, _CONST_I, _CONST_F, _CONST_B, _CONST_S, _TO_F = 1, 2, 3, 4, 5, 6, 7
_ARITH = {"add": (10, 13), "sub": (11, 14), "mul": (12, 15)}  # (inteiro, real)
_DIV_F, _NEG, _ABS = 16, (17, 18), (19, 20)
_CMP = {"eq": 0, "ne": 1, "lt": 2, "le": 3, "gt": 4, "ge": 5}
_CMP_BASE = {_I: 30, _F: 40, _S: 50}
_EQ_B, _NE_B, _AND, _OR, _NOT = 60, 61, 70, 71, 72
_TEXT = {"lower": 80, "upper": 81, "strip": 82, "remove_whitespace": 83}
_LEN = 84
_SYMBOL = {"add": "+", "sub": "-", "mul": "*", "div": "/", "eq": "==", "ne": "!=", "lt": "<", "le": "<=",
           "gt": ">", "ge": ">=", "and": "&", "or": "|"}  # fmt: skip
_I64 = (-(2**63), 2**63 - 1)


def _wrap(value: Any) -> Expr:
    if isinstance(value, Expr):
        return value
    return lit(value)


class Expr:
    """Um nó da expressão. Crie a partir de ``left``, ``right`` e ``lit``; combine com operadores."""

    __slots__ = ("args", "op", "value")
    __hash__ = None  # type: ignore[assignment]  # == monta uma expressão, como no Polars

    def __init__(self, op: str, *args: Expr, value: Any = None) -> None:
        self.op, self.args, self.value = op, args, value

    # aritmética
    def __add__(self, other: Any) -> Expr:
        return Expr("add", self, _wrap(other))

    def __radd__(self, other: Any) -> Expr:
        return Expr("add", _wrap(other), self)

    def __sub__(self, other: Any) -> Expr:
        return Expr("sub", self, _wrap(other))

    def __rsub__(self, other: Any) -> Expr:
        return Expr("sub", _wrap(other), self)

    def __mul__(self, other: Any) -> Expr:
        return Expr("mul", self, _wrap(other))

    def __rmul__(self, other: Any) -> Expr:
        return Expr("mul", _wrap(other), self)

    def __truediv__(self, other: Any) -> Expr:
        return Expr("div", self, _wrap(other))

    def __rtruediv__(self, other: Any) -> Expr:
        return Expr("div", _wrap(other), self)

    def __neg__(self) -> Expr:
        return Expr("neg", self)

    def __abs__(self) -> Expr:
        return Expr("abs", self)

    # comparações
    def __eq__(self, other: object) -> Expr:  # pyright: ignore[reportIncompatibleMethodOverride]
        return Expr("eq", self, _wrap(other))

    def __ne__(self, other: object) -> Expr:  # pyright: ignore[reportIncompatibleMethodOverride]
        return Expr("ne", self, _wrap(other))

    def __lt__(self, other: Any) -> Expr:
        return Expr("lt", self, _wrap(other))

    def __le__(self, other: Any) -> Expr:
        return Expr("le", self, _wrap(other))

    def __gt__(self, other: Any) -> Expr:
        return Expr("gt", self, _wrap(other))

    def __ge__(self, other: Any) -> Expr:
        return Expr("ge", self, _wrap(other))

    # lógica
    def __and__(self, other: Any) -> Expr:
        return Expr("and", self, _wrap(other))

    def __rand__(self, other: Any) -> Expr:
        return Expr("and", _wrap(other), self)

    def __or__(self, other: Any) -> Expr:
        return Expr("or", self, _wrap(other))

    def __ror__(self, other: Any) -> Expr:
        return Expr("or", _wrap(other), self)

    def __invert__(self) -> Expr:
        return Expr("not", self)

    def __bool__(self) -> bool:
        raise TypeError("uma expressão do dc.expr não é verdadeira nem falsa: combine com &, | e ~ (não and/or/not)")

    # métodos
    def abs(self) -> Expr:
        """Valor absoluto."""
        return Expr("abs", self)

    def lower(self) -> Expr:
        """Texto em minúsculas (Unicode)."""
        return Expr("lower", self)

    def upper(self) -> Expr:
        """Texto em maiúsculas (Unicode)."""
        return Expr("upper", self)

    def strip(self) -> Expr:
        """Texto sem os espaços em branco do começo e do fim."""
        return Expr("strip", self)

    def remove_whitespace(self) -> Expr:
        """Texto sem nenhum espaço em branco (como o ``ignore_whitespace``)."""
        return Expr("remove_whitespace", self)

    def len(self) -> Expr:
        """Número de caracteres do texto."""
        return Expr("len", self)

    def __repr__(self) -> str:
        if self.op in ("left", "right"):
            return self.op
        if self.op == "lit":
            return repr(self.value)
        if self.op in _SYMBOL:
            return f"({self.args[0]!r} {_SYMBOL[self.op]} {self.args[1]!r})"
        if self.op == "neg":
            return f"-{self.args[0]!r}"
        if self.op == "not":
            return f"~{self.args[0]!r}"
        return f"{self.args[0]!r}.{self.op}()"

    def comparator(self, dtype: pl.DataType) -> NativeComparator:
        """O comparador compilado para uma coluna do tipo ``dtype`` (o uso normal é ``comparators={...}``)."""
        return _Compiler(dtype).compile(self)


left = Expr("left")
right = Expr("right")


def lit(value: Any) -> Expr:
    """Uma constante: número, booleano ou texto."""
    if not isinstance(value, (bool, int, float, str)):
        raise TypeError(f"dc.expr: constante de tipo não suportado: {type(value).__name__} (número, bool ou texto)")
    if isinstance(value, int) and not isinstance(value, bool) and not _I64[0] <= value <= _I64[1]:
        raise ValueError(f"dc.expr: inteiro fora de 64 bits: {value}")
    return Expr("lit", value=value)


def _column_type(dtype: pl.DataType) -> int:
    if dtype == pl.Boolean:
        return _B
    if dtype == pl.String:
        return _S
    if dtype.is_float():
        return _F
    if dtype.is_integer() and dtype != pl.UInt64:
        return _I
    raise TypeError(
        f"dc.expr não lê colunas {dtype} (lê inteiros sem UInt64, reais, booleanos e texto): use uma função do "
        "Polars como comparador, ou converta o tipo antes (cast)"
    )


class _Compiler:
    def __init__(self, dtype: pl.DataType) -> None:
        self.dtype = dtype
        self.col = _column_type(dtype)
        self.code: list[int] = []
        self.floats: list[float] = []
        self.ints: list[int] = []
        self.texts: list[bytes] = []

    def compile(self, expr: Expr) -> NativeComparator:
        if self.type_of(expr) != _B:
            raise TypeError(f"dc.expr: o comparador precisa dar um booleano (iguais ou não), não {self._name(expr)}")
        self.emit(expr, _B)
        offsets = np.zeros(len(self.texts) + 1, np.int64)
        offsets[1:] = np.cumsum([len(t) for t in self.texts]) if self.texts else []
        return NativeComparator(
            repr(expr),
            np.array(self.code, np.int64),
            np.array(self.floats or [0.0], np.float64),
            np.array(self.ints or [0], np.int64),
            np.frombuffer(b"".join(self.texts) or b"\0", np.uint8).copy(),
            offsets,
        )

    def _name(self, e: Expr) -> str:
        return _NAMES[self.type_of(e)]

    def _fail(self, e: Expr, why: str) -> TypeError:
        return TypeError(f"dc.expr: {why} em {e!r} (coluna {self.dtype})")

    def type_of(self, e: Expr) -> int:  # noqa: PLR0911, PLR0912 - um caso por operação
        op = e.op
        if op in ("left", "right"):
            return self.col
        if op == "lit":
            v = e.value
            return _B if isinstance(v, bool) else _I if isinstance(v, int) else _F if isinstance(v, float) else _S
        types = [self.type_of(a) for a in e.args]
        if op in ("add", "sub", "mul", "div"):
            if any(t not in (_I, _F) for t in types):
                raise self._fail(e, "conta com algo que não é número")
            return _F if op == "div" or _F in types else _I
        if op in ("neg", "abs"):
            if types[0] not in (_I, _F):
                raise self._fail(e, f"{op} de algo que não é número")
            return types[0]
        if op in _CMP:
            a, b = types
            numeric = a in (_I, _F) and b in (_I, _F)
            if not (numeric or a == b):
                raise self._fail(e, f"comparação de {_NAMES[a]} com {_NAMES[b]}")
            if a == _B and op not in ("eq", "ne"):
                raise self._fail(e, "ordem entre booleanos")
            return _B
        if op in ("and", "or", "not"):
            if any(t != _B for t in types):
                raise self._fail(e, f"{_SYMBOL.get(op, '~')} com algo que não é booleano")
            return _B
        if op in _TEXT or op == "len":
            if types[0] != _S:
                raise self._fail(e, f"{op}() de algo que não é texto")
            return _I if op == "len" else _S
        raise self._fail(e, f"operação desconhecida {op!r}")  # pragma: no cover

    def emit(self, e: Expr, want: int) -> None:
        """Emite ``e`` e converte o resultado para ``want`` (só inteiro -> real)."""
        got = self.type_of(e)
        self._emit(e)
        if want == _F and got == _I:
            self.code += [_TO_F, 0]

    def _emit(self, e: Expr) -> None:
        op, args = e.op, e.args
        if op in ("left", "right"):
            self.code += [_LOAD_L if op == "left" else _LOAD_R, self.col]
        elif op == "lit":
            self._const(e.value)
        elif op in _ARITH or op == "div":
            t = _F if op == "div" else self.type_of(e)
            self.emit(args[0], t)
            self.emit(args[1], t)
            self.code += [_DIV_F if op == "div" else _ARITH[op][t == _F], 0]
        elif op in ("neg", "abs"):
            self._emit(args[0])
            self.code += [(_NEG if op == "neg" else _ABS)[self.type_of(args[0]) == _F], 0]
        elif op in _CMP:
            a, b = (self.type_of(x) for x in args)
            if a == _B:
                self._emit(args[0])
                self._emit(args[1])
                self.code += [_EQ_B if op == "eq" else _NE_B, 0]
                return
            t = _S if a == _S else _F if _F in (a, b) else _I
            self.emit(args[0], t)
            self.emit(args[1], t)
            self.code += [_CMP_BASE[t] + _CMP[op], 0]
        elif op in ("and", "or"):
            self._emit(args[0])
            self._emit(args[1])
            self.code += [_AND if op == "and" else _OR, 0]
        elif op == "not":
            self._emit(args[0])
            self.code += [_NOT, 0]
        else:  # texto
            self._emit(args[0])
            self.code += [_LEN if op == "len" else _TEXT[op], 0]

    def _const(self, v: Any) -> None:
        if isinstance(v, bool):
            self.code += [_CONST_B, int(v)]
        elif isinstance(v, int):
            self.ints.append(v)
            self.code += [_CONST_I, len(self.ints) - 1]
        elif isinstance(v, float):
            if math.isnan(v):
                v = math.nan
            self.floats.append(v)
            self.code += [_CONST_F, len(self.floats) - 1]
        else:
            self.texts.append(v.encode("utf-8"))
            self.code += [_CONST_S, len(self.texts) - 1]


@dataclass(frozen=True)
class NativeComparator:
    """Um comparador do dc.expr compilado: roda no C++ (``dc_expr_differs``)."""

    text: str
    code: np.ndarray
    floats: np.ndarray
    ints: np.ndarray
    texts: np.ndarray
    offsets: np.ndarray
    threads: int | None = None

    def differs(self, left: pl.Series, right: pl.Series, li: np.ndarray, ri: np.ndarray) -> np.ndarray:
        n = int(li.size)
        out = np.zeros(n, np.uint8)
        if n == 0:
            return out
        lcol, rcol = import_column(left), import_column(right)
        ldesc, lkeep = descriptor([lcol])
        rdesc, rkeep = descriptor([rcol])
        lrows, rrows = np.ascontiguousarray(li, np.uint32), np.ascontiguousarray(ri, np.uint32)
        got = int(
            lib().dc_expr_differs(
                ptr(ldesc), ptr(rdesc), ptr(lrows), ptr(rrows), n, ptr(self.code), self.code.size,
                ptr(self.floats), ptr(self.ints), ptr(self.texts), ptr(self.offsets), ptr(out),
                resolve_threads(self.threads),
            )
        )  # fmt: skip
        del lkeep, rkeep, lcol, rcol
        if got == -1:
            raise TypeError(f"dc.expr {self.text}: a coluna não é do tipo para o qual o comparador foi compilado")
        if got < 0:  # pragma: no cover - o compilador só gera programas válidos
            raise RuntimeError(f"dc.expr {self.text}: programa inválido")
        return out
