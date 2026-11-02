"""Alinhamento das chaves de uma janela (kernel ``dc_align_keys``, em ``cpp/keys.cpp``).

Recebe as colunas da chave dos dois lados (`Column`, lidas do Polars sem cópia) e devolve os pares de linhas com a
mesma chave e as contagens. Mesma semântica do datacompolars: nulo casa com nulo (coluna a coluna, numa chave
composta), ``-0.0 == 0.0`` e NaN casa com NaN; com chave repetida em qualquer lado, só as contagens saem (a comparação
aborta).

Estratégia: ``"auto"`` confere se os dois lados vêm ordenados pela chave e, se vêm, faz um merge linear; senão, hash
join. ``"merge"`` cai no hash quando um lado não está ordenado (ou a chave tem reais, que não entram no merge);
``"hash"`` força o hash.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .._native import lib, ptr
from .arrow import Column
from .memory import empty

__all__ = ["Alignment", "Strategy", "align_keys", "key_kind"]

Strategy = Literal["auto", "merge", "hash"]
_STRATEGIES = {"auto": 0, "merge": 1, "hash": 2}
_USED = {1: "merge", 2: "hash"}

# tipos do kernel (enum Kind em cpp/keys.cpp)
SIGNED, UNSIGNED, TEXT, BOOL, BYTES, FLOAT = 1, 2, 3, 4, 5, 6
_DESC = 6  # inteiros por coluna no descritor
_N_STATS = 7
MAX_ROWS = 0xFFFF_FFFF - 1  # índices uint32: uma janela tem menos linhas que isso


def key_kind(fmt: bytes) -> int:
    """Como o kernel lê uma coluna da chave, pelo formato do Arrow."""
    if fmt in (b"c", b"s", b"i", b"l") or fmt.startswith((b"td", b"tt", b"ts", b"tD")):
        return SIGNED  # inteiros com sinal, datas, horas, data e hora, durações
    if fmt in (b"C", b"S", b"I", b"L"):
        return UNSIGNED
    if fmt in (b"vu", b"vz"):
        return TEXT
    if fmt == b"b":
        return BOOL
    if fmt in (b"f", b"g"):
        return FLOAT
    return BYTES  # Decimal e o resto de largura fixa: igualdade byte a byte, sem merge


@dataclass(frozen=True)
class Alignment:
    """Pares de linhas (índices nos lados da janela) com a mesma chave e as contagens."""

    left: np.ndarray  # uint32
    right: np.ndarray  # uint32
    common: int
    only_left: int
    only_right: int
    dup_left: int  # linhas excedentes por chave repetida (0 = chave única)
    dup_right: int
    strategy: str  # a usada: "merge" ou "hash"

    @property
    def has_duplicates(self) -> bool:
        return bool(self.dup_left or self.dup_right)


def _descriptor(cols: Sequence[Column]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Os descritores das colunas (``k x 6`` int64) e os vetores de endereços de texto, que têm de ficar vivos."""
    desc = np.zeros((len(cols), _DESC), np.int64)
    keep: list[np.ndarray] = []
    for c, col in enumerate(cols):
        texts = np.array(col.texts or (0,), np.uint64)
        keep.append(texts)
        desc[c] = (
            key_kind(col.format),
            col.width,
            col.data or 0,
            col.validity or 0,
            col.bit_offset,
            texts.ctypes.data if col.texts else 0,
        )
    return desc, keep


def align_keys(left: Sequence[Column], right: Sequence[Column], strategy: Strategy = "auto") -> Alignment:
    """Pares e contagens das chaves ``left`` x ``right`` (mesmas colunas, na mesma ordem, dos mesmos tipos)."""
    if not left or len(left) != len(right):
        raise ValueError("a chave precisa das mesmas colunas (ao menos uma) nos dois lados")
    if strategy not in _STRATEGIES:
        raise ValueError(f"estratégia desconhecida: {strategy!r} (use 'auto', 'merge' ou 'hash')")
    for a, b in zip(left, right, strict=True):
        if key_kind(a.format) != key_kind(b.format) or a.width != b.width:
            raise ValueError(f"coluna da chave {a.name!r}: tipos diferentes nos dois lados ({a.dtype} x {b.dtype})")
    nl, nr = left[0].length, right[0].length
    if any(c.length != nl for c in left) or any(c.length != nr for c in right):
        raise ValueError("as colunas da chave de um lado têm tamanhos diferentes")
    if max(nl, nr) > MAX_ROWS:
        raise ValueError(f"janela grande demais: {max(nl, nr):,} linhas (máximo {MAX_ROWS:,})")

    ldesc, lkeep = _descriptor(left)
    rdesc, rkeep = _descriptor(right)
    cap = max(min(nl, nr), 1)
    li, ri = empty(cap, np.uint32), empty(cap, np.uint32)
    out = np.zeros(_N_STATS, np.int64)
    code = lib().dc_align_keys(
        len(left), ptr(ldesc), nl, ptr(rdesc), nr, _STRATEGIES[strategy], ptr(li), ptr(ri), ptr(out)
    )
    del lkeep, rkeep  # vivos até aqui: o kernel leu os endereços de texto deles
    if code != 0:  # pragma: no cover (as validações acima pegam antes)
        raise RuntimeError(f"dc_align_keys falhou (código {code})")
    common, only_l, only_r, dup_l, dup_r, used, pairs = (int(x) for x in out)
    return Alignment(li[:pairs], ri[:pairs], common, only_l, only_r, dup_l, dup_r, _USED[used])
