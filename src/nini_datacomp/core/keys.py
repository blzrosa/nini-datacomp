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
from .arrow import Column, FrameColumns
from .blocks import resolve_threads
from .memory import empty

__all__ = ["Alignment", "Strategy", "align_frames", "align_keys", "align_pieces", "descriptor", "key_kind"]

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


def descriptor(cols: Sequence[Column]) -> tuple[np.ndarray, list[np.ndarray]]:
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


def align_keys(
    left: Sequence[Column], right: Sequence[Column], strategy: Strategy = "auto", *, threads: int | None = None
) -> Alignment:
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

    ldesc, lkeep = descriptor(left)
    rdesc, rkeep = descriptor(right)
    out = _align(ldesc, nl, rdesc, nr, strategy, threads=threads)
    del lkeep, rkeep  # vivos até aqui: o kernel leu os endereços de texto deles
    return out


def align_frames(
    left: FrameColumns, right: FrameColumns, strategy: Strategy = "auto", *, threads: int | None = None
) -> Alignment:
    """O mesmo que `align_keys`, com as colunas da chave de cada lado lidas de uma vez (`import_frame`): uma
    exportação do Polars por lado, em vez de uma por coluna."""
    if not len(left) or len(left) != len(right):
        raise ValueError("a chave precisa das mesmas colunas (ao menos uma) nos dois lados")
    if strategy not in _STRATEGIES:
        raise ValueError(f"estratégia desconhecida: {strategy!r} (use 'auto', 'merge' ou 'hash')")
    for c in range(len(left)):
        if not np.array_equal(left.desc[c, :2], right.desc[c, :2]):
            raise ValueError(f"coluna da chave {left.names[c]!r}: tipos diferentes nos dois lados")
    if max(left.length, right.length) > MAX_ROWS:
        raise ValueError(f"janela grande demais: {max(left.length, right.length):,} linhas (máximo {MAX_ROWS:,})")
    out = _align(left.desc, left.length, right.desc, right.length, strategy, threads=threads)
    del left, right  # os buffers (owner) vivos até aqui
    return out


def align_pieces(left: Sequence[Column], right: Sequence[Column]) -> Alignment | None:
    """O merge de uma chave de uma coluna só direto dos pedaços de memória de cada lado (``left``/``right``: os
    pedaços da coluna, em ordem, um `Column` cada), sem juntá-los. None quando não serve (a chave não é int64/int32
    sem nulos, ou algum lado está fora de ordem): aí, `align_keys` com a coluna inteira."""
    if not left or not right:
        return None
    lstarts, ldesc, lkeep = _stack_pieces(left)
    rstarts, rdesc, rkeep = _stack_pieces(right)
    nl, nr = int(lstarts[-1]), int(rstarts[-1])
    cap = max(min(nl, nr), 1)
    li, ri = empty(cap, np.uint32), empty(cap, np.uint32)
    out = np.zeros(_N_STATS, np.int64)
    code = lib().dc_align_pieces(
        len(left), ptr(lstarts), ptr(ldesc), len(right), ptr(rstarts), ptr(rdesc), ptr(li), ptr(ri), ptr(out)
    )
    del lkeep, rkeep
    if code == 1:
        return None
    if code != 0:
        raise ValueError(f"janela grande demais: {max(nl, nr):,} linhas (máximo {MAX_ROWS:,})")
    common, only_l, only_r, dup_l, dup_r, used, pairs = (int(x) for x in out)
    return Alignment(li[:pairs], ri[:pairs], common, only_l, only_r, dup_l, dup_r, _USED[used])


def _stack_pieces(pieces: Sequence[Column]) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    starts = np.zeros(len(pieces) + 1, np.int64)
    np.cumsum([c.length for c in pieces], out=starts[1:])
    desc, keep = descriptor(pieces)
    return starts, desc, keep


def _align(
    ldesc: np.ndarray, nl: int, rdesc: np.ndarray, nr: int, strategy: Strategy, *, threads: int | None
) -> Alignment:
    cap = max(min(nl, nr), 1)
    li, ri = empty(cap, np.uint32), empty(cap, np.uint32)
    out = np.zeros(_N_STATS, np.int64)
    code = lib().dc_align_keys(
        int(ldesc.shape[0]),
        ptr(ldesc),
        nl,
        ptr(rdesc),
        nr,
        _STRATEGIES[strategy],
        ptr(li),
        ptr(ri),
        ptr(out),
        resolve_threads(threads),
    )
    if code != 0:  # pragma: no cover (as validações acima pegam antes)
        raise RuntimeError(f"dc_align_keys falhou (código {code})")
    common, only_l, only_r, dup_l, dup_r, used, pairs = (int(x) for x in out)
    return Alignment(li[:pairs], ri[:pairs], common, only_l, only_r, dup_l, dup_r, _USED[used])
