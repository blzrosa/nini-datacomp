"""Memória dos buffers grandes (pares de linhas, hashes, bitmaps): vem do pool do C++ (`cpp/memory.cpp`).

Copiado do nini-datagen 0.3.0, sem o modo ``arrow`` (aqui não há pool do pyarrow para comparar). Com ``np.empty`` (o
malloc do sistema), cada buffer grande pede memória nova ao sistema operacional, que entrega as páginas uma a uma na
primeira escrita, e devolve tudo quando o buffer é liberado (no Windows, ~54 ns/linha na primeira escrita contra ~15
na segunda, medido no nini-datagen). O pool guarda a memória liberada (até ``pool_limit``, 1 GiB por padrão): a
próxima janela da comparação, que pede os mesmos tamanhos, não paga nenhum dos dois custos.

O array devolvido é um array NumPy comum sobre o bloco do pool; o bloco volta ao pool quando o array (e tudo que o
segura: fatias, uma Series do Polars feita dele) deixa de existir. Buffers pequenos continuam com ``np.empty``.

``NINI_DATACOMP_POOL=0`` desliga o pool (tudo com ``np.empty``), para comparar.
"""

from __future__ import annotations

import ctypes
import math
import os
import weakref
from typing import Any

import numpy as np

from .._native import lib

__all__ = ["POOL_ENV", "POOL_MIN_BYTES", "empty", "pool_limit", "pool_stats", "release_memory"]

POOL_ENV = "NINI_DATACOMP_POOL"
POOL_MIN_BYTES = 1 << 20  # abaixo disso o malloc do sistema já reaproveita a memória
_GRAIN = 1 << 16  # o pool arredonda para múltiplos de 64 KiB
_Chunk = ctypes.c_uint8 * _GRAIN


def _native_block(nbytes: int) -> ctypes.Array[Any]:
    """Bloco do pool como array do ctypes, que devolve o bloco ao pool quando é coletado."""
    dc = lib()
    addr = dc.dc_pool_alloc(nbytes)
    if not addr:
        raise MemoryError(f"sem memória para {nbytes:,} bytes")
    # array de pedaços de 64 KiB: poucos tipos do ctypes distintos (o ctypes guarda um tipo por tamanho de array)
    block = (_Chunk * -(-nbytes // _GRAIN)).from_address(addr)
    weakref.finalize(block, dc.dc_pool_free, addr)
    return block


def empty(shape: int | tuple[int, ...], dtype: Any) -> np.ndarray:
    """Como ``np.empty(shape, dtype)`` (C-contíguo, conteúdo indefinido), com a memória do pool."""
    dt = np.dtype(dtype)
    dims = (shape,) if isinstance(shape, int) else tuple(shape)
    count = math.prod(dims)
    nbytes = count * dt.itemsize
    if nbytes < POOL_MIN_BYTES or os.environ.get(POOL_ENV, "1") == "0":
        return np.empty(dims, dt)
    # a memoryview segura o bloco; o array segura a memoryview
    return np.frombuffer(memoryview(_native_block(nbytes)).cast("B"), dt, count).reshape(dims)


def pool_stats() -> dict[str, int]:
    """Bytes em uso e guardados no pool, e quantas alocações ele atendeu (``hits``) ou pediu ao sistema."""
    out = (ctypes.c_int64 * 4)()
    lib().dc_pool_stats(out)
    return {"in_use": out[0], "cached": out[1], "hits": out[2], "misses": out[3]}


def pool_limit(nbytes: int | None = None) -> int:
    """Limite de bytes guardados no pool depois de liberados (padrão 1 GiB); ``None`` só consulta; 0 desliga."""
    return int(lib().dc_pool_limit(-1 if nbytes is None else max(0, int(nbytes))))


def release_memory() -> None:
    """Devolve ao sistema operacional a memória guardada no pool (os buffers ainda em uso não são afetados)."""
    lib().dc_pool_trim()
