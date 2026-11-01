"""Divisão em blocos de linhas e execução paralela (copiado do nini-datagen 0.3.0).

Os kernels trabalham em blocos de ``BLOCK_ROWS`` linhas: cada thread pega um bloco e passa por todas as colunas dele
(o hash de linha, por exemplo, combina as colunas de um bloco enquanto elas ainda estão no cache). Aqui o tamanho do
bloco não muda nenhum resultado, só o desempenho; fica fixo para os números de benchmark serem comparáveis.

As threads ficam num pool por número de threads, criado na primeira vez e reaproveitado (criar 16 threads custava
~0,8 ms no nini-datagen, e a comparação por janelas chama o pool janela após janela). Dentro de uma thread do pool,
`parallel_map` roda em série (um pool esperando por ele mesmo travaria).
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TypeVar

__all__ = ["BLOCK_ROWS", "THREADS_ENV", "Block", "map_blocks", "parallel_map", "resolve_threads", "split"]

BLOCK_ROWS = 65_536
THREADS_ENV = "NINI_DATACOMP_THREADS"

T = TypeVar("T")
Item = TypeVar("Item")


@dataclass(frozen=True, slots=True)
class Block:
    index: int  # posição do bloco (índice em buffers por bloco, como contagens parciais)
    start: int  # primeira linha do bloco
    rows: int  # quantidade de linhas (BLOCK_ROWS, exceto no último)


def split(n: int) -> list[Block]:
    """Blocos que cobrem as linhas ``0 .. n-1``."""
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"n deve ser um inteiro, não {type(n).__name__}")
    if n < 0:
        raise ValueError("n deve ser >= 0")
    return [Block(i, s, min(BLOCK_ROWS, n - s)) for i, s in enumerate(range(0, n, BLOCK_ROWS))]


def resolve_threads(threads: int | None = None) -> int:
    """Número de threads: o pedido, senão ``NINI_DATAGEN_THREADS``, senão o número de CPUs disponíveis."""
    if threads is None:
        env = os.environ.get(THREADS_ENV)
        if env:
            try:
                threads = int(env)
            except ValueError:
                raise ValueError(f"{THREADS_ENV}={env!r} não é um inteiro") from None
        else:
            threads = _cpu_count()
    if isinstance(threads, bool) or not isinstance(threads, int):
        raise TypeError("threads deve ser um inteiro")
    if threads < 1:
        raise ValueError("threads deve ser >= 1")
    return threads


def _cpu_count() -> int:
    if hasattr(os, "process_cpu_count"):  # Python 3.13+: respeita afinidade/cgroups
        return os.process_cpu_count() or 1  # type: ignore[attr-defined]
    affinity = getattr(os, "sched_getaffinity", None)  # Linux (não existe no Windows nem no macOS)
    if affinity is not None:
        return len(affinity(0)) or 1
    return os.cpu_count() or 1


_pools: dict[int, ThreadPoolExecutor] = {}
_pools_lock = threading.Lock()
_local = threading.local()


def _mark_worker() -> None:
    _local.worker = True


def _pool(workers: int) -> ThreadPoolExecutor:
    with _pools_lock:
        pool = _pools.get(workers)
        if pool is None:
            pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="nini-datacomp", initializer=_mark_worker)
            _pools[workers] = pool
        return pool


def _forget_pools() -> None:  # no processo filho de um fork, as threads do pai não existem
    global _pools_lock  # noqa: PLW0603
    _pools.clear()
    _pools_lock = threading.Lock()


_register_at_fork = getattr(os, "register_at_fork", None)  # não existe no Windows (nem para o pyright de lá)
if _register_at_fork is not None:
    _register_at_fork(after_in_child=_forget_pools)


def parallel_map(fn: Callable[[Item], T], items: Sequence[Item], threads: int | None = None) -> list[T]:
    """``[fn(x) for x in items]`` (mesma ordem), em paralelo quando há mais de um item e mais de uma thread."""
    workers = resolve_threads(threads)  # o pool é por número de threads pedido (não por tarefa): poucos pools vivos
    if min(workers, len(items)) <= 1 or getattr(_local, "worker", False):
        return [fn(x) for x in items]
    return list(_pool(workers).map(fn, items))


def map_blocks(fn: Callable[[Block], T], blocks: Sequence[Block], threads: int | None = None) -> list[T]:
    """Aplica ``fn`` a cada bloco e devolve os resultados na ordem dos blocos (paralelo quando vale a pena)."""
    return parallel_map(fn, blocks, threads)
