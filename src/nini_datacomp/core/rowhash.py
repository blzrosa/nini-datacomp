"""Hash de linha e linhas divergentes de uma janela (kernels ``dc_row_hash`` e ``dc_diff_pairs``, ``cpp/rowhash.cpp``).

O caminho exato da comparação: cada lado ganha um hash de 64 bits por linha, das colunas comparadas (na mesma ordem
nos dois lados), e um par de linhas com a mesma chave (de `align_keys`) diverge quando os hashes diferem. Igualdade
do ``eq_missing`` do Polars: nulo = nulo, ``-0.0 == 0.0``, NaN = NaN, texto byte a byte.

O hash é próprio, igual em toda plataforma e versão do Polars. Duas linhas que diferem numa coluna só nunca colidem;
diferenças em várias colunas colidem com chance ~2^-64 por par. ``verify=True`` reconfere célula a célula os pares de
hash igual (e conta as colisões achadas).

Os dois kernels rodam por blocos de linhas (ou pares), em paralelo (o ctypes solta o GIL).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .._native import lib, ptr
from ..errors import UnsupportedTypeError
from .arrow import Column, FrameColumns
from .blocks import BLOCK_ROWS, Block, map_blocks, parallel_map, resolve_threads, split
from .keys import descriptor, key_kind
from .memory import empty

__all__ = ["MIN_TASK_ROWS", "Divergent", "Part", "diff_pairs", "row_hash", "row_hash_parts"]


def _check_side(columns: Sequence[Column], n: int | None) -> int:
    rows = {c.length for c in columns}
    if len(rows) > 1:
        raise ValueError("as colunas de um lado têm tamanhos diferentes")
    if n is None:
        if not rows:
            raise ValueError("sem colunas, passe o número de linhas (n)")
        return rows.pop()
    if rows and rows.pop() != n:
        raise ValueError(f"as colunas têm {columns[0].length:,} linhas, não {n:,}")
    return n


def row_hash(columns: Sequence[Column], n: int | None = None, *, threads: int | None = None) -> np.ndarray:
    """Hash (uint64) de cada linha das ``columns``. Sem colunas, todas as ``n`` linhas têm o mesmo hash."""
    n = _check_side(columns, n)
    return row_hash_parts([(0, columns)], n, threads=threads)


Part = Sequence[Column] | FrameColumns  # as colunas de um pedaço: uma a uma, ou o DataFrame inteiro (import_frame)
MIN_TASK_ROWS = 4_096


def _task_rows(total: int, threads: int | None) -> int:
    """Linhas por tarefa: ao menos ~4 tarefas por thread (uma janela de tabela larga tem poucas linhas e muitas
    colunas: blocos de 65.536 linhas deixariam a maior parte das threads paradas), entre 4.096 e `BLOCK_ROWS`."""
    workers = resolve_threads(threads)
    return max(MIN_TASK_ROWS, min(BLOCK_ROWS, -(-total // (4 * workers))))


def row_hash_parts(
    parts: Sequence[tuple[int, Part]],
    n: int,
    *,
    first_column: int = 0,
    out: np.ndarray | None = None,
    threads: int | None = None,
) -> np.ndarray:
    """Hash de ``n`` linhas dadas em pedaços: ``(primeira linha, colunas desse pedaço)``, na mesma ordem de colunas.

    Para uma janela que o Polars entregou em vários pedaços de memória (um por row group, por exemplo): cada pedaço é
    lido sem cópia, em vez de juntar as colunas antes (``rechunk``, uma cópia da janela inteira). As colunas de um
    pedaço vêm uma a uma (`Column`) ou descritas de uma vez (`FrameColumns`, de `import_frame`: bem mais barato com
    muitas colunas). As tarefas de todos os pedaços rodam juntas no pool de threads; o resultado é o mesmo que com as
    colunas inteiras.

    Grupos de colunas: com ``first_column`` > 0, as colunas são as de posição ``first_column ..`` da comparação e o
    hash continua o que já está em ``out`` (das colunas anteriores). Hash das colunas em grupos = hash de todas juntas.
    """
    if out is None:
        if first_column:
            raise ValueError("para continuar um hash (first_column > 0), passe o out com o hash das colunas anteriores")
        out = empty(n, np.uint64)
    elif out.dtype != np.uint64 or out.shape != (n,) or not out.flags.c_contiguous:
        raise ValueError(f"out precisa ser um vetor uint64 contíguo de {n:,} posições")
    if n == 0:
        return out
    dc = lib()
    pieces: list[tuple[int, int, np.ndarray]] = []  # (primeira linha, linhas, descritores)
    keep: list[object] = []
    ks: set[int] = set()
    covered = 0
    for start, part in parts:
        if isinstance(part, FrameColumns):
            desc, rows = part.desc, part.length
            keep.append(part)
        else:
            desc, texts = descriptor(part)
            keep.append(texts)
            rows = _check_side(part, None) if part else n - start
        ks.add(int(desc.shape[0]))
        if start != covered or start + rows > n:
            raise ValueError("os pedaços precisam cobrir as linhas 0 .. n-1 em ordem, sem buracos")
        pieces.append((start, rows, desc))
        covered = start + rows
    if covered != n:
        raise ValueError("os pedaços precisam cobrir as linhas 0 .. n-1 em ordem, sem buracos")
    if len(ks) != 1:
        raise ValueError("os pedaços precisam ter as mesmas colunas")
    k = ks.pop()
    step = _task_rows(n, threads)
    tasks = [
        (base, first, min(step, rows - first), desc) for base, rows, desc in pieces for first in range(0, rows, step)
    ]

    def run(task: tuple[int, int, int, np.ndarray]) -> int:
        base, first, rows, desc = task
        return int(dc.dc_row_hash(k, ptr(desc), first, first + rows, first_column, ptr(out[base + first :])))

    codes = parallel_map(run, tasks, threads)
    del keep  # vivos até aqui: o kernel leu os endereços de texto deles
    if any(codes):
        raise UnsupportedTypeError(f"o hash de linha não lê alguma das colunas: {_names(parts[0][1])}")
    return out


def _names(part: Part) -> str:
    if isinstance(part, FrameColumns):
        return ", ".join(part.names)
    return ", ".join(f"{c.name} ({c.dtype}, {c.format.decode(errors='replace')})" for c in part)


@dataclass(frozen=True)
class Divergent:
    """Pares divergentes: posições (uint32) nos vetores de pares de `align_keys`, em ordem crescente."""

    pairs: np.ndarray
    collisions: int  # pares de hash igual mas valores diferentes (só com verify=True; também estão em `pairs`)

    def __len__(self) -> int:
        return int(self.pairs.size)


def diff_pairs(
    left: Sequence[Column],
    right: Sequence[Column],
    left_hash: np.ndarray,
    right_hash: np.ndarray,
    *,
    left_rows: np.ndarray,
    right_rows: np.ndarray,
    verify: bool = False,
    threads: int | None = None,
) -> Divergent:
    """Os pares ``(left_rows[p], right_rows[p])`` cujas linhas diferem nas colunas comparadas.

    ``left``/``right``: as colunas do hash (necessárias só com ``verify``), na mesma ordem nos dois lados.
    """
    if len(left) != len(right):
        raise ValueError("as colunas comparadas precisam ser as mesmas nos dois lados")
    for a, b in zip(left, right, strict=True):
        if key_kind(a.format) != key_kind(b.format) or a.width != b.width:
            raise ValueError(f"coluna {a.name!r}: tipos diferentes nos dois lados ({a.dtype} x {b.dtype})")
    hl, hr = np.ascontiguousarray(left_hash, np.uint64), np.ascontiguousarray(right_hash, np.uint64)
    li, ri = np.ascontiguousarray(left_rows, np.uint32), np.ascontiguousarray(right_rows, np.uint32)
    if li.size != ri.size:
        raise ValueError("left_rows e right_rows precisam ter o mesmo tamanho")
    npairs = int(li.size)
    if npairs == 0:
        return Divergent(np.empty(0, np.uint32), 0)
    if int(li.max()) >= hl.size or int(ri.max()) >= hr.size:
        raise ValueError("índice de linha fora do hash do seu lado")
    if verify:
        _check_side(left, int(hl.size))
        _check_side(right, int(hr.size))
    ldesc, lkeep = descriptor(left) if verify else (np.zeros((1, 6), np.int64), [])
    rdesc, rkeep = descriptor(right) if verify else (np.zeros((1, 6), np.int64), [])
    k = len(left) if verify else 0
    buf = empty(npairs, np.uint32)
    dc = lib()

    def run(b: Block) -> tuple[int, int, int]:
        start, rows = b.start, b.rows
        stats = np.zeros(2, np.int64)
        dc.dc_diff_pairs(
            k, ptr(ldesc), ptr(rdesc), ptr(hl), ptr(hr), ptr(li), ptr(ri), start, start + rows, int(verify),
            ptr(buf[start:]), ptr(stats),
        )  # fmt: skip
        return start, int(stats[0]), int(stats[1])

    parts = map_blocks(run, split(npairs), threads)
    del lkeep, rkeep
    pairs = np.concatenate([buf[s : s + m] for s, m, _ in parts]) if parts else np.empty(0, np.uint32)
    return Divergent(pairs, sum(c for _, _, c in parts))
