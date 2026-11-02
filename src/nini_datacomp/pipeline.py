"""O caminho de uma comparação por janelas: abre as fontes, planeja as janelas, lê cada uma pelo Polars (a seguinte em
paralelo) e entrega as colunas ao C++ sem cópia.

Fase 1c: só a contagem de linhas (`count_rows`): em comum, só esquerda, só direita e chaves repetidas, somadas janela
a janela. As fases 2 e 3 põem o hash de linha e a comparação por coluna em cima do mesmo `WindowReader`.

Memória: o `WindowReader` mantém no máximo duas janelas vivas (a que está sendo comparada e a seguinte, que é lida
numa thread enquanto isso: o ``collect`` do Polars e os kernels soltam o GIL). É a conta que o `windows.plan_windows`
faz com o ``memory_limit``.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

import polars as pl

from .core.arrow import import_column
from .core.keys import align_keys
from .io import Source, SourceLike, open_source
from .windows import Plan, Window, plan_windows

__all__ = ["Engine", "RowCounts", "WindowReader", "count_rows", "prepare_keys"]

Engine = Literal["auto", "in-memory", "streaming"]


def _is_dictionary(dtype: pl.DataType) -> bool:
    return isinstance(dtype, (pl.Categorical, pl.Enum))


def prepare_keys(left: Source, right: Source, keys: Sequence[str]) -> tuple[Source, Source, list[pl.DataType]]:
    """Confere a chave nos dois lados e converte Categorical/Enum para texto (a igualdade e a ordem são as do texto).

    Devolve as fontes ajustadas e os tipos das colunas da chave (iguais nos dois lados).
    """
    ls, rs = left.lazy.collect_schema(), right.lazy.collect_schema()
    missing = [(side, k) for side, schema in (("esquerda", ls), ("direita", rs)) for k in keys if k not in schema]
    if missing:
        raise ValueError(f"colunas da chave que faltam: {missing}")
    out = []
    for original, schema in ((left, ls), (right, rs)):
        casts = [pl.col(k).cast(pl.String) for k in keys if _is_dictionary(schema[k])]
        if not casts:
            out.append(original)
            continue
        frame = original.frame.with_columns(casts) if original.frame is not None else None
        out.append(Source(original.lazy.with_columns(casts), original.files, frame))
    dtypes = []
    for k in keys:
        dl = pl.String() if _is_dictionary(ls[k]) else ls[k]
        dr = pl.String() if _is_dictionary(rs[k]) else rs[k]
        if dl != dr:
            raise ValueError(f"coluna da chave {k!r} com tipos diferentes: esquerda {dl}, direita {dr}")
        dtypes.append(dl)
    return out[0], out[1], dtypes


@dataclass
class _Timer:
    seconds: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def add(self, name: str, since: float) -> float:
        now = time.perf_counter()
        self.seconds[name] += now - since
        return now


class WindowReader:
    """Itera ``(janela, esquerda, direita)`` com os dois lados de cada janela já em memória; lê a seguinte em paralelo.

    Quem itera precisa soltar as duas tabelas de uma janela (``del``) antes de pedir a próxima: o leitor já está lendo
    a seguinte, e uma referência que sobra na variável do laço é uma terceira janela em memória.

    ``columns_left``/``columns_right``: as colunas lidas (a chave sempre entra). ``prefetch=False`` lê uma janela por
    vez (menos memória, sem sobrepor leitura e comparação). ``engine``: o engine do ``collect`` do Polars nas janelas.
    """

    def __init__(
        self,
        left: Source,
        right: Source,
        plan: Plan,
        keys: Sequence[str],
        dtypes: Sequence[pl.DataType],
        *,
        columns_left: Sequence[str] | None = None,
        columns_right: Sequence[str] | None = None,
        prefetch: bool = True,
        engine: Engine = "auto",
    ) -> None:
        self.left, self.right, self.plan = left, right, plan
        self.keys, self.dtypes = list(keys), list(dtypes)
        self.columns_left = _with_keys(columns_left, keys)
        self.columns_right = _with_keys(columns_right, keys)
        self.prefetch = prefetch
        self.engine: Engine = engine
        self.timer = _Timer()
        self.window_bytes = 0  # maior janela lida (os dois lados), pelo tamanho estimado do Polars

    def _read_side(self, src: Source, window: Window, columns: list[str] | None) -> pl.DataFrame:
        pred = window.predicate(self.keys, self.dtypes)
        if src.frame is not None:  # em memória: filtrar o DataFrame não passa pelo planejador lazy
            frame = src.frame if pred is None else src.frame.filter(pred)
            return frame if columns is None else frame.select(columns)
        lf = src.lazy if pred is None else src.lazy.filter(pred)
        if columns is not None:
            lf = lf.select(columns)
        return lf.collect(engine=self.engine)

    def _read(self, window: Window) -> tuple[Window, pl.DataFrame, pl.DataFrame, float]:
        t = time.perf_counter()
        lf = self._read_side(self.left, window, self.columns_left)
        rf = self._read_side(self.right, window, self.columns_right)
        return window, lf, rf, time.perf_counter() - t

    def __iter__(self) -> Iterator[tuple[Window, pl.DataFrame, pl.DataFrame]]:
        windows = self.plan.windows
        if not self.prefetch or len(windows) == 1:
            for w in windows:
                _, lf, rf, spent = self._read(w)
                self.timer.seconds["read"] += spent
                self._track(lf, rf)
                yield w, lf, rf
            return
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="nini-datacomp-read") as pool:
            pending: Future[tuple[Window, pl.DataFrame, pl.DataFrame, float]] | None = pool.submit(
                self._read, windows[0]
            )
            for i in range(len(windows)):
                t = time.perf_counter()
                assert pending is not None
                w, lf, rf, spent = pending.result()
                self.timer.add("read_wait", t)  # só a parte da leitura que não ficou escondida
                self.timer.seconds["read_background"] += spent
                # a seguinte começa já; esta janela e a seguinte são as duas em memória
                pending = pool.submit(self._read, windows[i + 1]) if i + 1 < len(windows) else None
                self._track(lf, rf)
                yield w, lf, rf
                del lf, rf

    def _track(self, lf: pl.DataFrame, rf: pl.DataFrame) -> None:
        self.window_bytes = max(self.window_bytes, int(lf.estimated_size() + rf.estimated_size()))


def _with_keys(columns: Sequence[str] | None, keys: Sequence[str]) -> list[str] | None:
    if columns is None:
        return None
    return list(keys) + [c for c in columns if c not in keys]


@dataclass(frozen=True)
class RowCounts:
    left_total: int
    right_total: int
    common: int
    left_only: int
    right_only: int
    dup_left: int  # linhas excedentes por chave repetida
    dup_right: int
    windows: int
    rows_per_window: int
    aligned_to_row_groups: bool
    strategies: dict[str, int]  # janelas por estratégia de alinhamento ("merge", "hash")
    timings: dict[str, float]
    window_bytes: int  # a maior janela em memória (os dois lados)
    memory_limit: int

    @property
    def aborted(self) -> bool:
        """Chave repetida em algum lado: a comparação não continua (como no datacompolars)."""
        return bool(self.dup_left or self.dup_right)


def count_rows(
    left: SourceLike,
    right: SourceLike,
    on: Sequence[str],
    *,
    memory_limit: int | str = "auto",
    window_rows: int | None = None,
    columns: Sequence[str] | None = None,
    prefetch: bool = True,
    engine: Engine = "auto",
) -> RowCounts:
    """Linhas em comum, só de um lado e repetidas, por janelas que cabem em ``memory_limit``.

    ``columns``: as colunas lidas além da chave (padrão: todas as que existem nos dois lados, como numa comparação de
    verdade; ``[]`` lê só a chave).
    """
    t0 = time.perf_counter()
    keys = list(on)
    ls, rs = open_source(left), open_source(right)
    ls, rs, dtypes = prepare_keys(ls, rs, keys)
    if columns is None:
        rnames = set(rs.lazy.collect_schema().names())
        columns = [c for c in ls.lazy.collect_schema().names() if c in rnames]
    t = time.perf_counter()
    plan = plan_windows(ls, rs, keys, memory_limit, window_rows=window_rows)
    plan_s = time.perf_counter() - t

    reader = WindowReader(
        ls, rs, plan, keys, dtypes, columns_left=columns, columns_right=columns, prefetch=prefetch, engine=engine
    )
    totals = defaultdict(int)
    strategies: dict[str, int] = defaultdict(int)
    align_s = 0.0
    for _, lf, rf in reader:
        t = time.perf_counter()
        a = align_keys([import_column(lf[k]) for k in keys], [import_column(rf[k]) for k in keys])
        align_s += time.perf_counter() - t
        strategies[a.strategy] += 1
        totals["left"] += lf.height
        totals["right"] += rf.height
        for name in ("common", "only_left", "only_right", "dup_left", "dup_right"):
            totals[name] += getattr(a, name)
        # soltar a janela antes de pedir a próxima: senão, enquanto o leitor entrega a k+1 e já lê a k+2, esta k
        # continua viva pela variável do laço (três janelas em memória, não duas; medido: ~+200 MB em 10M linhas)
        del lf, rf, a

    timings = {"plan": plan_s, "align": align_s, **reader.timer.seconds, "total": time.perf_counter() - t0}
    aborted = bool(totals["dup_left"] or totals["dup_right"])
    return RowCounts(
        left_total=totals["left"],
        right_total=totals["right"],
        common=0 if aborted else totals["common"],
        left_only=0 if aborted else totals["only_left"],
        right_only=0 if aborted else totals["only_right"],
        dup_left=totals["dup_left"],
        dup_right=totals["dup_right"],
        windows=len(plan.windows),
        rows_per_window=plan.rows_per_window,
        aligned_to_row_groups=plan.aligned_to_row_groups,
        strategies=dict(strategies),
        timings={k: round(v, 6) for k, v in timings.items()},
        window_bytes=reader.window_bytes,
        memory_limit=plan.memory_limit,
    )
