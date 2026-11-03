"""O caminho de uma comparação por janelas: abre as fontes, planeja as janelas, lê cada uma pelo Polars (a seguinte em
paralelo) e entrega as colunas ao C++ sem cópia.

`count_rows` (fase 1c): só a contagem de linhas (em comum, só esquerda, só direita e chaves repetidas), somadas janela
a janela. `compare_rows` (fase 2): também as linhas divergentes, pelo hash de linha (igualdade exata). A fase 3 põe a
comparação por coluna em cima do mesmo `WindowReader`.

Memória: o `WindowReader` mantém no máximo duas janelas vivas (a que está sendo comparada e a seguinte, que é lida
numa thread enquanto isso: o ``collect`` do Polars e os kernels soltam o GIL). É a conta que o `windows.plan_windows`
faz com o ``memory_limit``.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Literal

import numpy as np
import polars as pl

from .core.arrow import import_column, import_frame
from .core.compare import Tolerance, compare_column, tolerance_for
from .core.keys import Alignment, align_keys
from .core.memory import empty
from .core.rowhash import Part, diff_pairs, row_hash, row_hash_parts
from .errors import UnsupportedTypeError
from .io import Source, SourceLike, open_source
from .windows import Plan, Window, plan_windows

__all__ = [
    "Casting",
    "Columns",
    "Engine",
    "RowCounts",
    "WindowReader",
    "align_columns",
    "compare_rows",
    "count_rows",
    "prepare_keys",
]

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
        out.append(original.with_columns(casts))
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

    ``columns_left``/``columns_right``: as colunas lidas (a chave sempre entra). ``groups``: em vez delas, grupos de
    colunas (as mesmas nos dois lados), cada um lido numa passada com a chave: `pieces` entrega cada janela grupo a
    grupo, e só um grupo (mais o seguinte, sendo lido) fica em memória. ``prefetch=False`` lê uma parte por vez (menos
    memória, sem sobrepor leitura e comparação). ``parallel_sides=False`` lê a direita depois da esquerda, em vez de
    ao mesmo tempo. ``engine``: o engine do ``collect`` do Polars.
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
        groups: Sequence[Sequence[str]] | None = None,
        prefetch: bool = True,
        parallel_sides: bool = True,
        engine: Engine = "auto",
    ) -> None:
        self.left, self.right, self.plan = left, right, plan
        self.parallel_sides = parallel_sides
        self.keys, self.dtypes = list(keys), list(dtypes)
        if groups is None:
            self.groups: list[tuple[list[str] | None, list[str] | None]] = [
                (_with_keys(columns_left, keys), _with_keys(columns_right, keys))
            ]
        else:
            self.groups = [(_with_keys(g, keys), _with_keys(g, keys)) for g in groups] or [(list(keys), list(keys))]
        self.prefetch = prefetch
        self.engine: Engine = engine
        self.timer = _Timer()
        self.window_bytes = 0  # maior parte lida (os dois lados), pelo tamanho estimado do Polars

    def _read_side(
        self, src: Source, window: Window, columns: list[str] | None, files: tuple[int, ...] | None
    ) -> pl.DataFrame:
        pred = window.predicate(self.keys, self.dtypes)
        if src.frame is not None:  # em memória: filtrar o DataFrame não passa pelo planejador lazy
            frame = src.frame if pred is None else src.frame.filter(pred)
            return frame if columns is None else frame.select(columns)
        if files is not None:  # só os arquivos cuja faixa de chaves encosta na janela
            if files:
                lf = src.scan([src.files[i] for i in files])
            else:  # nenhum: zero linhas, com o schema certo (lendo só um rodapé)
                lf, pred = src.scan(src.files[:1]), pl.lit(False)
        else:
            lf = src.lazy
        lf = lf if pred is None else lf.filter(pred)
        if columns is not None:
            lf = lf.select(columns)
        return lf.collect(engine=self.engine)

    def _read(self, task: tuple[Window, int]) -> tuple[Window, int, pl.DataFrame, pl.DataFrame, float]:
        window, g = task
        t = time.perf_counter()
        files_l, files_r = self.plan.files_for(window.index)
        cols_l, cols_r = self.groups[g]
        if self.parallel_sides and self.left.frame is None:
            # os dois lados ao mesmo tempo: cada leitura do Polars de uma janela pequena (poucos row groups) não ocupa
            # todas as threads, e as duas juntas, sim (a memória é a mesma: os dois lados são lidos de qualquer jeito)
            right = _side_pool().submit(self._read_side, self.right, window, cols_r, files_r)
            lf = self._read_side(self.left, window, cols_l, files_l)
            rf = right.result()
        else:
            lf = self._read_side(self.left, window, cols_l, files_l)
            rf = self._read_side(self.right, window, cols_r, files_r)
        return window, g, lf, rf, time.perf_counter() - t

    def pieces(self) -> Iterator[tuple[Window, int, pl.DataFrame, pl.DataFrame]]:
        """``(janela, grupo de colunas, esquerda, direita)``: janela a janela e, em cada uma, grupo a grupo."""
        tasks = [(w, g) for w in self.plan.windows for g in range(len(self.groups))]
        if not self.prefetch or len(tasks) == 1:
            for task in tasks:
                w, g, lf, rf, spent = self._read(task)
                self.timer.seconds["read"] += spent
                self._track(lf, rf)
                yield w, g, lf, rf
                _dispose(lf, rf)  # a última referência: a memória é solta noutra thread
                del lf, rf
            return
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="nini-datacomp-read") as pool:
            pending: Future[tuple[Window, int, pl.DataFrame, pl.DataFrame, float]] | None = pool.submit(
                self._read, tasks[0]
            )
            for i in range(len(tasks)):
                t = time.perf_counter()
                assert pending is not None
                w, g, lf, rf, spent = pending.result()
                self.timer.add("read_wait", t)  # só a parte da leitura que não ficou escondida
                self.timer.seconds["read_background"] += spent
                # a seguinte começa já; esta parte e a seguinte são as duas em memória
                pending = pool.submit(self._read, tasks[i + 1]) if i + 1 < len(tasks) else None
                self._track(lf, rf)
                yield w, g, lf, rf
                _dispose(lf, rf)  # a última referência: a memória é solta noutra thread
                del lf, rf

    def __iter__(self) -> Iterator[tuple[Window, pl.DataFrame, pl.DataFrame]]:
        if len(self.groups) != 1:
            raise RuntimeError("com grupos de colunas, itere com pieces()")
        for w, _, lf, rf in self.pieces():
            yield w, lf, rf
            del lf, rf

    def _track(self, lf: pl.DataFrame, rf: pl.DataFrame) -> None:
        self.window_bytes = max(self.window_bytes, int(lf.estimated_size() + rf.estimated_size()))


_side_executor: ThreadPoolExecutor | None = None
_side_lock = threading.Lock()


def _side_pool() -> ThreadPoolExecutor:
    """As threads que leem o lado direito enquanto a do leitor lê o esquerdo (reaproveitadas entre comparações)."""
    global _side_executor  # noqa: PLW0603
    with _side_lock:
        if _side_executor is None:
            _side_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="nini-datacomp-side")
        return _side_executor


_dispose_executor: ThreadPoolExecutor | None = None


def _dispose(*frames: pl.DataFrame) -> None:
    """Solta ``frames`` numa thread à parte. Com o alocador do Polars devolvendo a memória na hora (core/allocator.py),
    liberar uma janela custa: devolver as páginas ao sistema. Medido na fase 2: até ~1/4 do tempo de uma comparação,
    na thread principal, sem estar em nenhuma etapa. Assim, a comparação da janela seguinte não espera por isso."""
    global _dispose_executor  # noqa: PLW0603
    with _side_lock:
        if _dispose_executor is None:
            _dispose_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nini-datacomp-free")
        pool = _dispose_executor
    pool.submit(_drop, frames)


def _drop(frames: tuple[pl.DataFrame, ...]) -> None:
    del frames


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
    # só no compare_rows (o hash de linha); no count_rows ficam None / vazios
    mismatched: int | None = None  # linhas em comum com algum valor diferente nas colunas comparadas
    hash_collisions: int = 0  # pares de hash igual e valores diferentes, achados pelo verify_hash (estão em mismatched)
    hash_pieces: int = 0  # pedaços de memória lidos no hash (somando as janelas, os grupos e os dois lados)
    column_groups: int = 1  # em quantos grupos de colunas cada janela foi lida
    compared_columns: tuple[str, ...] = ()
    column_mismatches: dict[str, int] = field(default_factory=dict)  # divergências por coluna comparada
    type_mismatches: tuple[tuple[str, str, str], ...] = ()  # (coluna, tipo à esquerda, tipo à direita): não comparadas

    @property
    def aborted(self) -> bool:
        """Chave repetida em algum lado: a comparação não continua (como no datacompolars)."""
        return bool(self.dup_left or self.dup_right)

    @property
    def equal(self) -> int | None:
        """Linhas em comum com todos os valores iguais."""
        return None if self.mismatched is None else self.common - self.mismatched


Casting = Literal["none", "left", "right"]


@dataclass(frozen=True)
class Columns:
    """As colunas de uma comparação, como o datacompolars as escolhe (`align_schemas`)."""

    compared: list[str]  # nos dois lados, fora da chave, com o mesmo tipo (depois do ``casting``)
    type_mismatches: list[tuple[str, str, str]]  # nos dois lados, tipos diferentes: ficam fora da comparação
    left_casts: dict[str, pl.DataType]  # conversões pedidas pelo ``casting``
    right_casts: dict[str, pl.DataType]


def align_columns(
    left: Source, right: Source, keys: Sequence[str], casting: Casting = "none", columns: Sequence[str] | None = None
) -> Columns:
    """Colunas comparadas, tipos divergentes e conversões. ``casting="left"`` converte a direita para os tipos da
    esquerda (também na chave); ``"right"``, o contrário. ``columns`` restringe as colunas comparadas."""
    if casting not in ("none", "left", "right"):
        raise ValueError(f"casting desconhecido: {casting!r} (use 'none', 'left' ou 'right')")
    ls, rs = left.lazy.collect_schema(), right.lazy.collect_schema()
    wanted = None if columns is None else set(columns)
    if wanted is not None:
        unknown = sorted(wanted - (set(ls.names()) & set(rs.names())))
        if unknown:
            raise ValueError(f"colunas pedidas que não estão nos dois lados: {unknown}")
    compared: list[str] = []
    mismatches: list[tuple[str, str, str]] = []
    lcasts: dict[str, pl.DataType] = {}
    rcasts: dict[str, pl.DataType] = {}
    for name in ls.names():
        if name not in rs:
            continue
        dl, dr = ls[name], rs[name]
        is_key = name in keys
        if dl != dr:
            if casting == "left":
                rcasts[name] = dl
            elif casting == "right":
                lcasts[name] = dr
            elif not is_key:
                mismatches.append((name, str(dl), str(dr)))
                continue
        if not is_key and (wanted is None or name in wanted):
            compared.append(name)
    nested = [c for c in compared if lcasts.get(c, ls[c]).is_nested()]
    if nested:
        raise UnsupportedTypeError(f"colunas aninhadas (List, Struct, Array) ainda não são comparadas: {nested}")
    return Columns(compared, mismatches, lcasts, rcasts)


def _with_casts(src: Source, casts: dict[str, pl.DataType]) -> Source:
    if not casts:
        return src
    return src.with_columns([pl.col(c).cast(t) for c, t in casts.items()])


def _dictionary_as_text(src: Source, columns: Sequence[str]) -> Source:
    """Categorical/Enum viram texto: os códigos do dicionário diferem de uma tabela para outra."""
    schema = src.lazy.collect_schema()
    return _with_casts(src, {c: pl.String() for c in columns if _is_dictionary(schema[c])})


Tol = float | Mapping[str, float]  # uma tolerância para todas as colunas, ou por coluna (as ausentes: 0)


def _tol_of(value: Tol, column: str) -> float:
    return float(value.get(column, 0.0)) if isinstance(value, Mapping) else float(value)


def _tolerances(src: Source, columns: Sequence[str], abs_tol: Tol, rel_tol: Tol) -> dict[str, Tolerance]:
    """As colunas comparadas com tolerância (numéricas, com abs_tol ou rel_tol > 0) e a tolerância de cada uma."""
    schema = src.lazy.collect_schema()
    out = {}
    for c in columns:
        tol = tolerance_for(schema[c], _tol_of(abs_tol, c), _tol_of(rel_tol, c))
        if tol is not None:
            out[c] = tol
    return out


def _normalize_text(src: Source, columns: Sequence[str], *, ignore_case: bool, ignore_spaces: bool) -> Source:
    """Texto em minúsculas e/ou sem espaços, com as mesmas expressões do datacompolars (o resultado bate com o
    dele em qualquer texto, inclusive Unicode); a comparação depois é exata (e vai pelo hash de linha)."""
    if not (ignore_case or ignore_spaces):
        return src
    schema = src.lazy.collect_schema()
    exprs = []
    for c in columns:
        if schema[c] != pl.String:
            continue
        e = pl.col(c)
        if ignore_case:
            e = e.str.to_lowercase()
        if ignore_spaces:
            e = e.str.replace_all(r"\s+", "")
        exprs.append(e.alias(c))
    return src.with_columns(exprs) if exprs else src


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
    return _run(
        left, right, on, memory_limit=memory_limit, window_rows=window_rows, columns=columns, prefetch=prefetch,
        engine=engine, hashing=False,
    )  # fmt: skip


def compare_rows(
    left: SourceLike,
    right: SourceLike,
    on: Sequence[str],
    *,
    memory_limit: int | str = "auto",
    window_rows: int | None = None,
    columns: Sequence[str] | None = None,
    casting: Casting = "none",
    abs_tol: Tol = 0.0,
    rel_tol: Tol = 0.0,
    ignore_case: bool = False,
    ignore_spaces: bool = False,
    verify_hash: bool = False,
    prefetch: bool = True,
    engine: Engine = "auto",
    threads: int | None = None,
) -> RowCounts:
    """As contagens do `count_rows`, as linhas divergentes (``mismatched``) e as divergências por coluna
    (``column_mismatches``), com a semântica do datacompolars 0.4.4.

    Colunas comparadas: as dos dois lados, fora da chave, com o mesmo tipo (depois do ``casting``); ``columns``
    restringe. ``abs_tol``/``rel_tol`` (um número, ou por coluna num dicionário): nas colunas numéricas, igual se
    ``|l - r| <= abs_tol + rel_tol * |r|``. ``ignore_case``/``ignore_spaces``: o texto é comparado em minúsculas e/ou
    sem espaços em branco (``str.to_lowercase`` e ``str.replace_all`` de ``\\s+`` do Polars, como no datacompolars).
    As colunas exatas vão pelo hash de linha; ``verify_hash=True`` as confere célula a célula em todos os pares.
    """
    return _run(
        left, right, on, memory_limit=memory_limit, window_rows=window_rows, columns=columns, prefetch=prefetch,
        engine=engine, hashing=True, casting=casting, verify_hash=verify_hash, threads=threads, abs_tol=abs_tol,
        rel_tol=rel_tol, ignore_case=ignore_case, ignore_spaces=ignore_spaces,
    )  # fmt: skip


def _run(
    left: SourceLike,
    right: SourceLike,
    on: Sequence[str],
    *,
    memory_limit: int | str,
    window_rows: int | None,
    columns: Sequence[str] | None,
    prefetch: bool,
    engine: Engine,
    hashing: bool,
    casting: Casting = "none",
    verify_hash: bool = False,
    threads: int | None = None,
    abs_tol: Tol = 0.0,
    rel_tol: Tol = 0.0,
    ignore_case: bool = False,
    ignore_spaces: bool = False,
) -> RowCounts:
    t0 = time.perf_counter()
    keys = list(on)
    ls, rs = open_source(left), open_source(right)
    if hashing:
        cols = align_columns(ls, rs, keys, casting, columns)
        ls, rs, tolerances = _prepare_columns(
            ls, rs, cols, abs_tol=abs_tol, rel_tol=rel_tol, ignore_case=ignore_case, ignore_spaces=ignore_spaces
        )
        read = cols.compared
    else:
        cols, tolerances = None, {}
        read = list(columns) if columns is not None else _common_names(ls, rs)
    ls, rs, dtypes = prepare_keys(ls, rs, keys)
    t = time.perf_counter()
    setup_s = t - t0  # abrir as fontes, alinhar os schemas, as conversões de tipo
    plan = plan_windows(ls, rs, keys, memory_limit, window_rows=window_rows)
    plan_s = time.perf_counter() - t

    groups = plan.split_columns(read) if cols is not None else None
    reader = WindowReader(
        ls, rs, plan, keys, dtypes, columns_left=read, columns_right=read, groups=groups, prefetch=prefetch,
        engine=engine,
    )  # fmt: skip
    totals: dict[str, int] = defaultdict(int)
    strategies: dict[str, int] = defaultdict(int)
    spent: dict[str, float] = defaultdict(float)
    state: _WindowState | None = None
    heights = (0, 0)
    col_counts: dict[str, int] = dict.fromkeys(cols.compared if cols is not None else (), 0)
    for _, g, lf, rf in reader.pieces():
        t = time.perf_counter()
        if g == 0:  # a primeira parte da janela: alinha as chaves (todas as partes trazem as mesmas linhas)
            a = align_keys([import_column(lf[k]) for k in keys], [import_column(rf[k]) for k in keys])
            t = _lap(spent, "align", t)
            aborted = _count(a, lf.height, rf.height, totals, strategies)
            hashing_now = groups is not None and not aborted and a.left.size > 0
            state = _WindowState(a) if hashing_now else None
            heights = (lf.height, rf.height)
        if state is not None:
            assert groups is not None
            if (lf.height, rf.height) != heights:  # pragma: no cover (o Polars mudou a ordem das linhas?)
                raise RuntimeError("os grupos de colunas de uma janela vieram com linhas diferentes")
            totals["pieces"] += state.add(
                lf, rf, groups[g], tolerances=tolerances, counts=col_counts, verify=verify_hash, threads=threads,
                spent=spent, t=t,
            )  # fmt: skip
            if g == len(groups) - 1:
                mismatched, collisions = state.finish()
                totals["mismatched"] += mismatched
                totals["collisions"] += collisions
                state = None
        # soltar a parte antes de pedir a próxima: senão, enquanto o leitor entrega a k+1 e já lê a k+2, esta k
        # continua viva pela variável do laço (três partes em memória, não duas; medido: ~+200 MB em 10M linhas)
        del lf, rf

    timings = {"setup": setup_s, "plan": plan_s, **spent, **reader.timer.seconds}
    total = time.perf_counter() - t0
    # o que nenhuma etapa mediu (laço, Python, a leitura em paralelo disputando o processador); "read_background" não
    # entra na conta: corre em paralelo com as outras
    measured = sum(v for k, v in timings.items() if k != "read_background")
    timings |= {"other": max(0.0, total - measured), "total": total}
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
        mismatched=None if cols is None else (0 if aborted else totals["mismatched"]),
        hash_collisions=totals["collisions"],
        hash_pieces=totals["pieces"],
        column_groups=1 if groups is None else len(groups),
        compared_columns=() if cols is None else tuple(cols.compared),
        column_mismatches={} if cols is None or aborted else col_counts,
        type_mismatches=() if cols is None else tuple(cols.type_mismatches),
    )


def _common_names(ls: Source, rs: Source) -> list[str]:
    rnames = set(rs.lazy.collect_schema().names())
    return [c for c in ls.lazy.collect_schema().names() if c in rnames]


def _prepare_columns(
    ls: Source, rs: Source, cols: Columns, *, abs_tol: Tol, rel_tol: Tol, ignore_case: bool, ignore_spaces: bool
) -> tuple[Source, Source, dict[str, Tolerance]]:
    """As conversões das colunas comparadas (casting, dicionário para texto, opções de texto) e as tolerâncias."""
    ls, rs = _with_casts(ls, cols.left_casts), _with_casts(rs, cols.right_casts)
    ls, rs = _dictionary_as_text(ls, cols.compared), _dictionary_as_text(rs, cols.compared)
    ls = _normalize_text(ls, cols.compared, ignore_case=ignore_case, ignore_spaces=ignore_spaces)
    rs = _normalize_text(rs, cols.compared, ignore_case=ignore_case, ignore_spaces=ignore_spaces)
    return ls, rs, _tolerances(ls, cols.compared, abs_tol, rel_tol)


def _count(a: Alignment, nl: int, nr: int, totals: dict[str, int], strategies: dict[str, int]) -> bool:
    """Soma as contagens de uma janela; devolve se a comparação já abortou (chave repetida em algum lado)."""
    strategies[a.strategy] += 1
    totals["left"] += nl
    totals["right"] += nr
    for name in ("common", "only_left", "only_right", "dup_left", "dup_right"):
        totals[name] += getattr(a, name)
    return bool(totals["dup_left"] or totals["dup_right"])


def _lap(spent: dict[str, float], name: str, since: float) -> float:
    now = time.perf_counter()
    spent[name] += now - since
    return now


class _WindowState:
    """Uma janela, grupo de colunas a grupo: um byte por par (1 = a linha diverge em alguma coluna) e as contagens por
    coluna.

    Em cada grupo: as colunas exatas pelo hash de linha (só os pares de hash diferente divergem) e, nesses pares, a
    contagem coluna a coluna (as linhas deles são poucas: um ``gather`` do Polars e ``dc_compare_column``); as colunas
    com tolerância, uma a uma, em todos os pares. Com ``verify``, as exatas também são conferidas em todos os pares.
    """

    def __init__(self, a: Alignment) -> None:
        self.a = a
        self.bad = np.zeros(a.left.size, np.uint8)
        self.collisions = 0

    def add(
        self,
        lf: pl.DataFrame,
        rf: pl.DataFrame,
        columns: Sequence[str],
        *,
        tolerances: Mapping[str, Tolerance],
        counts: dict[str, int],
        verify: bool,
        threads: int | None,
        spent: dict[str, float],
        t: float,
    ) -> int:
        """Compara um grupo de colunas (somando em ``counts``); devolve quantos pedaços de memória o hash leu."""
        exact = [c for c in columns if c not in tolerances]
        pieces = 0
        if exact:
            hl, hr = empty(lf.height, np.uint64), empty(rf.height, np.uint64)
            pieces = _hash_side(lf, exact, threads, hl, 0) + _hash_side(rf, exact, threads, hr, 0)
            t = _lap(spent, "hash", t)
            d = diff_pairs([], [], hl, hr, left_rows=self.a.left, right_rows=self.a.right, threads=threads)
            del hl, hr
            t = _lap(spent, "diff", t)
            if verify:
                t = self._verify(lf, rf, exact, hashed=d.pairs, counts=counts, threads=threads, spent=spent, t=t)
            else:
                self.bad[d.pairs] = 1
                t = self._count_exact(lf, rf, exact, pairs=d.pairs, counts=counts, threads=threads, spent=spent, t=t)
        for c in columns:
            if c in tolerances:  # em todos os pares: a coluna inteira dos dois lados (juntando os pedaços: uma cópia)
                differs, n = compare_column(
                    import_column(lf[c]), import_column(rf[c]), self.a.left, self.a.right, tolerance=tolerances[c],
                    threads=threads,
                )  # fmt: skip
                self.bad |= differs
                counts[c] += n
        _lap(spent, "tolerance", t)
        return pieces

    def _count_exact(
        self,
        lf: pl.DataFrame,
        rf: pl.DataFrame,
        columns: Sequence[str],
        *,
        pairs: np.ndarray,
        counts: dict[str, int],
        threads: int | None,
        spent: dict[str, float],
        t: float,
    ) -> float:
        """Divergências por coluna, só nos pares divergentes (poucos): as linhas deles lado a lado, coluna a coluna."""
        if not pairs.size:
            return t
        li, ri = pl.Series(self.a.left[pairs]), pl.Series(self.a.right[pairs])
        left = lf.select([pl.col(c).gather(li) for c in columns])
        right = rf.select([pl.col(c).gather(ri) for c in columns])
        same = np.arange(pairs.size, dtype=np.uint32)
        for c in columns:
            _, n = compare_column(import_column(left[c]), import_column(right[c]), same, same, threads=threads)
            counts[c] += n
        return _lap(spent, "columns", t)

    def _verify(
        self,
        lf: pl.DataFrame,
        rf: pl.DataFrame,
        columns: Sequence[str],
        *,
        hashed: np.ndarray,
        counts: dict[str, int],
        threads: int | None,
        spent: dict[str, float],
        t: float,
    ) -> float:
        """Confere as colunas exatas célula a célula em todos os pares; colisões = divergentes que o hash não viu."""
        truly = np.zeros(self.a.left.size, np.uint8)
        for c in columns:
            _, n = compare_column(
                import_column(lf[c]), import_column(rf[c]), self.a.left, self.a.right, differs=truly, threads=threads
            )  # cada coluna inteira (fora de ordem: juntando os pedaços)
            counts[c] += n
        seen = np.zeros(self.a.left.size, np.uint8)
        seen[hashed] = 1
        self.collisions += int(np.count_nonzero(truly & ~seen.astype(bool)))
        self.bad |= truly
        return _lap(spent, "verify", t)

    def finish(self) -> tuple[int, int]:
        """(linhas divergentes, colisões do hash achadas pelo verify) da janela."""
        return int(np.count_nonzero(self.bad)), self.collisions


def _pieces(df: pl.DataFrame, columns: Sequence[str]) -> list[tuple[int, int]]:
    """Faixas de linhas em que cada coluna está num pedaço só de memória (a união dos cortes de todas)."""
    cuts = {0, df.height}
    for c in columns:
        pos = 0
        for n in df[c].chunk_lengths():
            pos += n
            cuts.add(pos)
    edges = sorted(cuts)
    return [(a, b) for a, b in pairwise(edges) if b > a]


def _hash_side(df: pl.DataFrame, columns: Sequence[str], threads: int | None, out: np.ndarray, first: int) -> int:
    """Continua (ou começa, com ``first`` = 0) em ``out`` o hash de linha de um lado da janela com ``columns``, pedaço
    a pedaço (sem juntar as colunas: nenhuma cópia). Devolve quantos pedaços de memória foram lidos."""
    if not columns:
        if first == 0:
            out[:] = row_hash([], df.height, threads=threads)
        return 0
    pieces = _pieces(df, columns)
    parts: list[tuple[int, Part]] = []
    for start, end in pieces:
        part = df.slice(start, end - start).select(columns)
        frame = import_frame(part)  # todas as colunas numa chamada; senão, coluna a coluna
        parts.append((start, frame if frame is not None else [import_column(part[c]) for c in columns]))
    row_hash_parts(parts, df.height, first_column=first, out=out, threads=threads)
    return len(pieces)
