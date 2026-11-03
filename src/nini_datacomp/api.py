"""`compare`: a função pública.

Abre as fontes, ajusta os nomes das colunas, confere a chave (os erros fatais vão para ``result.errors``, como no
datacompolars) e roda o motor (`pipeline._run`): janelas que cabem no ``memory_limit``, hash de linha nas colunas
exatas, coluna a coluna nas com tolerância, contagens por coluna e amostras.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import polars as pl

from .io import Source, SourceLike, open_source
from .options import CompareOptions
from .pipeline import RowCounts, _run, _tol_of, align_columns, sample_key_names
from .results import ColumnResult, ColumnSummary, ComparisonResult, ExecutionInfo, RowSummary, TypeMismatch
from .windows import parse_memory_limit

__all__ = ["compare"]

# Pedaços da conversão dos formatos que não são Parquet: 1/16 do memory_limit, entre 8 MB e 256 MB (o row group do
# Parquet temporário sai desse tamanho, e as janelas da comparação não ficam menores que um row group)
_STAGE_SHARE = 16
_STAGE_MIN, _STAGE_MAX = 8 * 1024**2, 256 * 1024**2


def compare(
    left: SourceLike,
    right: SourceLike,
    on: str | Sequence[str],
    *,
    options: CompareOptions | None = None,
    **kwargs: Any,
) -> ComparisonResult:
    """Compara ``left`` e ``right`` (caminho de arquivo, pasta, glob, ``pl.DataFrame`` ou ``pl.LazyFrame``) pela chave
    ``on`` (uma coluna ou várias).

    Arquivos: Parquet, CSV/TSV, NDJSON, JSON, Arrow IPC/Feather, Avro, Excel, SAS (.sas7bdat, .xpt), Stata (.dta) e
    SPSS (.sav, .zsav, .por), pela extensão; ``read=dc.ReadOptions(...)`` muda como são lidos (formato, codificação,
    separador...), para os dois lados ou um par (esquerda, direita). Os que não são Parquet são convertidos uma vez
    para Parquet numa pasta temporária, apagada no fim.

    As opções vêm em ``options`` (`CompareOptions`) ou como argumentos nomeados (``abs_tol=0.01``); com os dois, os
    argumentos nomeados ganham. Erros fatais (chave que falta, de tipos diferentes, repetida) não levantam exceção:
    ficam em ``result.errors``, com ``result.aborted`` verdadeiro. List, Array e Struct são comparados pelo valor
    inteiro (as listas de ``unordered_lists``, sem a ordem dos elementos).
    """
    t0 = time.perf_counter()
    if kwargs:
        base = options.model_dump() if options is not None else {}
        options = CompareOptions(**{**base, **kwargs})
    opts = options if options is not None else CompareOptions()
    lower = not opts.case_sensitive_columns
    keys = _keys(on, lower=lower)
    abs_tol, rel_tol = _names_of(opts.abs_tol, lower), _names_of(opts.rel_tol, lower)

    stage_bytes = min(_STAGE_MAX, max(_STAGE_MIN, parse_memory_limit(opts.memory_limit) // _STAGE_SHARE))
    read_left, read_right = opts.read_options
    ls = open_source(left, read_left, stage_bytes=stage_bytes)
    try:
        rs = open_source(right, read_right, stage_bytes=stage_bytes)
    except BaseException:
        ls.cleanup()
        raise
    try:
        return _compare(ls, rs, keys, opts, abs_tol=abs_tol, rel_tol=rel_tol, t0=t0)
    finally:
        ls.cleanup()
        rs.cleanup()


def _compare(
    ls: Source,
    rs: Source,
    keys: list[str],
    opts: CompareOptions,
    *,
    abs_tol: float | dict[str, float],
    rel_tol: float | dict[str, float],
    t0: float,
) -> ComparisonResult:
    lower = not opts.case_sensitive_columns
    errors: list[str] = []
    if lower:
        ls, rs = _lowercase(ls, "left", errors), _lowercase(rs, "right", errors)
    if errors:
        return _failed(opts, keys, errors, left_only=(), right_only=(), t0=t0)
    lschema, rschema = ls.lazy.collect_schema(), rs.lazy.collect_schema()
    left_only = tuple(c for c in lschema.names() if c not in rschema)
    right_only = tuple(c for c in rschema.names() if c not in lschema)
    errors += _key_errors(lschema, rschema, keys, opts.cast)
    if errors:
        return _failed(opts, keys, errors, left_only=left_only, right_only=right_only, t0=t0)

    cols = align_columns(ls, rs, keys, opts.cast)  # schemas só
    unordered = [c.lower() for c in opts.unordered_lists] if lower else list(opts.unordered_lists)
    warnings: list[str] = []
    if unknown_lists := sorted(set(unordered) - {c for c in cols.compared if isinstance(lschema[c], pl.List)}):
        warnings.append(f"unordered_lists names columns that are not compared lists: {unknown_lists}")
    if not cols.compared:
        warnings.append("No common columns besides the join key: only row membership was compared.")
    for label, tol in (("abs_tol", abs_tol), ("rel_tol", rel_tol)):
        if isinstance(tol, dict) and (unknown := sorted(set(tol) - set(cols.compared))):
            warnings.append(f"{label} names columns that are not compared: {unknown}")

    res = _run(
        ls.lazy, rs.lazy, keys, memory_limit=opts.memory_limit, window_rows=opts.window_rows, columns=None,
        prefetch=True, engine="auto", hashing=True, casting=opts.cast, verify_hash=opts.verify_hash,
        threads=opts.threads, abs_tol=abs_tol, rel_tol=rel_tol, ignore_case=opts.ignore_case,
        ignore_spaces=opts.ignore_whitespace, details=opts.details, samples=opts.samples, sources=(ls, rs),
        unordered_lists=unordered, only_common_keys=opts.only_common_keys,
    )  # fmt: skip
    if res.dup_left:
        errors.append(f"Duplicate join keys in the left table: {res.dup_left:,} surplus rows")
    if res.dup_right:
        errors.append(f"Duplicate join keys in the right table: {res.dup_right:,} surplus rows")

    dtypes = {c: cols.left_casts.get(c, lschema[c]) for c in lschema.names()}
    key_dtypes = [_sample_dtype(dtypes[k]) for k in keys]
    details = opts.details and not errors
    results = {}
    for c in res.compared_columns:
        n = res.column_mismatches.get(c, 0) if details else None
        tol = (_tol_of(abs_tol, c), _tol_of(rel_tol, c)) if c in res.tolerance_columns else None
        results[c] = ColumnResult(
            name=c,
            dtype=str(dtypes[c]),
            mismatches=n,
            rate=None if n is None else (n / res.common if res.common else 0.0),
            samples=_samples(
                res, c, keys=keys, key_dtypes=key_dtypes, dtype=_sample_dtype(dtypes[c]), names=opts.names
            ),
            tolerance=tol,
        )
    columns = ColumnSummary(
        join_columns=tuple(keys),
        results=results,
        left_only=left_only,
        right_only=right_only,
        type_mismatches=tuple(TypeMismatch(c, a, b) for c, a, b in res.type_mismatches),
    )
    return ComparisonResult(
        names=opts.names,
        join_columns=tuple(keys),
        rows=_rows(res, aborted=bool(errors)),
        columns=columns,
        errors=tuple(errors),
        warnings=tuple(warnings),
        execution=_execution(res, opts, details=details, t0=t0),
    )


# ----------------------------------------------------------------------------- nomes e chave
def _keys(on: str | Sequence[str], *, lower: bool) -> list[str]:
    keys = [on] if isinstance(on, str) else list(on)
    if not keys:
        raise ValueError("on precisa de ao menos uma coluna")
    if any(not isinstance(k, str) or not k.strip() for k in keys):
        raise ValueError("on não pode ter nomes vazios")
    if lower:
        keys = [k.lower() for k in keys]
    if len(set(keys)) != len(keys):
        raise ValueError(f"on tem colunas repetidas: {keys}")
    return keys


def _names_of(tol: float | dict[str, float], lower: bool) -> float | dict[str, float]:
    if isinstance(tol, dict) and lower:
        return {k.lower(): v for k, v in tol.items()}
    return tol


def _lowercase(src: Source, side: str, errors: list[str]) -> Source:
    """Os nomes das colunas em minúsculas; dois que viram o mesmo são erro fatal (como no datacompolars), e a fonte
    volta como estava (renomear com a colisão quebraria no Polars)."""
    names = src.lazy.collect_schema().names()
    seen: set[str] = set()
    clash = False
    for n in names:
        low = n.lower()
        if low in seen:
            errors.append(f"Columns collide in the {side} table after lowercasing names: '{low}'")
            clash = True
        seen.add(low)
    return src if clash else src.rename({n: n.lower() for n in names if n != n.lower()})


def _key_errors(lschema: pl.Schema, rschema: pl.Schema, keys: Sequence[str], cast: str) -> list[str]:
    errors = []
    for side, schema in (("left", lschema), ("right", rschema)):
        missing = [k for k in keys if k not in schema]
        if missing:
            errors.append(f"Join keys missing from the {side} table: {missing}")
    if errors or cast != "none":
        return errors
    return [
        f"Join key '{k}' has different types: left {lschema[k]} vs right {rschema[k]}"
        for k in keys
        if lschema[k] != rschema[k]
    ]


# ----------------------------------------------------------------------------- resultado
def _sample_dtype(dtype: pl.DataType) -> pl.DataType:
    """O tipo da coluna nas amostras: Categorical e Enum são comparados (e mostrados) como texto."""
    return pl.String() if isinstance(dtype, (pl.Categorical, pl.Enum)) else dtype


def _samples(
    res: RowCounts,
    column: str,
    *,
    keys: Sequence[str],
    key_dtypes: Sequence[pl.DataType],
    dtype: pl.DataType,
    names: tuple[str, str],
) -> pl.DataFrame:
    headers = [*keys, f"{column} ({names[0]})", f"{column} ({names[1]})"]
    frame = res.samples.get(column)
    if frame is None:
        return pl.DataFrame(schema=dict(zip(headers, [*key_dtypes, dtype, dtype], strict=True)))
    raw = [*sample_key_names(len(keys)), "__left", "__right"]
    return frame.rename(dict(zip(raw, headers, strict=True)))


def _rows(res: RowCounts, *, aborted: bool) -> RowSummary:
    if aborted:  # como no datacompolars: nada contado
        return RowSummary(0, 0, 0, 0, 0, 0)
    return RowSummary(
        left_total=res.left_total,
        right_total=res.right_total,
        common=res.common,
        left_only=res.left_only,
        right_only=res.right_only,
        mismatched=res.mismatched or 0,
    )


def _versions() -> tuple[str, str]:
    from . import __version__  # noqa: PLC0415 - o pacote já carregou quando compare roda

    return __version__, pl.__version__


def _execution(res: RowCounts, opts: CompareOptions, *, details: bool, t0: float) -> ExecutionInfo:
    version, polars_version = _versions()
    return ExecutionInfo(
        windows=res.windows,
        rows_per_window=res.rows_per_window,
        column_groups=res.column_groups,
        memory_limit=res.memory_limit,
        window_bytes=res.window_bytes,
        tolerance_columns=res.tolerance_columns,
        verify_hash=opts.verify_hash,
        hash_collisions=res.hash_collisions,
        details_computed=details,
        version=version,
        polars_version=polars_version,
        total_seconds=round(time.perf_counter() - t0, 6),
        timings=dict(res.timings),
    )


def _failed(
    opts: CompareOptions,
    keys: Sequence[str],
    errors: Sequence[str],
    *,
    left_only: tuple[str, ...],
    right_only: tuple[str, ...],
    t0: float,
) -> ComparisonResult:
    """O resultado de uma comparação que não começou (erro fatal antes de ler os dados)."""
    version, polars_version = _versions()
    return ComparisonResult(
        names=opts.names,
        join_columns=tuple(keys),
        rows=RowSummary(0, 0, 0, 0, 0, 0),
        columns=ColumnSummary(join_columns=tuple(keys), results={}, left_only=left_only, right_only=right_only),
        errors=tuple(errors),
        execution=ExecutionInfo(
            verify_hash=opts.verify_hash,
            version=version,
            polars_version=polars_version,
            total_seconds=round(time.perf_counter() - t0, 6),
        ),
    )
