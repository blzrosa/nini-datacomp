"""Os testes do datacompolars 0.4.4 portados (tests/test_port_*.py) rodam como lá, por este adaptador: `run` recebe
as opções com os nomes do datacompolars e devolve um resultado com os atributos que aqueles testes leem
(``fatal_errors``, ``column_stats``, ``samples`` como lista de tabelas...), montados do `ComparisonResult`.

Mudanças de nome (a tabela de migração do README): ``join_columns`` -> ``on``, ``casting`` -> ``cast``,
``ignore_spaces`` -> ``ignore_whitespace``, ``sample_count`` -> ``samples``, ``column_details`` -> ``details``,
``custom_comparators`` -> ``comparators``, ``common_keys_only`` -> ``only_common_keys``, ``lowercase_columns`` ->
``case_sensitive_columns`` (o contrário), ``left_name``/``right_name`` -> ``names``. Os ajustes do motor
(``cache_windows``, ``prefetch_windows``, ``columns_per_batch``) não existem aqui: o adaptador os ignora, e os testes
passam pelos mesmos casos (janela única e várias janelas) que lá.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import polars as pl
import pytest

import nini_datacomp as dc

NAN = float("nan")

_RENAMES = {
    "join_columns": "on",
    "casting": "cast",
    "ignore_spaces": "ignore_whitespace",
    "sample_count": "samples",
    "column_details": "details",
    "custom_comparators": "comparators",
    "common_keys_only": "only_common_keys",
}
_ENGINE_KNOBS = {"cache_windows", "prefetch_windows", "columns_per_batch"}

# os mesmos modos dos testes do datacompolars: o caminho exato (hash de linha) e o coluna a coluna (qualquer
# tolerância > 0), com janela única e forçando várias janelas
EXACT = [
    pytest.param({"window_rows": None}, id="hash"),
    pytest.param({"window_rows": 2}, id="hash-windows"),
    pytest.param({"window_rows": 2, "verify_hash": True}, id="hash-windows-verify"),
]
COLUMNWISE = [
    pytest.param({"abs_tol": 1e-12, "window_rows": None}, id="columnwise"),
    pytest.param({"abs_tol": 1e-12, "window_rows": 2}, id="columnwise-windows"),
]
ALL_MODES = EXACT + COLUMNWISE


@dataclass(frozen=True)
class ColumnStat:
    column: str
    mismatches: int


@dataclass(frozen=True)
class SampleTable:
    column: str
    total_mismatches: int
    headers: list[str]
    rows: list[list[Any]]


@dataclass(frozen=True)
class Columns:
    compared: list[str]
    left_only: list[str]
    right_only: list[str]
    type_mismatches: tuple[dc.TypeMismatch, ...]


class Result:
    """O `ComparisonResult` com a forma do resultado do datacompolars que os testes portados leem."""

    def __init__(self, result: dc.ComparisonResult) -> None:
        self.result = result
        self.rows = result.rows
        self.is_match = result.is_match
        self.aborted = result.aborted
        self.fatal_errors = list(result.errors)
        self.warnings = list(result.warnings)
        self.execution = result.execution
        c = result.columns
        self.columns = Columns(list(c.compared), list(c.left_only), list(c.right_only), c.type_mismatches)
        mismatched = c.mismatched() if result.execution.details_computed else []
        self.column_stats = [ColumnStat(col.name, col.mismatches or 0) for col in mismatched]
        self.samples = [
            SampleTable(col.name, col.mismatches or 0, list(col.samples.columns), [list(r) for r in col.samples.rows()])
            for col in mismatched
            if col.samples.height
        ]

    def report(self, **kwargs: Any) -> str:
        return self.result.report(**kwargs)


def options(**kwargs: Any) -> tuple[list[str], dict[str, Any]]:
    """As opções do datacompolars nos nomes do nini-datacomp: (on, opções)."""
    kw = dict(kwargs)
    on = kw.pop("join_columns", ["id"])
    if "lowercase_columns" in kw:
        kw["case_sensitive_columns"] = not kw.pop("lowercase_columns")
    if "left_name" in kw or "right_name" in kw:
        kw["names"] = (kw.pop("left_name", "left"), kw.pop("right_name", "right"))
    if kw.get("window_rows") == "auto":
        kw["window_rows"] = None
    return on, {_RENAMES.get(k, k): v for k, v in kw.items() if k not in _ENGINE_KNOBS}


def run(left: Any, right: Any, **kwargs: Any) -> Result:
    on, opts = options(**kwargs)
    return Result(dc.compare(left, right, on, **opts))


def mode_of(param: Any) -> dict[str, Any]:
    """As opções de um dos modos (um ``pytest.param`` de EXACT/COLUMNWISE)."""
    return cast("dict[str, Any]", param.values[0])


def df(**cols: Any) -> pl.DataFrame:
    return pl.DataFrame(cols)
