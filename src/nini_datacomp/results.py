"""`ComparisonResult`: o que `compare` devolve.

Os mesmos dados do resultado do datacompolars 0.4.4, com duas mudanças da tabela de migração: ``columns`` é um
mapeamento por nome (``result.columns["valor"].mismatches``) e as amostras são ``pl.DataFrame``. Dataclasses
imutáveis (não pydantic: as amostras são DataFrames); ``to_dict()``/``to_json()`` dão a forma serializável.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from typing import Any

import polars as pl

__all__ = [
    "ColumnResult",
    "ColumnSummary",
    "ComparisonResult",
    "ExecutionInfo",
    "RowSummary",
    "TypeMismatch",
]


@dataclass(frozen=True)
class RowSummary:
    left_total: int  # linhas da esquerda (em comum + só à esquerda)
    right_total: int
    common: int  # chaves que existem nos dois lados
    left_only: int
    right_only: int
    mismatched: int  # linhas em comum com algum valor diferente

    @property
    def matched(self) -> int:
        return self.common - self.mismatched

    @property
    def mismatch_rate(self) -> float:
        return self.mismatched / self.common if self.common else 0.0


@dataclass(frozen=True)
class TypeMismatch:
    column: str
    left: str
    right: str


@dataclass(frozen=True)
class ColumnResult:
    """Uma coluna comparada."""

    name: str
    dtype: str
    mismatches: int | None  # None com details=False
    rate: float | None  # mismatches / linhas em comum
    samples: pl.DataFrame  # chave(s), "<coluna> (<esquerda>)", "<coluna> (<direita>)"; na ordem da chave
    tolerance: tuple[float, float] | None = None  # (abs_tol, rel_tol), se a coluna foi comparada com tolerância


@dataclass(frozen=True)
class ColumnSummary(Mapping[str, ColumnResult]):
    """As colunas da comparação. Como mapeamento: só as comparadas, por nome, na ordem da esquerda."""

    join_columns: tuple[str, ...]
    results: Mapping[str, ColumnResult]
    left_only: tuple[str, ...] = ()
    right_only: tuple[str, ...] = ()
    type_mismatches: tuple[TypeMismatch, ...] = ()  # tipos diferentes: NÃO comparadas

    def __getitem__(self, name: str) -> ColumnResult:
        return self.results[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    @property
    def compared(self) -> tuple[str, ...]:
        return tuple(self.results)

    def mismatched(self) -> list[ColumnResult]:
        """As colunas com divergência, da que mais diverge para a que menos (empate: pelo nome)."""
        bad = [c for c in self.results.values() if c.mismatches]
        return sorted(bad, key=lambda c: (-(c.mismatches or 0), c.name))


@dataclass(frozen=True)
class ExecutionInfo:
    windows: int = 0
    rows_per_window: int = 0
    column_groups: int = 1  # em quantos grupos de colunas cada janela foi lida
    memory_limit: int = 0  # bytes
    window_bytes: int = 0  # a maior janela em memória (os dois lados)
    tolerance_columns: tuple[str, ...] = ()  # comparadas coluna a coluna; as outras, pelo hash de linha
    verify_hash: bool = False
    hash_collisions: int = 0  # só com verify_hash
    details_computed: bool = False
    version: str = ""
    polars_version: str = ""
    total_seconds: float = 0.0
    timings: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ComparisonResult:
    names: tuple[str, str]
    join_columns: tuple[str, ...]
    rows: RowSummary
    columns: ColumnSummary
    errors: tuple[str, ...] = ()  # erros fatais: a comparação não terminou (as contagens ficam em zero)
    warnings: tuple[str, ...] = ()
    execution: ExecutionInfo = field(default_factory=ExecutionInfo)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def left_name(self) -> str:
        return self.names[0]

    @property
    def right_name(self) -> str:
        return self.names[1]

    @property
    def aborted(self) -> bool:
        return bool(self.errors)

    @property
    def is_match(self) -> bool:
        """Os dois lados são iguais: mesmas linhas, mesmos valores e mesmas colunas (com os mesmos tipos)."""
        return (
            not self.aborted
            and self.rows.mismatched == 0
            and self.rows.left_only == 0
            and self.rows.right_only == 0
            and not self.columns.left_only
            and not self.columns.right_only
            and not self.columns.type_mismatches
        )

    def to_dict(self) -> dict[str, Any]:
        """Tudo em tipos do JSON (as amostras como listas de linhas, com cabeçalhos)."""
        r, c, e = self.rows, self.columns, self.execution
        return {
            "names": list(self.names),
            "join_columns": list(self.join_columns),
            "aborted": self.aborted,
            "is_match": self.is_match,
            "rows": {
                "left_total": r.left_total,
                "right_total": r.right_total,
                "common": r.common,
                "left_only": r.left_only,
                "right_only": r.right_only,
                "mismatched": r.mismatched,
                "matched": r.matched,
                "mismatch_rate": r.mismatch_rate,
            },
            "columns": {
                "compared": list(c.compared),
                "left_only": list(c.left_only),
                "right_only": list(c.right_only),
                "type_mismatches": [{"column": t.column, "left": t.left, "right": t.right} for t in c.type_mismatches],
                "results": {
                    name: {
                        "dtype": col.dtype,
                        "mismatches": col.mismatches,
                        "rate": col.rate,
                        "tolerance": list(col.tolerance) if col.tolerance else None,
                        "samples": {
                            "headers": col.samples.columns,
                            "rows": [[_jsonable(v) for v in row] for row in col.samples.rows()],
                        },
                    }
                    for name, col in c.items()
                },
            },
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "execution": {
                "windows": e.windows,
                "rows_per_window": e.rows_per_window,
                "column_groups": e.column_groups,
                "memory_limit": e.memory_limit,
                "window_bytes": e.window_bytes,
                "tolerance_columns": list(e.tolerance_columns),
                "verify_hash": e.verify_hash,
                "hash_collisions": e.hash_collisions,
                "details_computed": e.details_computed,
                "version": e.version,
                "polars_version": e.polars_version,
                "total_seconds": e.total_seconds,
                "timings": dict(e.timings),
            },
            "created_at": self.created_at.isoformat(),
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def _jsonable(v: Any) -> Any:
    """Um valor de amostra em tipo do JSON: datas e horas em ISO, NaN e infinitos como texto (o JSON não os tem),
    bytes em hexadecimal, Decimal, durações e o resto como texto."""
    if v is None or isinstance(v, (bool, int, str)) or (isinstance(v, float) and math.isfinite(v)):
        out = v
    elif isinstance(v, (datetime, date, time)):
        out = v.isoformat()
    elif isinstance(v, bytes):
        out = v.hex()
    else:  # Decimal, timedelta, NaN, infinitos
        out = str(v)
    return out
