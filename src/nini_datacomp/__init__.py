"""nini-datacomp: comparação de tabelas em escala, com kernels C++ sobre as colunas do Polars.

Reescrita do datacompolars::

    import nini_datacomp as dc

    result = dc.compare("esquerda/", "direita.parquet", on=["id"], abs_tol={"valor": 0.005}, memory_limit="2GB")
    result.is_match, result.rows.mismatched, result.columns["valor"].mismatches
"""

from __future__ import annotations

from importlib import metadata

from . import expr
from .api import compare
from .core.allocator import mode as _allocator_mode
from .core.memory import pool_limit, pool_stats, release_memory
from .errors import AllocatorWarning, NativeLibraryError, PerformanceWarning, UnsupportedTypeError
from .options import CompareOptions
from .readers import (
    ReadOptions,
    read_sas,
    read_spss,
    read_stata,
    scan_sas,
    scan_spss,
    scan_stata,
)
from .results import (
    ColumnResult,
    ColumnSummary,
    ComparisonResult,
    ExecutionInfo,
    RowSummary,
    TypeMismatch,
)

# o modo do alocador do Polars neste processo ("default" ou "tuned"); o import não mexe no ambiente (core/allocator.py)
ALLOCATOR_STATUS = _allocator_mode()

try:
    __version__ = metadata.version("nini-datacomp")
except metadata.PackageNotFoundError:  # pragma: no cover (rodando de src/ sem instalar)
    __version__ = "0+desconhecida"

__all__ = [
    "ALLOCATOR_STATUS",
    "AllocatorWarning",
    "ColumnResult",
    "ColumnSummary",
    "CompareOptions",
    "ComparisonResult",
    "ExecutionInfo",
    "NativeLibraryError",
    "PerformanceWarning",
    "ReadOptions",
    "RowSummary",
    "TypeMismatch",
    "UnsupportedTypeError",
    "__version__",
    "compare",
    "expr",
    "pool_limit",
    "pool_stats",
    "read_sas",
    "read_spss",
    "read_stata",
    "release_memory",
    "scan_sas",
    "scan_spss",
    "scan_stata",
]
