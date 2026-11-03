"""nini-datacomp: comparação de tabelas em escala, com kernels C++ sobre as colunas do Polars.

Reescrita do datacompolars::

    import nini_datacomp as dc

    result = dc.compare("esquerda/", "direita.parquet", on=["id"], abs_tol={"valor": 0.005}, memory_limit="2GB")
    result.is_match, result.rows.mismatched, result.columns["valor"].mismatches
"""

from __future__ import annotations

from importlib import metadata

# antes de qualquer import do Polars: o alocador dele lê a configuração uma vez só, quando carrega
from .core.allocator import configure as _configure_allocator

ALLOCATOR_STATUS = _configure_allocator()

from . import expr  # noqa: E402
from .api import compare  # noqa: E402
from .core.memory import pool_limit, pool_stats, release_memory  # noqa: E402
from .errors import AllocatorWarning, NativeLibraryError, PerformanceWarning, UnsupportedTypeError  # noqa: E402
from .options import CompareOptions  # noqa: E402
from .readers import (  # noqa: E402
    ReadOptions,
    read_sas,
    read_spss,
    read_stata,
    scan_sas,
    scan_spss,
    scan_stata,
)
from .results import (  # noqa: E402
    ColumnResult,
    ColumnSummary,
    ComparisonResult,
    ExecutionInfo,
    RowSummary,
    TypeMismatch,
)

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
