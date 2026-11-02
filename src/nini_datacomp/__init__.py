"""nini-datacomp: comparação de tabelas em escala, com kernels C++ sobre as colunas do Polars.

Reescrita do datacompolars. Fase 0a: só a fundação (biblioteca C++, ABI, memória, leitura de colunas sem cópia);
``compare`` chega nas próximas fases.
"""

from __future__ import annotations

from importlib import metadata

# antes de qualquer import do Polars: o alocador dele lê a configuração uma vez só, quando carrega
from .core.allocator import configure as _configure_allocator

ALLOCATOR_STATUS = _configure_allocator()

from .core.memory import pool_limit, pool_stats, release_memory  # noqa: E402
from .errors import AllocatorWarning, NativeLibraryError, UnsupportedTypeError  # noqa: E402

try:
    __version__ = metadata.version("nini-datacomp")
except metadata.PackageNotFoundError:  # pragma: no cover (rodando de src/ sem instalar)
    __version__ = "0+desconhecida"

__all__ = [
    "ALLOCATOR_STATUS",
    "AllocatorWarning",
    "NativeLibraryError",
    "UnsupportedTypeError",
    "__version__",
    "pool_limit",
    "pool_stats",
    "release_memory",
]
