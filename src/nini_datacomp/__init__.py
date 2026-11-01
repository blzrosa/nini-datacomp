"""nini-datacomp: comparação de tabelas em escala, com kernels C++ sobre as colunas do Polars.

Reescrita do datacompolars. Fase 0a: só a fundação (biblioteca C++, ABI, memória, leitura de colunas sem cópia);
``compare`` chega nas próximas fases.
"""

from __future__ import annotations

from importlib import metadata

from .core.memory import pool_limit, pool_stats, release_memory
from .errors import NativeLibraryError, UnsupportedTypeError

try:
    __version__ = metadata.version("nini-datacomp")
except metadata.PackageNotFoundError:  # pragma: no cover (rodando de src/ sem instalar)
    __version__ = "0+desconhecida"

__all__ = [
    "NativeLibraryError",
    "UnsupportedTypeError",
    "__version__",
    "pool_limit",
    "pool_stats",
    "release_memory",
]
