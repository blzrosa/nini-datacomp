"""Exceções do pacote."""

from __future__ import annotations

__all__ = ["AllocatorWarning", "NativeLibraryError", "PerformanceWarning", "UnsupportedTypeError"]


class NativeLibraryError(RuntimeError):
    """A biblioteca C++ não foi encontrada, não carregou ou tem outra versão de ABI."""


class UnsupportedTypeError(TypeError):
    """Coluna de um tipo que os kernels C++ não leem (o pacote não tem caminho lento escondido para ela)."""


class PerformanceWarning(UserWarning):
    """Parte da comparação roda no Polars em vez dos kernels C++ (comparadores próprios em Python): mais lenta."""


class AllocatorWarning(UserWarning):
    """O Polars carregou antes do nini_datacomp, e o alocador dele não pôde ser configurado (mais memória no pico)."""
