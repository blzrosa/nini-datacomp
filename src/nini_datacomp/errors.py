"""Exceções do pacote."""

from __future__ import annotations

__all__ = ["AllocatorWarning", "NativeLibraryError", "UnsupportedTypeError"]


class NativeLibraryError(RuntimeError):
    """A biblioteca C++ não foi encontrada, não carregou ou tem outra versão de ABI."""


class UnsupportedTypeError(TypeError):
    """Coluna de um tipo que os kernels C++ não leem (o pacote não tem caminho lento escondido para ela)."""


class AllocatorWarning(UserWarning):
    """O Polars carregou antes do nini_datacomp, e o alocador dele não pôde ser configurado (mais memória no pico)."""
