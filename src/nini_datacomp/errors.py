"""Exceções do pacote."""

from __future__ import annotations

__all__ = ["NativeLibraryError", "UnsupportedTypeError"]


class NativeLibraryError(RuntimeError):
    """A biblioteca C++ não foi encontrada, não carregou ou tem outra versão de ABI."""


class UnsupportedTypeError(TypeError):
    """Coluna de um tipo que os kernels C++ não leem (o pacote não tem caminho lento escondido para ela)."""
