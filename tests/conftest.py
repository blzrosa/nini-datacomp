"""Importa o nini_datacomp antes de tudo (antes do Polars), como o pacote recomenda: o alocador do Polars sai
configurado (`core/allocator.py`) e o aviso de import fora de ordem não aparece nos testes."""

import nini_datacomp  # noqa: F401
