"""Fontes de dados: abre caminhos, pastas, globs e frames do Polars como LazyFrame e guarda os arquivos Parquet na ordem
em que o Polars os lê (o planejamento das janelas usa os row groups deles).

Por enquanto: Parquet (arquivo, pasta com subpastas, glob), ``pl.DataFrame`` e ``pl.LazyFrame``. CSV e SAS chegam na
fase 6 (``ReadOptions``).
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from ._native import lib, ptr

__all__ = ["Source", "SourceLike", "open_source", "row_groups"]

SourceLike = str | Path | pl.DataFrame | pl.LazyFrame
_PARQUET = (".parquet", ".pq")
_GLOB = set("*?[")
_MAGIC = b"PAR1"


@dataclass(frozen=True)
class Source:
    """Uma fonte aberta: o LazyFrame e, se veio de Parquet, os arquivos na ordem de leitura."""

    lazy: pl.LazyFrame
    files: tuple[Path, ...] = ()
    frame: pl.DataFrame | None = None  # a fonte já em memória (DataFrame), se for o caso

    @property
    def is_parquet(self) -> bool:
        return bool(self.files)


def _parquet_files(raw: str) -> list[Path]:
    if _GLOB & set(raw):
        files = sorted(Path(m) for m in glob.glob(raw, recursive=True) if Path(m).is_file())
        if not files:
            raise FileNotFoundError(f"nenhum arquivo casa com {raw!r}")
        return files
    path = Path(raw)
    if path.is_dir():
        # como o datacompolars: subpastas incluídas; pastas e arquivos ocultos (".", "_") ficam de fora
        files = sorted(
            f
            for f in path.rglob("*")
            if f.is_file()
            and f.suffix.lower() in _PARQUET
            and not any(part.startswith((".", "_")) for part in f.relative_to(path).parts)
        )
        if not files:
            raise FileNotFoundError(f"nenhum arquivo Parquet em {raw!r}")
        return files
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"fonte não encontrada: {raw!r}")


def open_source(source: SourceLike) -> Source:
    """Abre uma fonte (caminho, pasta, glob, DataFrame ou LazyFrame) como LazyFrame."""
    if isinstance(source, pl.LazyFrame):
        return Source(source)
    if isinstance(source, pl.DataFrame):
        return Source(source.lazy(), frame=source)
    if isinstance(source, (str, Path)):
        raw = str(source)
        files = _parquet_files(raw)
        bad = [f.name for f in files if f.suffix.lower() not in _PARQUET]
        if bad:
            raise ValueError(f"por enquanto só Parquet (CSV e SAS chegam na fase 6): {bad[:3]}")
        return Source(pl.scan_parquet([str(f) for f in files]), tuple(files))
    raise TypeError(f"fonte não suportada: {type(source).__name__} (use caminho, pl.DataFrame ou pl.LazyFrame)")


def row_groups(path: Path) -> list[int]:
    """Linhas de cada row group do arquivo Parquet, lidas do rodapé (kernel ``dc_parquet_row_groups``)."""
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        if size < 12:
            raise ValueError(f"{path}: pequeno demais para ser Parquet")
        f.seek(size - 8)
        tail = f.read(8)
        if tail[4:] != _MAGIC:
            raise ValueError(f"{path}: não termina com PAR1 (não é Parquet)")
        n = int.from_bytes(tail[:4], "little")
        if n <= 0 or n > size - 12:
            raise ValueError(f"{path}: tamanho de rodapé inválido ({n})")
        f.seek(size - 8 - n)
        footer = np.frombuffer(f.read(n), np.uint8).copy()
    cap = 4096
    while True:
        out = np.zeros(cap, np.int64)
        count = int(lib().dc_parquet_row_groups(ptr(footer), n, ptr(out), cap))
        if count < 0:
            raise ValueError(f"{path}: rodapé Parquet ilegível")
        if count <= cap:
            return [int(x) for x in out[:count]]
        cap = count
