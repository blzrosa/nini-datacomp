"""Fontes de dados: abre caminhos, pastas, globs e frames do Polars como LazyFrame e guarda os arquivos Parquet na ordem
em que o Polars os lê (o planejamento das janelas usa os row groups deles).

Por enquanto: Parquet (arquivo, pasta com subpastas, glob), ``pl.DataFrame`` e ``pl.LazyFrame``. CSV e SAS chegam na
fase 6 (``ReadOptions``).
"""

from __future__ import annotations

import functools
import glob
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from ._native import lib, ptr

__all__ = ["KeyStats", "Source", "SourceLike", "key_stats", "open_source", "row_groups"]

SourceLike = str | Path | pl.DataFrame | pl.LazyFrame
Step = tuple[pl.Expr, ...] | dict[str, str]  # um with_columns ou uma renomeação
_PARQUET = (".parquet", ".pq")
_GLOB = set("*?[")
_MAGIC = b"PAR1"


@dataclass(frozen=True)
class Source:
    """Uma fonte aberta: o LazyFrame e, se veio de Parquet, os arquivos na ordem de leitura."""

    lazy: pl.LazyFrame
    files: tuple[Path, ...] = ()
    frame: pl.DataFrame | None = None  # a fonte já em memória (DataFrame), se for o caso
    # o que se aplica depois de ler (em toda leitura de `scan`): with_columns (conversões de tipo) e renomeações
    steps: tuple[Step, ...] = ()

    @property
    def is_parquet(self) -> bool:
        return bool(self.files)

    def with_columns(self, exprs: Sequence[pl.Expr]) -> Source:
        """A mesma fonte com ``exprs`` aplicadas (no LazyFrame, no DataFrame e em toda leitura de `scan`)."""
        exprs = tuple(exprs)
        frame = self.frame.with_columns(exprs) if self.frame is not None else None
        return Source(self.lazy.with_columns(exprs), self.files, frame, (*self.steps, exprs))

    def rename(self, mapping: Mapping[str, str]) -> Source:
        """A mesma fonte com colunas renomeadas (também em toda leitura de `scan`)."""
        mapping = dict(mapping)
        if not mapping:
            return self
        frame = self.frame.rename(mapping) if self.frame is not None else None
        return Source(self.lazy.rename(mapping), self.files, frame, (*self.steps, mapping))

    def scan(self, files: Sequence[Path]) -> pl.LazyFrame:
        """Só alguns dos arquivos Parquet da fonte, com as mesmas conversões: uma janela que cai em poucos arquivos
        não faz o Polars abrir o rodapé de todos (com 300 colunas, ~1 ms cada, a cada janela)."""
        lf = _scan([str(f) for f in files])
        for step in self.steps:
            lf = lf.rename(step) if isinstance(step, dict) else lf.with_columns(step)
        return lf


def _scan(files: list[str]) -> pl.LazyFrame:
    # o `parallel` padrão do Polars ("auto") ganhou dos outros modos na fase 5 (row_groups e columns: ~2x mais lentos
    # em 1M x 300; prefiltered: igual)
    return pl.scan_parquet(files)


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
        return Source(_scan([str(f) for f in files]), tuple(files))
    raise TypeError(f"fonte não suportada: {type(source).__name__} (use caminho, pl.DataFrame ou pl.LazyFrame)")


def _footer(path: Path) -> np.ndarray:
    """Os bytes do FileMetaData do arquivo Parquet (o rodapé, sem os 8 finais). Guardados por arquivo (caminho, tamanho
    e data de modificação): o planejamento lê o rodapé de cada arquivo várias vezes, e com 300 colunas ele tem dezenas
    de KB."""
    st = path.stat()
    return _read_footer(str(path), st.st_size, st.st_mtime_ns)


@functools.lru_cache(maxsize=256)
def _read_footer(raw: str, size: int, mtime_ns: int) -> np.ndarray:
    del mtime_ns  # só faz parte da chave do cache
    path = Path(raw)
    with path.open("rb") as f:
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
    footer.flags.writeable = False  # compartilhado pelo cache
    return footer


def row_groups(path: Path) -> list[int]:
    """Linhas de cada row group do arquivo Parquet, lidas do rodapé (kernel ``dc_parquet_row_groups``)."""
    footer = _footer(path)
    cap = 4096
    while True:
        out = np.zeros(cap, np.int64)
        count = int(lib().dc_parquet_row_groups(ptr(footer), footer.size, ptr(out), cap))
        if count < 0:
            raise ValueError(f"{path}: rodapé Parquet ilegível")
        if count <= cap:
            return [int(x) for x in out[:count]]
        cap = count


@dataclass(frozen=True)
class KeyStats:
    """As estatísticas de uma coluna inteira num row group, do rodapé do Parquet (None: o arquivo não gravou)."""

    rows: int
    min: int | None
    max: int | None
    nulls: int | None


def key_stats(path: Path, column: str) -> list[KeyStats]:
    """Mínimo, máximo e nulos da coluna ``column`` (inteira, do nível de cima) em cada row group, do rodapé (kernel
    ``dc_parquet_key_stats``): o planejamento das janelas usa isso no lugar de ler a chave."""
    footer = _footer(path)
    name = np.frombuffer(column.encode("utf-8") or b"\0", np.uint8).copy()
    name_len = len(column.encode("utf-8"))
    cap = 4096
    while True:
        out = np.zeros((cap, 5), np.int64)
        count = int(lib().dc_parquet_key_stats(ptr(footer), footer.size, ptr(name), name_len, ptr(out), cap))
        if count < 0:
            raise ValueError(f"{path}: rodapé Parquet ilegível")
        if count <= cap:
            break
        cap = count
    return [
        KeyStats(
            int(rows), int(lo) if flags & 1 else None, int(hi) if flags & 1 else None, int(nulls) if flags & 2 else None
        )
        for rows, lo, hi, nulls, flags in out[:count].tolist()
    ]
