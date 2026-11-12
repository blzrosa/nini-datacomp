"""Fontes de dados: abre caminhos, pastas, globs e frames do Polars como LazyFrame e guarda os arquivos Parquet na ordem
em que o Polars os lê (o planejamento das janelas usa os row groups deles).

Parquet (arquivo, pasta com subpastas, glob), ``pl.DataFrame`` e ``pl.LazyFrame``. Os outros formatos (CSV, JSON,
Arrow IPC, Avro, Excel, SAS, Stata, SPSS; `readers`) são convertidos uma vez para Parquet numa pasta temporária, que
some no fim da comparação (ou quando a fonte some).
"""

from __future__ import annotations

import functools
import glob
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ._native import lib, ptr
from .readers import (
    DEFAULT_STAGE_BYTES,
    SUFFIXES,
    WINDOW_COLUMN,
    FileFormat,
    ReadOptions,
    Staged,
    StageLayout,
    format_of,
    stage,
    stage_partitioned,
)

if TYPE_CHECKING:
    from .windows import Window

__all__ = ["KeyStats", "Source", "SourceLike", "Where", "key_stats", "open_source", "partition_source", "row_groups"]

SourceLike = str | Path | pl.DataFrame | pl.LazyFrame


@dataclass(frozen=True)
class Where:
    """Um filtro de linhas (`Source.filter`)."""

    predicate: pl.Expr


Step = tuple[pl.Expr, ...] | dict[str, str] | Where  # um with_columns, uma renomeação ou um filtro
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
    staged: Staged | None = None  # a pasta temporária com o Parquet convertido (formatos que não são Parquet)
    # o Parquet reparticionado "sorted" (`partition_source`): a coluna com o índice da janela de cada linha; a janela
    # k é lida por ``coluna == k`` (e a coluna sai)
    window_column: str | None = None

    @property
    def is_parquet(self) -> bool:
        return bool(self.files)

    def with_columns(self, exprs: Sequence[pl.Expr]) -> Source:
        """A mesma fonte com ``exprs`` aplicadas (no LazyFrame, no DataFrame e em toda leitura de `scan`)."""
        exprs = tuple(exprs)
        frame = self.frame.with_columns(exprs) if self.frame is not None else None
        return replace(self, lazy=self.lazy.with_columns(exprs), frame=frame, steps=(*self.steps, exprs))

    def rename(self, mapping: Mapping[str, str]) -> Source:
        """A mesma fonte com colunas renomeadas (também em toda leitura de `scan`)."""
        mapping = dict(mapping)
        if not mapping:
            return self
        frame = self.frame.rename(mapping) if self.frame is not None else None
        return replace(self, lazy=self.lazy.rename(mapping), frame=frame, steps=(*self.steps, mapping))

    def filter(self, predicate: pl.Expr) -> Source:
        """A mesma fonte só com as linhas de ``predicate`` (também em toda leitura de `scan`)."""
        frame = self.frame.filter(predicate) if self.frame is not None else None
        return replace(self, lazy=self.lazy.filter(predicate), frame=frame, steps=(*self.steps, Where(predicate)))

    def scan(self, files: Sequence[Path]) -> pl.LazyFrame:
        """Só alguns dos arquivos Parquet da fonte, com as mesmas conversões: uma janela que cai em poucos arquivos
        não faz o Polars abrir o rodapé de todos (com 300 colunas, ~1 ms cada, a cada janela)."""
        lf = _scan([str(f) for f in files])
        for step in self.steps:
            if isinstance(step, dict):
                lf = lf.rename(step)
            elif isinstance(step, Where):
                lf = lf.filter(step.predicate)
            else:
                lf = lf.with_columns(step)
        return lf

    def cleanup(self) -> None:
        """Apaga o Parquet temporário dos formatos convertidos (nada a fazer nos outros)."""
        if self.staged is not None:
            self.staged.cleanup()


def partition_source(
    source: Source,
    windows: Sequence[Window],
    keys: Sequence[str],
    options: ReadOptions,
    *,
    chunk_bytes: int,
    layout: StageLayout = "window",
) -> tuple[Source, list[tuple[int, ...]]]:
    """``source`` regravada por janela (`readers.stage_partitioned`, no ``layout``): a nova fonte (sem passos: as
    conversões já estão nos arquivos) e, por janela, os índices dos arquivos dela. `Source.cleanup` apaga a pasta."""
    staged = stage_partitioned(source, windows, keys, options, chunk_bytes=chunk_bytes, layout=layout)
    lazy = _scan([str(f) for f in staged.files])
    if layout == "sorted":
        return Source(lazy, tuple(staged.files), staged=staged, window_column=WINDOW_COLUMN), staged.groups
    return Source(lazy, tuple(staged.files), staged=staged), staged.groups


def _scan(files: list[str]) -> pl.LazyFrame:
    # o `parallel` padrão do Polars ("auto") ganhou dos outros modos na fase 5 (row_groups e columns: ~2x mais lentos
    # em 1M x 300; prefiltered: igual)
    return pl.scan_parquet(files)


def _hidden(f: Path, root: Path) -> bool:
    # como o datacompolars: pastas e arquivos ocultos (".", "_": _SUCCESS, _tmp/...) ficam de fora
    return any(part.startswith((".", "_")) for part in f.relative_to(root).parts)


def _files(raw: str, read: ReadOptions) -> list[Path]:
    """Os arquivos de uma fonte (arquivo, pasta ou glob), na ordem em que são lidos."""
    if _GLOB & set(raw):
        files = sorted(Path(m) for m in glob.glob(raw, recursive=True) if Path(m).is_file())
        if not files:
            raise FileNotFoundError(f"nenhum arquivo casa com {raw!r}")
        return files
    path = Path(raw)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"fonte não encontrada: {raw!r}")
    found = path.rglob("*") if read.recursive else path.glob("*")
    files = sorted(f for f in found if f.is_file() and f.suffix.lower() in SUFFIXES and not _hidden(f, path))
    if read.format is not None:
        files = [f for f in files if SUFFIXES[f.suffix.lower()] == read.format]
    elif any(f.suffix.lower() in _PARQUET for f in files):
        files = [f for f in files if f.suffix.lower() in _PARQUET]  # pasta de Parquet: o resto (um .csv solto) fica
    if not files:
        raise FileNotFoundError(f"nenhum arquivo de dados em {raw!r} (extensões: {', '.join(sorted(SUFFIXES))})")
    return files


def _format(files: Sequence[Path], read: ReadOptions) -> FileFormat:
    formats: set[FileFormat] = {format_of(f, read) for f in files}
    if len(formats) > 1:
        raise ValueError(
            f"a fonte mistura formatos ({', '.join(sorted(formats))}); separe os arquivos ou escolha um com "
            "ReadOptions(format=...)"
        )
    return formats.pop()


def open_source(
    source: SourceLike, read: ReadOptions | None = None, *, stage_bytes: int = DEFAULT_STAGE_BYTES
) -> Source:
    """Abre uma fonte (caminho, pasta, glob, DataFrame ou LazyFrame) como LazyFrame.

    Arquivos que não são Parquet são convertidos para Parquet numa pasta temporária (`readers.stage`), em pedaços de
    ~``stage_bytes`` na memória; `Source.cleanup` apaga a pasta. A pasta é da `Source`: se ela some (coleta de
    lixo), a pasta some junto, então guarde a `Source` enquanto usar o ``lazy`` dela.
    """
    if isinstance(source, pl.LazyFrame):
        return Source(source)
    if isinstance(source, pl.DataFrame):
        return Source(source.lazy(), frame=source)
    if isinstance(source, (str, Path)):
        read = read if read is not None else ReadOptions()
        files = _files(str(source), read)
        fmt = _format(files, read)
        if fmt == "parquet":
            return Source(_scan([str(f) for f in files]), tuple(files))
        staged = stage(files, fmt, read, chunk_bytes=stage_bytes)
        return Source(_scan([str(f) for f in staged.files]), tuple(staged.files), staged=staged)
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
