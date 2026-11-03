"""Formatos de arquivo além do Parquet: `ReadOptions`, a conversão para Parquet temporário que a comparação usa e as
funções públicas de leitura de SAS, Stata e SPSS (``read_sas``, ``scan_sas``...).

A comparação lê cada janela com um filtro pela chave; num CSV ou num .sas7bdat isso leria o arquivo inteiro a cada
janela. Por isso, antes de comparar, cada fonte que não é Parquet é convertida uma vez para Parquet numa pasta
temporária (`stage`), em pedaços que cabem no ``memory_limit``, e a comparação segue como sempre. A pasta some no fim
da comparação.

Formatos (pela extensão, ou ``ReadOptions(format=...)``):

* ``csv`` (.csv, .tsv), ``ndjson`` (.ndjson, .jsonl), ``ipc`` (.arrow, .ipc, .feather): lidos e gravados aos pedaços
  pelo Polars (``scan_*`` + ``sink_parquet``), sem passar o arquivo inteiro pela memória;
* ``json`` (.json), ``avro`` (.avro) e ``excel`` (.xlsx, .xlsm, .xlsb, .xls, .ods): o Polars só lê inteiros, então
  cabem na memória ou não dá (o Excel precisa do pacote ``fastexcel``);
* ``sas`` (.sas7bdat, .xpt), ``stata`` (.dta), ``spss`` (.sav, .zsav, .por): pelo ReadStat da própria biblioteca
  (`core.stat`), aos pedaços.
"""

from __future__ import annotations

import codecs
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .core.stat import StatReader, encoding_supported, stat_family

__all__ = [
    "DEFAULT_STAGE_BYTES",
    "SUFFIXES",
    "FileFormat",
    "ReadOptions",
    "Staged",
    "format_of",
    "read_sas",
    "read_spss",
    "read_stata",
    "scan_sas",
    "scan_spss",
    "scan_stata",
    "stage",
]

FileFormat = Literal["parquet", "csv", "ndjson", "json", "ipc", "avro", "excel", "sas", "stata", "spss"]
SUFFIXES: dict[str, FileFormat] = {
    ".parquet": "parquet",
    ".pq": "parquet",
    ".csv": "csv",
    ".tsv": "csv",
    ".ndjson": "ndjson",
    ".jsonl": "ndjson",
    ".json": "json",
    ".arrow": "ipc",
    ".ipc": "ipc",
    ".feather": "ipc",
    ".avro": "avro",
    ".xlsx": "excel",
    ".xlsm": "excel",
    ".xlsb": "excel",
    ".xls": "excel",
    ".ods": "excel",
    ".sas7bdat": "sas",
    ".xpt": "sas",
    ".dta": "stata",
    ".sav": "spss",
    ".zsav": "spss",
    ".por": "spss",
}
_STAT = ("sas", "stata", "spss")
DEFAULT_STAGE_BYTES = 64 * 1024**2  # tamanho de cada pedaço na conversão (a comparação usa ~1/16 do memory_limit)
# bytes de cada linha em memória, por tipo, para escolher o tamanho dos row groups do Parquet temporário
_TEXT_BYTES = 24
# bytes por valor dos tipos de largura fixa; os outros (texto, binário, aninhados) contam _TEXT_BYTES
_WIDTH: dict[type[pl.DataType], int] = {
    pl.Boolean: 1, pl.Int8: 1, pl.UInt8: 1, pl.Int16: 2, pl.UInt16: 2, pl.Int32: 4, pl.UInt32: 4, pl.Float32: 4,
    pl.Date: 4, pl.Int64: 8, pl.UInt64: 8, pl.Float64: 8, pl.Datetime: 8, pl.Duration: 8, pl.Time: 8, pl.Decimal: 16,
}  # fmt: skip
# CSV numa codificação que o Polars não lê: convertido para UTF-8 aos pedaços deste tamanho
_TRANSCODE_CHUNK = 16 * 1024**2


class ReadOptions(BaseModel):
    """Como ler os arquivos das fontes. Vale para os dois lados (``dc.compare(..., read=ReadOptions(...))``) ou
    um para cada (``read=(ReadOptions(...), ReadOptions(...))``)."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    format: FileFormat | None = Field(default=None, description="Força o formato; por padrão vem da extensão.")
    encoding: str | None = Field(
        default=None,
        description=(
            "Codificação do texto: do CSV (padrão UTF-8; ex. 'latin1', 'cp1252') e dos arquivos de SAS/Stata/SPSS que "
            "não a declaram (.xpt) ou declaram errado (padrão: a do arquivo)."
        ),
    )
    blank_strings_as_null: bool = Field(
        default=True,
        description="SAS/Stata/SPSS: texto vazio (ou só espaços) vira nulo, que é como o SAS grava texto ausente.",
    )
    convert_dates: bool = Field(
        default=True,
        description=(
            "SAS/Stata/SPSS: colunas com formato de data, data e hora ou hora viram Date, Datetime('us') e Time "
            "(False: ficam como o número que o programa guarda)."
        ),
    )
    separator: str | None = Field(default=None, description="CSV: separador (padrão ',' e, em .tsv, tabulação).")
    has_header: bool = Field(default=True, description="CSV: a primeira linha tem os nomes das colunas.")
    decimal_comma: bool = Field(default=False, description="CSV: números com vírgula decimal (1.234,56).")
    null_values: str | list[str] | None = Field(default=None, description="CSV: textos lidos como nulo.")
    infer_schema_length: int | None = Field(
        default=10_000, ge=0, description="CSV/JSON: linhas usadas para deduzir os tipos (None: o arquivo inteiro)."
    )
    schema_overrides: dict[str, Any] | None = Field(
        default=None, description="CSV/JSON/Excel: tipos de algumas colunas, no lugar dos deduzidos."
    )
    sheet: str | int | None = Field(default=None, description="Excel: nome ou número (a partir de 1) da planilha.")
    csv_options: dict[str, Any] = Field(
        default_factory=dict, description="CSV: outros argumentos de pl.scan_csv (quote_char, skip_rows...)."
    )
    recursive: bool = Field(default=True, description="Pasta: procura arquivos também nas subpastas.")
    temp_dir: Path | None = Field(
        default=None, description="Onde guardar o Parquet temporário dos formatos convertidos (padrão: o do sistema)."
    )

    @field_validator("separator")
    @classmethod
    def _one_char(cls, v: str | None) -> str | None:
        if v is not None and len(v) != 1:
            raise ValueError("separator deve ser um caractere só")
        return v

    @field_validator("encoding")
    @classmethod
    def _known_encoding(cls, v: str | None) -> str | None:
        # o Python conhece (CSV) ou o leitor de SAS/Stata/SPSS conhece (também os nomes do SAS, como "wlatin1")
        if v is not None and v.lower() != "utf8-lossy" and not _python_codec(v) and not encoding_supported(v):
            raise ValueError(f"codificação desconhecida: {v!r}")
        return v


def _python_codec(name: str) -> bool:
    try:
        codecs.lookup(name)
    except LookupError:
        return False
    return True


def format_of(path: Path, options: ReadOptions) -> FileFormat:
    """O formato de ``path`` (pela extensão, a não ser que ``options.format`` diga)."""
    if options.format is not None:
        return options.format
    fmt = SUFFIXES.get(path.suffix.lower())
    if fmt is None:
        raise ValueError(
            f"{path.name}: formato desconhecido pela extensão; use ReadOptions(format=...) (extensões conhecidas: "
            f"{', '.join(sorted(SUFFIXES))})"
        )
    return fmt


# ----------------------------------------------------------------------------- conversão para Parquet temporário
@dataclass
class Staged:
    """A pasta temporária com o Parquet convertido de uma fonte; `cleanup` apaga (também quando o objeto some)."""

    files: list[Path] = field(default_factory=list)
    _dir: tempfile.TemporaryDirectory[str] | None = None

    @property
    def path(self) -> Path:
        assert self._dir is not None
        return Path(self._dir.name)

    def next_file(self) -> Path:
        out = self.path / f"part-{len(self.files):05d}.parquet"
        self.files.append(out)
        return out

    def cleanup(self) -> None:
        if self._dir is not None:
            self._dir.cleanup()
            self._dir = None


def _row_group_rows(schema: pl.Schema, chunk_bytes: int) -> int:
    """Linhas por row group para ~``chunk_bytes`` em memória (as janelas da comparação não ficam menores que um)."""
    per_row = sum(_WIDTH.get(dtype.base_type(), _TEXT_BYTES) for dtype in schema.dtypes())
    return int(min(1_000_000, max(1_024, chunk_bytes // max(1, per_row))))


def stage(files: Sequence[Path], fmt: FileFormat, options: ReadOptions, *, chunk_bytes: int) -> Staged:
    """Converte ``files`` (todos do formato ``fmt``) para Parquet numa pasta temporária, na mesma ordem."""
    tmp = tempfile.TemporaryDirectory(  # dono: o Staged (e a fonte que o guarda)
        prefix="nini-datacomp-", dir=options.temp_dir, ignore_cleanup_errors=True
    )
    staged = Staged(_dir=tmp)
    try:
        if fmt in _STAT:
            for f in files:
                _stage_stat(f, options, staged, chunk_bytes)
        elif fmt in ("csv", "ndjson", "ipc"):
            lf = _scan_streaming(files, fmt, options, staged)
            out = staged.next_file()
            rows = _row_group_rows(lf.collect_schema(), chunk_bytes)
            lf.sink_parquet(out, compression="lz4", statistics=True, row_group_size=rows)
        else:
            for f in files:
                df = _read_whole(f, fmt, options)
                df.write_parquet(
                    staged.next_file(),
                    compression="lz4",
                    statistics=True,
                    row_group_size=_row_group_rows(df.schema, chunk_bytes),
                )
    except BaseException:
        staged.cleanup()
        raise
    return staged


def _stage_stat(path: Path, options: ReadOptions, staged: Staged, chunk_bytes: int) -> None:
    with StatReader(
        path,
        encoding=options.encoding,
        blank_strings_as_null=options.blank_strings_as_null,
        convert_dates=options.convert_dates,
        chunk_bytes=chunk_bytes,
    ) as reader:
        wrote = False
        for frame in reader:
            frame.write_parquet(staged.next_file(), compression="lz4", statistics=True, row_group_size=frame.height)
            wrote = True
        if not wrote:  # arquivo sem linhas: um Parquet vazio, com as colunas
            pl.DataFrame(schema=reader.schema).write_parquet(staged.next_file())


def _utf8(path: Path, encoding: str | None, staged: Staged) -> tuple[Path, str]:
    """O CSV em UTF-8 (convertido para a pasta temporária, se precisar) e o ``encoding`` para o Polars."""
    if encoding is not None and encoding.lower() != "utf8-lossy" and not _python_codec(encoding):
        raise ValueError(f"{path.name}: a codificação {encoding!r} só vale para SAS/Stata/SPSS, não para CSV")
    if encoding is None or encoding.lower() == "utf8-lossy" or codecs.lookup(encoding).name == "utf-8":
        return path, "utf8-lossy" if encoding and encoding.lower() == "utf8-lossy" else "utf8"
    out = staged.path / f"utf8-{len(list(staged.path.glob('utf8-*')))}{path.suffix}"
    decoder = codecs.getincrementaldecoder(encoding)("strict")
    with path.open("rb") as src, out.open("wb") as dst:
        while chunk := src.read(_TRANSCODE_CHUNK):
            dst.write(decoder.decode(chunk).encode("utf-8"))
        dst.write(decoder.decode(b"", final=True).encode("utf-8"))
    return out, "utf8"


def _scan_streaming(files: Sequence[Path], fmt: FileFormat, options: ReadOptions, staged: Staged) -> pl.LazyFrame:
    if fmt == "csv":
        converted = [_utf8(f, options.encoding, staged) for f in files]
        sep = options.separator or ("\t" if all(f.suffix.lower() == ".tsv" for f in files) else ",")
        return pl.scan_csv(
            [str(p) for p, _ in converted],
            separator=sep,
            has_header=options.has_header,
            decimal_comma=options.decimal_comma,
            null_values=options.null_values,
            infer_schema_length=options.infer_schema_length,
            schema_overrides=options.schema_overrides,
            encoding="utf8-lossy" if converted[0][1] == "utf8-lossy" else "utf8",
            **options.csv_options,
        )
    if fmt == "ndjson":
        return pl.scan_ndjson(
            [str(f) for f in files],
            infer_schema_length=options.infer_schema_length,
            schema_overrides=options.schema_overrides,
        )
    return pl.scan_ipc([str(f) for f in files])


def _read_whole(path: Path, fmt: FileFormat, options: ReadOptions) -> pl.DataFrame:
    if fmt == "json":
        return pl.read_json(
            path, infer_schema_length=options.infer_schema_length, schema_overrides=options.schema_overrides
        )
    if fmt == "avro":
        return pl.read_avro(path)
    if fmt == "excel":
        kwargs: dict[str, Any] = {}
        if isinstance(options.sheet, int):
            kwargs["sheet_id"] = options.sheet
        elif options.sheet is not None:
            kwargs["sheet_name"] = options.sheet
        try:
            return pl.read_excel(path, schema_overrides=options.schema_overrides, **kwargs)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(f"ler Excel precisa do pacote fastexcel (pip install fastexcel): {e}") from e
    raise ValueError(f"formato {fmt!r} não se converte assim")  # pragma: no cover


# ----------------------------------------------------------------------------- leitura pública: SAS, Stata, SPSS
_EXT = {
    "sas": (".sas7bdat", ".xpt"),
    "stata": (".dta",),
    "spss": (".sav", ".zsav", ".por"),
}


def _reader(path: str | Path, family: str, **kwargs: Any) -> StatReader:
    p = Path(path)
    if stat_family(p) != family:
        raise ValueError(f"{p.name}: não é um arquivo de {family.upper()} (extensões: {', '.join(_EXT[family])})")
    return StatReader(p, **kwargs)


def _read(path: str | Path, family: str, **kwargs: Any) -> pl.DataFrame:
    with _reader(path, family, **kwargs) as reader:
        return reader.read()


def _scan(path: str | Path, family: str, **kwargs: Any) -> pl.LazyFrame:
    from polars.io.plugins import register_io_source  # noqa: PLC0415 - só quem usa scan_* paga o import

    with _reader(path, family, **kwargs) as reader:
        schema = reader.schema

    def source(
        with_columns: list[str] | None, predicate: pl.Expr | None, n_rows: int | None, batch_size: int | None
    ) -> Iterator[pl.DataFrame]:
        del batch_size  # os pedaços já vêm no tamanho do chunk_bytes
        left = n_rows
        with _reader(path, family, **kwargs) as reader:
            for frame in reader:
                df = frame if with_columns is None else frame.select(with_columns)
                if predicate is not None:
                    df = df.filter(predicate)
                if left is not None:
                    df = df.head(left)
                    left -= df.height
                yield df
                if left is not None and left <= 0:
                    return

    return register_io_source(io_source=source, schema=schema)


def read_sas(
    path: str | Path,
    *,
    encoding: str | None = None,
    blank_strings_as_null: bool = True,
    convert_dates: bool = True,
) -> pl.DataFrame:
    """Lê um arquivo do SAS (.sas7bdat ou .xpt) num ``pl.DataFrame``, pelo ReadStat da própria biblioteca.

    ``encoding``: a codificação do texto, se o arquivo não declara (.xpt) ou declara errado (ex. ``"latin1"``,
    ``"wlatin1"``). ``blank_strings_as_null``: texto vazio vira nulo (como o SAS grava texto ausente).
    ``convert_dates``: colunas com formato de data/hora (DATE9., DATETIME20., TIME8....) viram ``Date``,
    ``Datetime("us")`` e ``Time``.
    """
    return _read(
        path, "sas", encoding=encoding, blank_strings_as_null=blank_strings_as_null, convert_dates=convert_dates
    )


def read_stata(
    path: str | Path,
    *,
    encoding: str | None = None,
    blank_strings_as_null: bool = True,
    convert_dates: bool = True,
) -> pl.DataFrame:
    """Lê um arquivo do Stata (.dta) num ``pl.DataFrame`` (veja `read_sas`; datas: formatos %td e %tc)."""
    return _read(
        path, "stata", encoding=encoding, blank_strings_as_null=blank_strings_as_null, convert_dates=convert_dates
    )


def read_spss(
    path: str | Path,
    *,
    encoding: str | None = None,
    blank_strings_as_null: bool = True,
    convert_dates: bool = True,
) -> pl.DataFrame:
    """Lê um arquivo do SPSS (.sav, .zsav ou .por) num ``pl.DataFrame`` (veja `read_sas`; valores marcados como
    ausentes pelo usuário no SPSS também viram nulo)."""
    return _read(
        path, "spss", encoding=encoding, blank_strings_as_null=blank_strings_as_null, convert_dates=convert_dates
    )


def scan_sas(
    path: str | Path,
    *,
    encoding: str | None = None,
    blank_strings_as_null: bool = True,
    convert_dates: bool = True,
) -> pl.LazyFrame:
    """Um ``pl.LazyFrame`` sobre um arquivo do SAS: lido aos pedaços quando a consulta roda (veja `read_sas`). Cada
    ``collect`` lê o arquivo de novo; para várias consultas, `read_sas` ou converter para Parquet sai mais barato."""
    return _scan(
        path, "sas", encoding=encoding, blank_strings_as_null=blank_strings_as_null, convert_dates=convert_dates
    )


def scan_stata(
    path: str | Path,
    *,
    encoding: str | None = None,
    blank_strings_as_null: bool = True,
    convert_dates: bool = True,
) -> pl.LazyFrame:
    """Um ``pl.LazyFrame`` sobre um arquivo do Stata (veja `scan_sas`)."""
    return _scan(
        path, "stata", encoding=encoding, blank_strings_as_null=blank_strings_as_null, convert_dates=convert_dates
    )


def scan_spss(
    path: str | Path,
    *,
    encoding: str | None = None,
    blank_strings_as_null: bool = True,
    convert_dates: bool = True,
) -> pl.LazyFrame:
    """Um ``pl.LazyFrame`` sobre um arquivo do SPSS (veja `scan_sas`)."""
    return _scan(
        path, "spss", encoding=encoding, blank_strings_as_null=blank_strings_as_null, convert_dates=convert_dates
    )
