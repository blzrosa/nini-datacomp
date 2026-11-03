"""Leitura de SAS (.sas7bdat, .xpt), Stata (.dta) e SPSS (.sav, .zsav, .por) pelo ReadStat compilado na biblioteca
(cpp/stat.cpp), sem dependência a mais.

O C++ lê o arquivo numa thread e entrega pedaços de linhas já em colunas do Arrow; aqui cada pedaço vira um
``pl.DataFrame`` (as colunas passam ao Polars pela interface C do Arrow, `arrow_c.export_array`) e as colunas com
formato de data, data e hora ou hora viram ``Date``, ``Datetime("us")`` e ``Time``.

Tipos: o SAS e o SPSS só guardam números reais (``Float64``) e texto; o Stata guarda também int8/16/32 e float32.
Texto vazio (ou só espaços) vira nulo por padrão: no SAS é assim que um texto ausente fica gravado.
"""

from __future__ import annotations

import ctypes
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from .._native import lib, ptr
from .arrow_c import export_array

__all__ = [
    "STAT_SUFFIXES",
    "StatColumn",
    "StatFamily",
    "StatReader",
    "encoding_supported",
    "stat_family",
]

StatFamily = Literal["sas", "stata", "spss"]

# extensão -> (código do formato no C++, família)
_FORMATS: dict[str, tuple[int, StatFamily]] = {
    ".sas7bdat": (0, "sas"),
    ".xpt": (1, "sas"),
    ".dta": (2, "stata"),
    ".sav": (3, "spss"),
    ".zsav": (3, "spss"),
    ".por": (4, "spss"),
}
STAT_SUFFIXES = tuple(_FORMATS)

_K_STRING, _K_I8, _K_I16, _K_I32, _K_F32, _K_F64 = range(6)
_ARROW = {_K_I8: b"c", _K_I16: b"s", _K_I32: b"i", _K_F32: b"f", _K_F64: b"g", _K_STRING: b"U"}
_NUMPY = {_K_I8: np.int8, _K_I16: np.int16, _K_I32: np.int32, _K_F32: np.float32, _K_F64: np.float64}
_FLAG_BLANK_NULL = 1
DEFAULT_CHUNK_BYTES = 64 * 1024**2

# ----------------------------------------------------------------------------- formatos de data
# Dias de 1960-01-01 (SAS e Stata) e de 1582-10-14 (SPSS) até 1970-01-01
_SAS_EPOCH_DAYS = 3653
_SPSS_EPOCH_DAYS = 141428


def _words(text: str) -> frozenset[str]:
    return frozenset(text.split())


# Formatos do SAS (sem a largura e as casas: DATE9. -> DATE), como no pyreadstat e no haven
_SAS_DATE = _words(
    "DATE DAY DDMMYY DDMMYYB DDMMYYC DDMMYYD DDMMYYN DDMMYYP DDMMYYS MMDDYY MMDDYYB MMDDYYC MMDDYYD MMDDYYN MMDDYYP "
    "MMDDYYS YYMMDD YYMMDDB YYMMDDC YYMMDDD YYMMDDN YYMMDDP YYMMDDS MONYY YYMON MMYY MMYYC MMYYD MMYYN MMYYP MMYYS "
    "YYMM YYMMC YYMMD YYMMN YYMMP YYMMS YYQ YYQC YYQD YYQN YYQP YYQS YYQR YEAR JULDAY JULIAN MINGUO NENGO WEEKDATE "
    "WEEKDATX WEEKDAY WORDDATE WORDDATX DOWNAME MONNAME MONTH QTR QTRR E8601DA B8601DA ND8601DA NLDATE NLDATEL NLDATEM "
    "NLDATEMD NLDATEMDL NLDATEMDM NLDATEMDS NLDATEMN NLDATES NLDATEW NLDATEWN NLDATEYM NLDATEYML NLDATEYMM NLDATEYMS "
    "NLDATEYQ NLDATEYQL NLDATEYQM NLDATEYQS NLDATEYR NLDATEYW EURDFDE EURDFDD EURDFDN EURDFMY EURDFWDX EURDFWKX"
)
_SAS_DATETIME = _words(
    "DATETIME DATEAMPM DTDATE DTMONYY DTWKDATX DTYEAR DTYYQC E8601DT E8601DX E8601DZ E8601LX B8601DT B8601DX B8601DZ "
    "B8601LX MDYAMPM NLDATM NLDATMAP NLDATMDT NLDATML NLDATMM NLDATMMD NLDATMMDL NLDATMMDM NLDATMMDS NLDATMMN NLDATMS "
    "NLDATMTM NLDATMW NLDATMWN NLDATMYM NLDATMYML NLDATMYMM NLDATMYMS NLDATMYQ NLDATMYQL NLDATMYQM NLDATMYQS NLDATMYR "
    "NLDATMYW NLDATMZ"
)
_SAS_TIME = _words(
    "TIME TIMEAMPM TOD HHMM HOUR MMSS E8601TM E8601TX E8601TZ E8601LZ B8601TM B8601TX B8601TZ B8601LZ NLTIME NLTIMAP"
)
_SPSS_DATE = _words("DATE ADATE EDATE JDATE SDATE MOYR QYR WKYR")
_SPSS_DATETIME = _words("DATETIME YMDHMS")
_SPSS_TIME = _words("TIME MTIME")

TimeKind = Literal["date", "datetime", "time"]
_KINDS: dict[str, tuple[tuple[frozenset[str], TimeKind], ...]] = {
    "sas": ((_SAS_DATE, "date"), (_SAS_DATETIME, "datetime"), (_SAS_TIME, "time")),
    "spss": ((_SPSS_DATE, "date"), (_SPSS_DATETIME, "datetime"), (_SPSS_TIME, "time")),
}


def _sas_base(fmt: str) -> str:
    # DATE9. -> DATE, E8601DA10. -> E8601DA, DDMMYYN8. -> DDMMYYN, DATETIME22.3 -> DATETIME
    return re.sub(r"\d*\.\d*$|\d+$", "", fmt.strip().upper())


def time_kind(family: StatFamily, fmt: str) -> TimeKind | None:
    """Se a coluna (numérica) com formato de exibição ``fmt`` guarda datas, datas e horas ou horas."""
    if not fmt:
        return None
    f = fmt.strip()
    if family == "stata":
        if f.startswith(("%td", "%-td", "%d", "%-d")):
            return "date"
        return "datetime" if f.startswith(("%tc", "%-tc", "%tC", "%-tC")) else None
    base = _sas_base(f) if family == "sas" else re.sub(r"\d.*$", "", f.upper())
    return next((kind for names, kind in _KINDS[family] if base in names), None)


def _date_expr(name: str, family: StatFamily, kind: TimeKind) -> pl.Expr:
    """A expressão que converte a coluna ``name`` (número no calendário do programa) para o tipo do Polars."""
    x = pl.col(name).cast(pl.Float64)
    if family == "stata":
        if kind == "date":  # dias desde 1960-01-01
            return (x.floor() - _SAS_EPOCH_DAYS).cast(pl.Int32).cast(pl.Date)
        # milissegundos desde 1960-01-01
        return ((x - _SAS_EPOCH_DAYS * 86_400_000) * 1000).round().cast(pl.Int64).cast(pl.Datetime("us"))
    epoch_days = _SAS_EPOCH_DAYS if family == "sas" else _SPSS_EPOCH_DAYS
    if kind == "date":
        days = x if family == "sas" else x / 86_400  # o SPSS guarda datas em segundos
        return (days.floor() - epoch_days).cast(pl.Int32).cast(pl.Date)
    if kind == "datetime":  # segundos desde a época
        return ((x - epoch_days * 86_400) * 1_000_000).round().cast(pl.Int64).cast(pl.Datetime("us"))
    return (x * 1_000_000_000).round().cast(pl.Int64).cast(pl.Time)  # segundos desde a meia-noite


# ----------------------------------------------------------------------------- leitura
@dataclass(frozen=True)
class StatColumn:
    """Uma coluna do arquivo: nome, tipo no arquivo (código do C++) e formato de exibição (ex. ``DATE9.``)."""

    name: str
    kind: int
    format: str
    time: TimeKind | None  # o que a coluna vira com convert_dates (None: fica como está)

    def dtype(self, *, convert_dates: bool) -> pl.DataType:
        """O tipo da coluna no ``pl.DataFrame``."""
        if convert_dates and self.time is not None:
            return {"date": pl.Date(), "datetime": pl.Datetime("us"), "time": pl.Time()}[self.time]
        return {
            _K_STRING: pl.String(),
            _K_I8: pl.Int8(),
            _K_I16: pl.Int16(),
            _K_I32: pl.Int32(),
            _K_F32: pl.Float32(),
            _K_F64: pl.Float64(),
        }[self.kind]


def stat_family(path: Path) -> StatFamily | None:
    """A família do arquivo pela extensão (None: não é de programa estatístico)."""
    entry = _FORMATS.get(path.suffix.lower())
    return entry[1] if entry else None


def encoding_supported(name: str) -> bool:
    """O leitor converte texto de ``name`` (ex. "latin1", "cp1252", "utf-8")?"""
    raw = name.encode("utf-8")
    buf = np.frombuffer(raw or b"\0", np.uint8).copy()
    return bool(lib().dc_stat_encoding_supported(ptr(buf), len(raw)))


def _bytes(addr: int, n: int) -> np.ndarray:
    if n <= 0 or not addr:
        return np.zeros(1, np.uint8)  # o Arrow quer um endereço válido mesmo vazio
    return np.frombuffer(ctypes.string_at(addr, n), np.uint8)


class StatReader:
    """Um arquivo de SAS, Stata ou SPSS aberto. Use com ``with``; iterar dá os pedaços como ``pl.DataFrame``.

    ``encoding``: a codificação do texto, no lugar da que o arquivo declara (para .xpt, que não declara, e para
    arquivos que declaram errado). ``chunk_bytes``: o tamanho aproximado de cada pedaço em memória.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        encoding: str | None = None,
        blank_strings_as_null: bool = True,
        convert_dates: bool = True,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
        max_rows: int | None = None,
    ) -> None:
        self.path = Path(path)
        entry = _FORMATS.get(self.path.suffix.lower())
        if entry is None:
            raise ValueError(f"{self.path.name}: extensão de SAS/Stata/SPSS desconhecida (aceitas: {STAT_SUFFIXES})")
        if not self.path.is_file():
            raise FileNotFoundError(f"arquivo não encontrado: {self.path}")
        if encoding is not None and not encoding_supported(encoding):
            raise ValueError(
                f"codificação {encoding!r} não suportada na leitura de SAS/Stata/SPSS (aceitas: UTF-8 e as de 1 byte, "
                "como latin1, cp1252/wlatin1, iso-8859-15, cp850)"
            )
        code = entry[0]
        self.family: StatFamily = entry[1]
        self.convert_dates = convert_dates
        raw = str(self.path).encode("utf-8")
        enc = (encoding or "").encode("utf-8")
        praw = np.frombuffer(raw, np.uint8).copy()
        penc = np.frombuffer(enc or b"\0", np.uint8).copy()
        flags = _FLAG_BLANK_NULL if blank_strings_as_null else 0
        self._lib = lib()
        self._h = int(
            self._lib.dc_stat_open(ptr(praw), len(raw), code, ptr(penc), len(enc), chunk_bytes, max_rows or 0, flags)
        )
        if self._h < 0:
            raise MemoryError("não consegui abrir a leitura (memória insuficiente)")
        try:
            self.columns = self._schema()
        except BaseException:
            self.close()
            raise
        n = int(self._lib.dc_stat_row_count(self._h))
        self.row_count: int | None = n if n >= 0 else None
        self._exprs = self._conversions()

    def _error(self) -> str:
        buf = np.zeros(4096, np.uint8)
        n = int(self._lib.dc_stat_error(self._h, ptr(buf), buf.size))
        return bytes(buf[: min(n, buf.size - 1)]).decode("utf-8", "replace")

    def _schema(self) -> list[StatColumn]:
        n = int(self._lib.dc_stat_schema(self._h))
        if n < 0:
            raise ValueError(f"{self.path.name}: não consegui ler o arquivo ({self._error() or 'erro desconhecido'})")
        cols = []
        name, fmt, info = np.zeros(1024, np.uint8), np.zeros(1024, np.uint8), np.zeros(3, np.int64)
        for i in range(n):
            if self._lib.dc_stat_column(self._h, i, ptr(name), name.size, ptr(fmt), fmt.size, ptr(info)) != 0:
                raise RuntimeError(f"coluna {i} sumiu da leitura")
            kind = int(info[0])
            text = bytes(name[: min(int(info[1]), name.size - 1)]).decode("utf-8", "replace")
            form = bytes(fmt[: min(int(info[2]), fmt.size - 1)]).decode("utf-8", "replace")
            time = time_kind(self.family, form) if kind != _K_STRING else None
            cols.append(StatColumn(text, kind, form, time))
        return cols

    @property
    def schema(self) -> pl.Schema:
        """Os nomes e os tipos das colunas como vão sair."""
        return pl.Schema({c.name: c.dtype(convert_dates=self.convert_dates) for c in self.columns})

    def _conversions(self) -> list[pl.Expr]:
        if not self.convert_dates:
            return []
        return [_date_expr(c.name, self.family, c.time) for c in self.columns if c.time is not None]

    def _frame(self, rows: int) -> pl.DataFrame:
        out = np.zeros(5, np.int64)
        series = []
        for i, c in enumerate(self.columns):
            if self._lib.dc_stat_buffers(self._h, i, ptr(out)) != 0:
                raise RuntimeError(f"buffers da coluna {c.name!r} indisponíveis")
            data_addr, data_len, valid_addr, off_addr, nulls = (int(x) for x in out)
            valid = _bytes(valid_addr, (rows + 7) // 8) if nulls else None
            if c.kind == _K_STRING:
                offsets = np.frombuffer(ctypes.string_at(off_addr, 8 * (rows + 1)), np.int64)
                buffers = [valid, offsets, _bytes(data_addr, data_len)]
            else:
                width = np.dtype(_NUMPY[c.kind]).itemsize
                buffers = [valid, _bytes(data_addr, rows * width)]
            exported = export_array(_ARROW[c.kind], rows, buffers, nulls)
            series.append(pl.Series(values=exported).alias(c.name))  # pyright: ignore[reportArgumentType]
        frame = pl.DataFrame(series)
        if self._exprs:
            _check_times(frame, self.columns, self.path)
            frame = frame.with_columns(self._exprs)
        return frame

    def __iter__(self) -> Iterator[pl.DataFrame]:
        while True:
            rows = int(self._lib.dc_stat_next(self._h))
            if rows < 0:
                raise ValueError(f"{self.path.name}: erro na leitura ({self._error() or 'erro desconhecido'})")
            if rows == 0:
                return
            try:
                frame = self._frame(rows)
            finally:
                self._lib.dc_stat_release(self._h)
            yield frame

    def read(self) -> pl.DataFrame:
        """O arquivo inteiro num ``pl.DataFrame``."""
        frames = list(self)
        if not frames:
            return pl.DataFrame(schema=self.schema)
        return pl.concat(frames, how="vertical", rechunk=False)

    def close(self) -> None:
        if getattr(self, "_h", -1) > 0:
            self._lib.dc_stat_close(self._h)
            self._h = -1

    def __enter__(self) -> StatReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def _check_times(frame: pl.DataFrame, columns: Sequence[StatColumn], path: Path) -> None:
    """Horas (segundos desde a meia-noite, ainda como número) fora de [0, 24h) não cabem no ``pl.Time``: erro claro em
    vez de um valor errado."""
    for c in columns:
        if c.time != "time":
            continue
        seconds = frame.get_column(c.name).cast(pl.Float64)
        bad = ((seconds < 0) | (seconds >= 86_400)).sum()
        if bad:
            raise ValueError(
                f"{path.name}: a coluna {c.name!r} (formato {c.format}) tem {bad} valor(es) fora de 00:00–24:00, que "
                "não cabem numa hora do dia; leia com convert_dates=False para ter os segundos como número"
            )
