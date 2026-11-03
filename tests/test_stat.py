"""Leitura de SAS, Stata e SPSS pelo ReadStat compilado na biblioteca (`core.stat`, `read_sas`...).

Os arquivos de tests/fixtures/stat foram gravados pelos escritores do próprio ReadStat (make_fixtures.c), com a mesma
tabela em todos os formatos; real.sas7bdat é um .sas7bdat gravado pelo SAS (o mesmo dos testes do datacompolars).
"""

from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

import polars as pl
import pytest

import nini_datacomp as dc
from nini_datacomp.core.stat import StatReader, encoding_supported, time_kind

STAT = Path(__file__).parent / "fixtures" / "stat"

IDS = [1.0, 2.0, 3.0, 4.0, 5.0]
VALOR = [1.5, None, -3.25, 1e10, 0.1]
NOME = ["Ana", None, "José", "  lead", "Zoë"]
DATA = [dt.date(2024, 1, 31), None, dt.date(1960, 1, 1), dt.date(1959, 12, 31), dt.date(2000, 2, 29)]
MOMENTO = [
    dt.datetime(2024, 1, 31, 12, 34, 56),
    dt.datetime(1960, 1, 1),
    None,
    dt.datetime(1999, 12, 31, 23, 59, 59, 500_000),
    dt.datetime(2024, 2, 29, 0, 0, 1),
]
HORA = [dt.time(1), dt.time(0), dt.time(23, 59, 59), None, dt.time(12, 34, 56)]
SCHEMA = {
    "id": pl.Float64,
    "valor": pl.Float64,
    "nome": pl.String,
    "data": pl.Date,
    "momento": pl.Datetime("us"),
    "hora": pl.Time,
}


def _check_sample(df: pl.DataFrame, *, ascii_text: bool = False) -> None:
    df = df.rename({c: c.lower() for c in df.columns})
    assert dict(df.schema) == SCHEMA
    assert df["id"].to_list() == IDS
    assert df["valor"].to_list() == VALOR
    assert df["nome"].to_list() == (["Ana", None, "Jose", "  lead", "Zoe"] if ascii_text else NOME)
    assert df["data"].to_list() == DATA
    assert df["momento"].to_list() == MOMENTO
    assert df["hora"].to_list() == HORA


@pytest.mark.parametrize("name", ["sample.sas7bdat", "sample.xpt"])
def test_read_sas(name: str) -> None:
    _check_sample(dc.read_sas(STAT / name))


@pytest.mark.parametrize("name", ["sample.sav", "sample.zsav", "sample.por"])
def test_read_spss(name: str) -> None:
    _check_sample(dc.read_spss(STAT / name), ascii_text=name.endswith(".por"))


def test_read_stata() -> None:
    df = dc.read_stata(STAT / "sample.dta")
    assert dict(df.schema) == {
        "id": pl.Int32,
        "valor": pl.Float64,
        "nome": pl.String,
        "data": pl.Date,
        "momento": pl.Datetime("us"),
        "flag": pl.Int8,
        "taxa": pl.Float32,
    }
    assert df["id"].to_list() == [1, 2, 3, 4, 5]
    assert df["valor"].to_list() == VALOR
    assert df["nome"].to_list() == NOME
    assert df["data"].to_list() == DATA
    assert df["momento"].to_list() == MOMENTO
    assert df["flag"].to_list() == [0, 1, None, -5, 100]
    assert df["taxa"].to_list() == [0.5, None, 2.25, -1.0, 3.0]


def test_real_sas_file() -> None:
    """Um .sas7bdat gravado pelo SAS: 10 linhas, 100 colunas, texto, números e datas (MMDDYY10.)."""
    df = dc.read_sas(STAT / "real.sas7bdat")
    assert df.shape == (10, 100)
    assert df.columns[:4] == ["Column1", "Column2", "Column3", "Column4"]
    assert df.schema["Column2"] == pl.String
    assert df.schema["Column4"] == pl.Date
    assert df["Column1"].head(3).to_list() == [0.636, 0.283, 0.452]
    assert df["Column2"].head(6).to_list() == ["pear", "dog", "pear", "dog", None, "dog"]
    epoch = dt.date(1960, 1, 1)
    assert df["Column4"].head(3).to_list() == [epoch + dt.timedelta(days=d) for d in (2170, 6275, 8627)]
    assert df["Column8"][0] is None


def test_blank_strings_and_raw_dates() -> None:
    df = dc.read_sas(STAT / "sample.sas7bdat", blank_strings_as_null=False, convert_dates=False)
    assert df["nome"].to_list() == ["Ana", "", "José", "  lead", "Zoë"]  # espaços no começo ficam; no fim, não
    assert df.schema["data"] == pl.Float64
    assert df["data"].to_list() == [23406.0, None, 0.0, -1.0, 14669.0]  # dias desde 1960-01-01
    spss = dc.read_spss(STAT / "sample.sav", convert_dates=False)
    assert spss["hora"].to_list() == [3600.0, 0.0, 86399.0, None, 45296.0]


def test_encoding() -> None:
    """O .xpt não declara a codificação: texto que não é UTF-8 é erro claro, e ``encoding=`` resolve."""
    with pytest.raises(ValueError, match="encoding="):
        dc.read_sas(STAT / "latin1.xpt")
    for enc in ("latin1", "ISO-8859-1", "wlatin1", "cp1252"):
        df = dc.read_sas(STAT / "latin1.xpt", encoding=enc)
        assert df["nome"].to_list() == ["José", "São Paulo", "açú"]
    with pytest.raises(ValueError, match="não suportada"):
        dc.read_sas(STAT / "latin1.xpt", encoding="shift_jis")
    assert encoding_supported("utf-8")
    assert encoding_supported("WLATIN1")
    assert not encoding_supported("klingon")


def test_scan_and_chunks() -> None:
    lf = dc.scan_sas(STAT / "sample.sas7bdat")
    assert dict(lf.collect_schema()) == SCHEMA
    _check_sample(lf.collect())
    assert lf.filter(pl.col("id") > 3).select("nome").collect()["nome"].to_list() == ["  lead", "Zoë"]
    assert lf.head(2).collect().height == 2
    # pedaços pequenos: o arquivo vem em vários pl.DataFrame, e juntos dão o mesmo
    with StatReader(STAT / "real.sas7bdat", chunk_bytes=1) as reader:
        frames = list(reader)
        assert reader.row_count == 10
    assert len(frames) > 1
    assert pl.concat(frames).equals(dc.read_sas(STAT / "real.sas7bdat"))


def test_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        dc.read_sas(tmp_path / "nada.sas7bdat")
    with pytest.raises(ValueError, match="não é um arquivo de SAS"):
        dc.read_sas(STAT / "sample.dta")
    bad = tmp_path / "lixo.sas7bdat"
    bad.write_bytes(b"isto nao e um sas7bdat" * 100)
    with pytest.raises(ValueError, match=r"lixo\.sas7bdat"):
        dc.read_sas(bad)
    cut = tmp_path / "cortado.sas7bdat"
    cut.write_bytes((STAT / "real.sas7bdat").read_bytes()[:70_000])  # arquivo truncado: erro, não travamento
    with pytest.raises(ValueError, match=r"cortado\.sas7bdat"):
        dc.read_sas(cut)


def test_non_ascii_path(tmp_path: Path) -> None:
    """Caminho com acentos (no Windows, o C++ abre pelo nome em UTF-16)."""
    folder = tmp_path / "relatórios ação"
    folder.mkdir()
    shutil.copy(STAT / "sample.sas7bdat", folder / "café.sas7bdat")
    _check_sample(dc.read_sas(folder / "café.sas7bdat"))


@pytest.mark.parametrize(
    ("family", "fmt", "kind"),
    [
        ("sas", "DATE9.", "date"),
        ("sas", "MMDDYY10.", "date"),
        ("sas", "E8601DA10.", "date"),
        ("sas", "DATETIME22.3", "datetime"),
        ("sas", "TIME8.", "time"),
        ("sas", "BEST12.", None),
        ("sas", "$20.", None),
        ("sas", "", None),
        ("stata", "%td", "date"),
        ("stata", "%tdDD/NN/CCYY", "date"),
        ("stata", "%tc", "datetime"),
        ("stata", "%9.0g", None),
        ("spss", "ADATE10", "date"),
        ("spss", "DATETIME22.1", "datetime"),
        ("spss", "TIME8", "time"),
        ("spss", "F8.2", None),
    ],
)
def test_time_formats(family: str, fmt: str, kind: str | None) -> None:
    assert time_kind(family, fmt) == kind  # type: ignore[arg-type]
