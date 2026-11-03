"""Portado do datacompolars 0.4.4: o relatório (tests/test_report.py), as fontes de arquivo (tests/test_io.py e
tests/test_sas.py) e a validação das opções (tests/test_schema_settings.py), nos nomes do nini-datacomp.

Mudam de propósito (tabela de migração do README): o relatório recebe as opções direto em ``report(...)``, não num
``ReportSettings``; o SAS não precisa do extra ``[sas]`` (o ReadStat vem compilado), então o teste da mensagem de
instalação do extra não existe; e as funções internas de esquema (``normalize_schema``, ``align_schemas``) e o
``uses_exact_hash`` não têm equivalente público: os mesmos casos são conferidos pelo resultado da comparação.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

import nini_datacomp as dc
from nini_datacomp.io import open_source
from nini_datacomp.readers import ReadOptions
from port_helpers import Result, df, options, run

FORMATS = ["text", "markdown", "html", "json"]
REAL_SAS = Path(__file__).parent / "fixtures" / "stat" / "real.sas7bdat"


# ----------------------------------------------------------------------------- relatório
def _mismatching() -> Result:
    left = df(id=list(range(30)), a=list(range(30)), s=["x"] * 30)
    right = df(id=list(range(1, 31)), a=[v + 1 if v < 12 else v for v in range(1, 31)], s=["x"] * 29 + ["y"])
    return run(left, right, left_name="base", right_name="novo", sample_count=3)


@pytest.mark.parametrize("fmt", FORMATS)
def test_every_format_renders_names_and_columns(fmt: str) -> None:
    text = _mismatching().report(format=fmt)
    assert "base" in text and "novo" in text and "a" in text
    if fmt == "json":
        data = json.loads(text)
        assert data["rows"]["mismatched"] > 0 and data["names"][0] == "base"


@pytest.mark.parametrize("fmt", FORMATS)
def test_match_and_aborted_results_render(fmt: str) -> None:
    d = df(id=[1, 2], a=[1, 2])
    assert run(d, d).report(format=fmt)
    aborted = run(d, df(other=[1], a=[1]))
    assert aborted.aborted and aborted.report(format=fmt)


def test_ascii_style_has_no_unicode() -> None:
    assert _mismatching().report(style="ascii").isascii()


def test_print_output(capsys: pytest.CaptureFixture[str]) -> None:
    text = _mismatching().report(print_output=True)
    assert text.strip() in capsys.readouterr().out
    _mismatching().report()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(("ext", "fmt"), [("md", "markdown"), ("html", "html"), ("json", "json"), ("txt", "text")])
def test_save_path_infers_format_and_writes_utf8(tmp_path: Path, ext: str, fmt: str) -> None:
    path = tmp_path / "out" / f"report.{ext}"
    r = _mismatching()
    text = r.report(path=path)
    assert path.read_text(encoding="utf-8") == text
    assert text == r.report(format=fmt)


def test_max_columns_limits_text_but_json_keeps_all() -> None:
    n_cols = 30
    left = df(id=[1, 2], **{f"c{i:02d}": [1, 2] for i in range(n_cols)})
    right = df(id=[1, 2], **{f"c{i:02d}": [1, 3] for i in range(n_cols)})
    r = run(left, right)
    text = r.report(max_columns=5)
    assert "c29" not in text or "c05" not in text  # lista truncada
    assert len(json.loads(r.report(format="json"))["columns"]["results"]) == n_cols


def test_to_json_roundtrip() -> None:
    r = _mismatching()
    data = json.loads(r.result.to_json())
    assert data["rows"]["mismatched"] == r.rows.mismatched
    assert data["is_match"] is False and data["aborted"] is False


def test_csv_and_parquet_sources_agree(tmp_path: Path) -> None:
    left = pl.DataFrame({"id": [1, 2, 3], "a": [1, 2, 3], "s": ["x", "y", "z"]})
    right = pl.DataFrame({"id": [1, 2, 3], "a": [1, 9, 3], "s": ["x", "y", "z"]})
    for ext, write in (("parquet", "write_parquet"), ("csv", "write_csv")):
        getattr(left, write)(tmp_path / f"l.{ext}")
        getattr(right, write)(tmp_path / f"r.{ext}")
        r = run(str(tmp_path / f"l.{ext}"), str(tmp_path / f"r.{ext}"))
        assert (r.rows.common, r.rows.mismatched) == (3, 1), ext


# ----------------------------------------------------------------------------- fontes (io)
def test_unsupported_source_type() -> None:
    with pytest.raises(TypeError, match="fonte não suportada"):
        open_source(123)  # type: ignore[arg-type]


def test_missing_path_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        open_source(tmp_path / "nao_existe.parquet")


def test_unknown_extension_asks_for_format(tmp_path: Path) -> None:
    f = tmp_path / "dados.txt"
    f.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"ReadOptions\(format=...\)"):
        open_source(f)


def test_empty_directory_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="nenhum arquivo de dados"):
        open_source(tmp_path)


def test_directory_ignores_hidden_and_underscore_files(tmp_path: Path) -> None:
    pl.DataFrame({"id": [1, 2]}).write_parquet(tmp_path / "part-0.parquet")
    pl.DataFrame({"id": [99]}).write_parquet(tmp_path / "_temp.parquet")
    (tmp_path / ".hidden").mkdir()
    pl.DataFrame({"id": [98]}).write_parquet(tmp_path / ".hidden" / "x.parquet")
    assert open_source(tmp_path).lazy.collect()["id"].to_list() == [1, 2]


def test_csv_separator_and_forced_format(tmp_path: Path) -> None:
    f = tmp_path / "dados.txt"  # extensão desconhecida: só abre porque o formato foi forçado
    f.write_text("id;valor\n1;10\n2;20\n", encoding="utf-8")
    src = open_source(f, ReadOptions(format="csv", separator=";"))
    data = src.lazy.collect()
    assert data.columns == ["id", "valor"]
    assert data.height == 2


def test_glob_without_matches_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        open_source(str(tmp_path / "*.parquet"))


# ----------------------------------------------------------------------------- SAS
def _pick_key(data: pl.DataFrame) -> str:
    """Primeira coluna sem nulos e com valores únicos (a amostra não precisa ter uma PK declarada)."""
    for name in data.columns:
        s = data[name]
        if s.null_count() == 0 and s.n_unique() == data.height:
            return name
    pytest.fail("a amostra não tem coluna única para usar como chave")


def _pick_float_column(data: pl.DataFrame, key: str) -> str:
    for name, dtype in data.schema.items():
        if name != key and dtype.is_float() and data[name].null_count() < data.height:
            return name
    pytest.fail("a amostra não tem coluna numérica para alterar")


def test_open_sas7bdat_returns_rows() -> None:
    data = dc.read_sas(REAL_SAS)
    assert data.height > 0
    assert data.width > 1


def test_sas7bdat_identical_files_match() -> None:
    """O mesmo arquivo dos dois lados tem de dar igualdade total (nulos, texto, datas e floats)."""
    key = _pick_key(dc.read_sas(REAL_SAS))
    r = run(REAL_SAS, REAL_SAS, join_columns=[key])
    assert not r.aborted, r.fatal_errors
    assert r.is_match
    assert r.rows.left_only == 0 and r.rows.right_only == 0 and r.rows.mismatched == 0


def test_sas7bdat_vs_parquet_detects_single_changed_cell(tmp_path: Path) -> None:
    """SAS de um lado, parquet do outro, com UMA célula alterada: tem de achar exatamente essa."""
    data = dc.read_sas(REAL_SAS)
    key = _pick_key(data)
    col = _pick_float_column(data, key)
    idx = data[col].is_not_null().arg_max()  # primeira linha com valor
    changed = data.with_columns(
        pl.when(pl.int_range(pl.len()) == idx).then(pl.col(col) + 1.0).otherwise(pl.col(col)).alias(col)
    )
    right = tmp_path / "right.parquet"
    changed.write_parquet(right)
    r = run(REAL_SAS, right, join_columns=[key])
    assert not r.aborted, r.fatal_errors
    assert r.rows.common == data.height
    assert r.rows.mismatched == 1
    assert {s.column: s.mismatches for s in r.column_stats} == {col.lower(): 1}


def test_directory_of_sas_files_is_concatenated(tmp_path: Path) -> None:
    n = dc.read_sas(REAL_SAS).height
    shutil.copy(REAL_SAS, tmp_path / "a.sas7bdat")
    (tmp_path / "sub").mkdir()
    shutil.copy(REAL_SAS, tmp_path / "sub" / "b.sas7bdat")
    both = open_source(tmp_path)
    assert both.lazy.collect().height == 2 * n
    one = open_source(tmp_path, ReadOptions(recursive=False))
    assert one.lazy.collect().height == n


def test_glob_of_sas_files_is_expanded(tmp_path: Path) -> None:
    n = dc.read_sas(REAL_SAS).height
    shutil.copy(REAL_SAS, tmp_path / "a.sas7bdat")
    shutil.copy(REAL_SAS, tmp_path / "b.sas7bdat")
    src = open_source(str(tmp_path / "*.sas7bdat"))
    assert src.lazy.collect().height == 2 * n


# ----------------------------------------------------------------------------- opções
@pytest.mark.parametrize(
    "kw",
    [
        {"join_columns": ["id", "ID"]},
        {"join_columns": [" "]},
        {"typo_option": 1},  # extra="forbid"
        {"abs_tol": -1.0},
        {"rel_tol": float("inf")},
        {"abs_tol": {"a": float("nan")}},
        {"window_rows": 0},
        {"window_rows": True},
        {"sample_count": -1},
        {"memory_limit": "weekly"},
    ],
)
def test_invalid_settings_are_rejected(kw: dict[str, Any]) -> None:
    on, opts = options(**kw)
    with pytest.raises((ValidationError, ValueError)):
        dc.compare(df(id=[1]), df(id=[1]), on, **opts)


def test_empty_join_columns_is_rejected() -> None:
    with pytest.raises(ValueError, match="ao menos uma coluna"):
        dc.compare(df(id=[1]), df(id=[1]), [])


def test_comparator_on_key_is_fatal() -> None:
    r = run(df(id=[1], a=[1]), df(id=[1], a=[1]), custom_comparators={"ID": lambda a, b: a == b})
    assert r.aborted


def test_schema_alignment_through_compare() -> None:
    """Os casos de ``align_schemas``: exclusivas, tipos diferentes por política e chave de tipos diferentes."""
    r = run(df(id=[1], a=[1], x=[1]), df(id=[1], a=[1], y=[1]))
    assert r.columns.compared == ["a"]
    assert (r.columns.left_only, r.columns.right_only) == (["x"], ["y"])
    assert run(df(id=[1]), df(other=[1])).fatal_errors[0].count("right") == 1
    left, right = df(id=[1], a=[1]), df(id=[1], a=["1"])
    none = run(left, right, casting="none")
    assert none.columns.compared == []
    assert [(t.column, t.left, t.right) for t in none.columns.type_mismatches] == [("a", "Int64", "String")]
    assert run(left, right, casting="right").columns.compared == ["a"]
    k32 = df(id=[1], a=[1]).with_columns(pl.col("id").cast(pl.Int32))
    assert run(df(id=[1], a=[1]), k32, casting="none").aborted
    assert not run(df(id=[1], a=[1]), k32, casting="left").aborted
    with pytest.raises(ValidationError):
        dc.CompareOptions(case_sensitive_columns="talvez")  # type: ignore[arg-type]
