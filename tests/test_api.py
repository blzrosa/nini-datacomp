"""A API pública (`compare`, `CompareOptions`, `ComparisonResult`), conferida contra os gabaritos do datacompolars:
contagens, colunas, divergências por coluna e as amostras, valor a valor."""

from __future__ import annotations

import json
import math
from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

import make_golden
import nini_datacomp as dc
from cases import CASES

# as configurações do datacompolars nos nomes do nini-datacomp (a tabela de migração do README)
RENAMED = {
    "casting": "cast", "ignore_spaces": "ignore_whitespace", "ignore_case": "ignore_case", "abs_tol": "abs_tol",
    "rel_tol": "rel_tol",
}  # fmt: skip


def _options(settings: dict[str, Any]) -> dict[str, Any]:
    return {RENAMED[k]: v for k, v in settings.items() if k in RENAMED}


def _plain(value: Any) -> Any:
    """Como o gabarito grava (json com default=str), com NaN comparável (NaN != NaN)."""
    data = json.loads(json.dumps(value, default=str))

    def fix(v: Any) -> Any:
        if isinstance(v, list):
            return [fix(x) for x in v]
        return "NaN" if isinstance(v, float) and math.isnan(v) else v

    return fix(data)


def _skip_decimal_on_old_polars(case: Any) -> None:
    if case.name == "temporal" and pl.__version__.startswith("1.30"):
        pytest.skip(f"Decimal não exportado pelo Polars {pl.__version__} (veja test_arrow)")


@pytest.mark.parametrize("window_rows", [None, 2])
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_golden(case: Any, window_rows: int | None) -> None:
    golden = json.loads((make_golden.GOLDEN / f"{case.name}.json").read_text(encoding="utf-8"))
    left, right = case.build()
    try:
        res = dc.compare(left, right, case.settings["join_columns"], window_rows=window_rows, **_options(case.settings))
    except dc.UnsupportedTypeError:
        _skip_decimal_on_old_polars(case)
        raise
    assert res.aborted == golden["aborted"]
    assert len(res.errors) == golden["fatal_errors"]
    rows = res.rows
    got = [rows.left_total, rows.right_total, rows.common, rows.left_only, rows.right_only, rows.mismatched]
    assert got == list(golden["rows"].values())
    cols = golden["columns"]
    assert list(res.columns.compared) == cols["compared"]
    assert [list(res.columns.left_only), list(res.columns.right_only)] == [cols["left_only"], cols["right_only"]]
    got_types = [[t.column, make_golden._type_name(t.left), make_golden._type_name(t.right)]
                 for t in res.columns.type_mismatches]  # fmt: skip
    assert got_types == cols["type_mismatches"]
    if res.aborted:
        return
    # a mesma ordem do datacompolars: da coluna que mais diverge para a que menos
    assert {c.name: c.mismatches for c in res.columns.mismatched()} == golden["column_mismatches"]
    assert [c.name for c in res.columns.mismatched()] == list(golden["column_mismatches"])
    samples = {name: _plain(col.samples.rows()) for name, col in res.columns.items() if col.samples.height}
    assert samples == _plain(golden["samples"])
    for name, col in res.columns.items():
        keys = list(case.settings["join_columns"])
        assert col.samples.columns == [*keys, f"{name} (left)", f"{name} (right)"]
    assert res.is_match == (
        rows.mismatched == rows.left_only == rows.right_only == 0
        and not (cols["left_only"] or cols["right_only"] or cols["type_mismatches"])
    )


def test_options_as_object_or_kwargs() -> None:
    left = pl.DataFrame({"id": [1, 2], "v": [1.0, 2.0]})
    right = pl.DataFrame({"id": [1, 2], "v": [1.25, 2.0]})
    assert dc.compare(left, right, "id").rows.mismatched == 1
    assert dc.compare(left, right, "id", abs_tol=0.25).rows.mismatched == 0
    opts = dc.CompareOptions(abs_tol=0.25)
    assert dc.compare(left, right, "id", options=opts).rows.mismatched == 0
    assert dc.compare(left, right, "id", options=opts, abs_tol=0.0).rows.mismatched == 1  # o kwarg ganha
    res = dc.compare(left, right, ["id"], abs_tol={"V": 0.25})  # nomes sem caixa, também nas tolerâncias
    assert res.rows.mismatched == 0
    assert res.columns["v"].tolerance == (0.25, 0.0)
    assert res.execution.tolerance_columns == ("v",)


@pytest.mark.parametrize(
    "bad",
    [
        {"abs_tol": -1},
        {"rel_tol": {"a": float("nan")}},
        {"samples": -1},
        {"window_rows": 0},
        {"window_rows": True},
        {"threads": 0},
        {"memory_limit": "muito"},
        {"memory_limit": True},
        {"names": ("a", "a")},
        {"cast": "both"},
        {"nao_existe": 1},
    ],
)
def test_invalid_options(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        dc.CompareOptions(**bad)


def test_on_validation() -> None:
    df = pl.DataFrame({"id": [1]})
    for on in ([], [""], ["id", "ID"]):
        with pytest.raises(ValueError, match="on"):
            dc.compare(df, df, on)


def test_case_insensitive_columns() -> None:
    left = pl.DataFrame({"ID": [1, 2], "Valor": [1, 2], "SoEsquerda": [0, 0]})
    right = pl.DataFrame({"id": [1, 2], "VALOR": [1, 3]})
    res = dc.compare(left, right, "Id")
    assert res.join_columns == ("id",)
    assert res.columns.compared == ("valor",)
    assert res.columns.left_only == ("soesquerda",)
    assert res.rows.mismatched == 1
    assert res.columns["valor"].samples.columns == ["id", "valor (left)", "valor (right)"]
    strict = dc.compare(left, right, "ID", case_sensitive_columns=True)
    assert strict.aborted
    assert strict.errors == ("Join keys missing from the right table: ['ID']",)


def test_name_collision_is_fatal() -> None:
    left = pl.DataFrame({"id": [1], "a": [1], "A": [2]})
    res = dc.compare(left, pl.DataFrame({"id": [1], "a": [1]}), "id")
    assert res.aborted
    assert "'a'" in res.errors[0]
    assert res.rows.common == 0


def test_key_type_mismatch_is_fatal_unless_cast() -> None:
    left = pl.DataFrame({"id": [1, 2], "v": [1, 2]})
    right = pl.DataFrame({"id": [1, 2], "v": [1, 2]}, schema_overrides={"id": pl.Int32})
    res = dc.compare(left, right, "id")
    assert res.errors == ("Join key 'id' has different types: left Int64 vs right Int32",)
    assert not res.is_match
    assert dc.compare(left, right, "id", cast="left").is_match


def test_duplicates_are_fatal() -> None:
    left = pl.DataFrame({"id": [1, 2, 2, 3], "v": [1, 2, 2, 3]})
    right = pl.DataFrame({"id": [1, 2, 3, 3, 3], "v": [1, 2, 3, 3, 3]})
    res = dc.compare(left, right, "id")
    assert res.errors == (
        "Duplicate join keys in the left table: 1 surplus rows",
        "Duplicate join keys in the right table: 2 surplus rows",
    )
    assert res.rows == dc.RowSummary(0, 0, 0, 0, 0, 0)
    assert res.columns["v"].mismatches is None


def test_samples_across_windows_follow_the_key() -> None:
    """Com uma janela por linha, as amostras vêm das primeiras janelas, na ordem da chave, até o limite."""
    left = pl.DataFrame({"id": [5, 3, 1, 4, 2], "v": [50, 30, 10, 40, 20], "s": ["a", "b", "c", "d", "e"]})
    right = pl.DataFrame({"id": [1, 2, 3, 4, 5], "v": [11, 20, 31, 41, 51], "s": ["c", "e", "b", "d", "A"]})
    res = dc.compare(left, right, "id", window_rows=1, samples=2)
    v = res.columns["v"]
    assert v.mismatches == 4
    assert v.samples.rows() == [(1, 10, 11), (3, 30, 31)]
    assert v.samples.schema == pl.Schema({"id": pl.Int64, "v (left)": pl.Int64, "v (right)": pl.Int64})
    assert res.columns["s"].samples.rows() == [(5, "a", "A")]
    assert dc.compare(left, right, "id", samples=0).columns["v"].samples.height == 0
    named = dc.compare(left, right, "id", samples=1, names=("sas", "lake"))
    assert named.columns["v"].samples.columns == ["id", "v (sas)", "v (lake)"]


def test_ignore_case_samples_show_the_original_text() -> None:
    left = pl.DataFrame({"id": [1, 2, 3], "s": ["São Paulo", "a b", "x"]})
    right = pl.DataFrame({"id": [1, 2, 3], "s": ["SÃO PAULO", "AB", "y"]})
    res = dc.compare(left, right, "id", ignore_case=True, ignore_whitespace=True, window_rows=2)
    assert res.rows.mismatched == 1
    assert res.columns["s"].samples.rows() == [(3, "x", "y")]
    hard = dc.compare(left, right, "id", ignore_case=True, verify_hash=True)
    assert hard.rows.mismatched == 2
    assert hard.columns["s"].samples.rows() == [(2, "a b", "AB"), (3, "x", "y")]


def test_details_off() -> None:
    left = pl.DataFrame({"id": [1, 2], "v": [1, 2]})
    right = pl.DataFrame({"id": [1, 2], "v": [1, 3]})
    res = dc.compare(left, right, "id", details=False)
    assert res.rows.mismatched == 1
    assert res.columns["v"].mismatches is None
    assert res.columns["v"].samples.height == 0
    assert not res.execution.details_computed


def test_warnings_and_membership_only() -> None:
    left = pl.DataFrame({"id": [1, 2]})
    right = pl.DataFrame({"id": [2, 3]})
    res = dc.compare(left, right, "id", abs_tol={"x": 1.0})
    assert res.warnings[0].startswith("No common columns")
    assert "abs_tol names columns that are not compared: ['x']" in res.warnings
    assert (res.rows.common, res.rows.left_only, res.rows.right_only) == (1, 1, 1)


def test_to_json_is_plain_json() -> None:
    left = pl.DataFrame({"id": [1, 2], "f": [float("nan"), 1.0], "d": [None, None]}, schema_overrides={"d": pl.Date})
    right = pl.DataFrame({"id": [1, 2], "f": [float("inf"), 1.0], "d": [None, None]}, schema_overrides={"d": pl.Date})
    res = dc.compare(left, right, "id")
    data = json.loads(res.to_json())  # sem NaN nem tipos fora do JSON
    assert data["rows"]["mismatched"] == 1
    assert data["columns"]["results"]["f"]["samples"]["rows"] == [[1, "nan", "inf"]]
    assert data["is_match"] is False
    assert data["execution"]["version"] == dc.__version__


def test_mapping_interface() -> None:
    left = pl.DataFrame({"id": [1, 2], "b": [1, 2], "a": [1, 2]})
    right = pl.DataFrame({"id": [1, 2], "b": [9, 9], "a": [1, 9]})
    cols = dc.compare(left, right, "id").columns
    assert list(cols) == ["b", "a"]
    assert len(cols) == 2
    assert "a" in cols
    assert [c.name for c in cols.mismatched()] == ["b", "a"]
    assert cols["b"].rate == 1.0


@pytest.mark.parametrize("order", ["sorted", "shuffled"])
def test_reader_native_same_result(tmp_path: Any, order: str) -> None:
    n = 12_000
    left = pl.DataFrame(
        {
            "id": pl.int_range(n, eager=True),
            "a": pl.int_range(n, eager=True) * 3,
            "x": pl.Series([float(i) / 7 for i in range(n)]),
            "s": [f"nome-{i % 50}" for i in range(n)],
        }
    )
    right = left.filter(pl.col("id") % 97 != 0).with_columns(
        a=pl.when(pl.col("id") % 31 == 0).then(pl.col("a") + 1).otherwise(pl.col("a")),
        s=pl.when(pl.col("id") % 53 == 0).then(pl.lit("outro")).otherwise(pl.col("s")),
    )
    if order == "shuffled":
        left, right = left.sample(fraction=1.0, shuffle=True, seed=1), right.sample(fraction=1.0, shuffle=True, seed=2)
    paths = []
    for name, df in (("left", left), ("right", right)):
        path = tmp_path / f"{name}.parquet"
        df.write_parquet(path, row_group_size=3_000, statistics=True)
        paths.append(path)

    def run(reader: Any) -> dict[str, Any]:
        res = dc.compare(paths[0], paths[1], "id", read=dc.ReadOptions(reader=reader), window_rows=2_500)
        out = res.to_dict()
        for key in ("execution", "created_at"):
            out.pop(key)
        return out

    assert run("native") == run("polars")
