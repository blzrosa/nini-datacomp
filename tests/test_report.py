"""O relatório (`ComparisonResult.report`) nos quatro formatos."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

import nini_datacomp as dc

LEFT = pl.DataFrame({"id": [1, 2, 3, 4], "v": [1.0, 2.0, 3.0, 4.0], "s": ["a", "b", "c", "d"], "only_l": [0] * 4})
RIGHT = pl.DataFrame({"id": [1, 2, 3, 5], "v": [1.0, 2.5, 3.0, 5.0], "s": ["a", "b", "x|y", "e"]})


@pytest.fixture(scope="module")
def result() -> dc.ComparisonResult:
    return dc.compare(LEFT, RIGHT, "id", names=("sas", "lake"))


def test_text(result: dc.ComparisonResult) -> None:
    text = result.report(style="unicode")
    assert "nini-datacomp Comparison Report" in text
    assert "DIFFERENCES FOUND" in text
    assert "2 of 3 common rows differ" in text
    assert "1 rows only in sas" in text
    assert "Only in sas: only_l" in text
    assert "v (sas)" in text
    assert "s (lake)" in text
    assert all(len(line) <= 100 for line in text.splitlines())


def test_text_ascii_and_width(result: dc.ComparisonResult) -> None:
    text = result.report(style="ascii", width=60)
    assert text.isascii()
    assert "[DIFF] DIFFERENCES FOUND" in text
    assert all(len(line) <= 60 for line in text.splitlines())


def test_markdown(result: dc.ComparisonResult) -> None:
    md = result.report("markdown")
    assert md.startswith("# nini-datacomp Comparison Report")
    assert "| Column | Type | Differences | % of common |  |" in md
    assert "x\\|y" in md  # a barra vertical do valor não quebra a tabela


def test_html(result: dc.ComparisonResult) -> None:
    page = result.report("html")
    assert page.startswith("<!DOCTYPE html>")
    assert "<title>nini-datacomp: sas vs lake</title>" in page
    assert "prefers-color-scheme:dark" in page
    assert "x|y" in page


def test_json(result: dc.ComparisonResult) -> None:
    data = json.loads(result.report("json"))
    assert data["rows"]["mismatched"] == 2
    assert data["columns"]["results"]["v"]["samples"]["rows"] == [[2, 2.0, 2.5]]
    assert data == json.loads(result.to_json())


@pytest.mark.parametrize(
    ("name", "start"),
    [("r.md", "# nini"), ("r.html", "<!DOCTYPE"), ("r.json", "{"), ("r.txt", "="), ("sub/dir/r.log", "=")],
)
def test_format_from_path(result: dc.ComparisonResult, tmp_path: Path, name: str, start: str) -> None:
    path = tmp_path / name
    text = result.report(path=path, style="ascii")
    assert text.startswith(start)
    assert path.read_text(encoding="utf-8") == text


def test_identical_and_aborted() -> None:
    same = dc.compare(LEFT, LEFT, "id")
    assert "IDENTICAL" in same.report()
    assert "Differences by column" not in same.report("markdown")
    dup = dc.compare(pl.DataFrame({"id": [1, 1], "v": [1, 2]}), pl.DataFrame({"id": [1], "v": [1]}), "id")
    text = dup.report(style="ascii")
    assert "COMPARISON ABORTED" in text
    assert "Duplicate join keys in the left table: 1 surplus rows" in text
    assert "ROWS" not in text


def test_limits() -> None:
    n = 30
    left = pl.DataFrame({"id": [1, 2], **{f"c{i:02d}": [0, 0] for i in range(n)}})
    right = pl.DataFrame({"id": [1, 2], **{f"c{i:02d}": [1, 0] for i in range(n)}})
    res = dc.compare(left, right, "id")
    text = res.report(max_columns=5, max_sample_columns=2)
    assert "25 more columns with differences" in text
    assert "Samples for 28 more columns omitted." in text
    assert "Samples for" not in res.report(max_sample_columns=0)
    page = res.report("html")
    assert "c29 (left)" in page  # o HTML traz tudo


def test_details_off_note() -> None:
    res = dc.compare(LEFT, RIGHT, "id", details=False)
    assert "Per-column details were not computed (details=False)." in res.report()


def test_print_output(result: dc.ComparisonResult, capsys: pytest.CaptureFixture[str]) -> None:
    text = result.report("markdown", print_output=True)
    assert capsys.readouterr().out == text + "\n"
    result.report()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"format": "pdf"}, "formato desconhecido"),
        ({"width": 59}, "width"),
        ({"width": 201}, "width"),
        ({"style": "fancy"}, "style desconhecido"),
        ({"max_columns": 0}, "max_columns"),
    ],
)
def test_invalid_report_options(result: dc.ComparisonResult, kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        result.report(**kwargs)  # type: ignore[arg-type]
