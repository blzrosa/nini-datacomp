"""``isolation``: quando a comparação roda num processo à parte (``_worker.py``) e que o resultado é o mesmo."""

from __future__ import annotations

import os
import pickle
import tempfile
import warnings
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

import nini_datacomp as dc
from nini_datacomp import pipeline
from nini_datacomp.isolation import inline_reason, use_process

HERE = Path(__file__).parent


def double_is_equal(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    """Comparador próprio de módulo (picklável por referência): vale no worker com o ``tests/`` no PYTHONPATH."""
    return left == right


@pytest.fixture
def pair(tmp_path: Path) -> tuple[Path, Path]:
    n = 3_000
    left = pl.DataFrame({"id": range(n), "v": [i % 7 for i in range(n)], "s": [f"t{i}" for i in range(n)]})
    right = left.with_columns(
        pl.when(pl.col("id") % 500 == 3).then(99).otherwise(pl.col("v")).alias("v"),
    ).filter(pl.col("id") != 10)
    lp, rp = tmp_path / "left.parquet", tmp_path / "right.parquet"
    left.write_parquet(lp, row_group_size=1_000)
    right.write_parquet(rp, row_group_size=1_000)
    return lp, rp


# ----------------------------------------------------------------------------- decisão
@pytest.mark.parametrize(
    ("isolation", "allocator", "fits", "reason", "expected"),
    [
        ("auto", "tuned", False, None, False),
        ("auto", "tuned", True, None, False),
        ("auto", "default", True, None, False),
        ("auto", "default", False, None, True),
        ("auto", "default", False, "fonte em memória", False),
        ("process", "default", True, None, True),
        ("process", "tuned", True, None, True),
        ("process", "default", False, "fonte em memória", False),
        ("inline", "default", False, None, False),
        ("inline", "tuned", True, None, False),
    ],
)
def test_isolation_decision(isolation: str, allocator: str, fits: bool, reason: str | None, expected: bool) -> None:
    assert use_process(isolation, allocator, fits, reason) is expected  # type: ignore[arg-type]


def test_inline_reasons(tmp_path: Path) -> None:
    opts = dc.CompareOptions()
    df = pl.DataFrame({"id": [1]})
    assert inline_reason("a.parquet", "b.parquet", opts, environ={}, frozen=False) is None
    assert inline_reason(Path("a"), "b/", opts, environ={}, frozen=False) is None
    assert inline_reason(df, "b.parquet", opts, environ={}, frozen=False)
    assert inline_reason("a.parquet", df.lazy(), opts, environ={}, frozen=False)
    assert inline_reason("a", "b", opts, environ={"NINI_DATACOMP_WORKER": "1"}, frozen=False)
    assert inline_reason("a", "b", opts, environ={}, frozen=True)
    unpicklable = dc.CompareOptions(comparators={"v": lambda a, b: a == b})
    assert inline_reason("a", "b", unpicklable, environ={}, frozen=False)
    assert (
        inline_reason("a", "b", dc.CompareOptions(comparators={"v": double_is_equal}), environ={}, frozen=False) is None
    )


def test_isolation_option_validation() -> None:
    assert dc.CompareOptions().isolation == "auto"
    for value in ("auto", "inline", "process"):
        assert dc.CompareOptions(isolation=value).isolation == value  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        dc.CompareOptions(isolation="thread")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        dc.CompareOptions(isolation=True)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- worker
def _same(a: dc.ComparisonResult, b: dc.ComparisonResult) -> None:
    da, db = a.to_dict(), b.to_dict()
    for d in (da, db):  # o que muda de uma execução para outra
        d.pop("created_at", None)
        execution = d.get("execution", {})
        for k in ("total_seconds", "timings", "notes"):
            execution.pop(k, None)
    assert da == db
    for c in a.columns:
        assert a.columns[c].samples.equals(b.columns[c].samples)


def test_worker_matches_inline(pair: tuple[Path, Path]) -> None:
    # memory_limit fixo: o "auto" depende da memória livre de cada processo
    inline = dc.compare(*pair, on="id", isolation="inline", window_rows=700, memory_limit="1GB")
    isolated = dc.compare(*pair, on="id", isolation="process", window_rows=700, memory_limit="1GB")
    assert inline.rows.mismatched > 0
    assert inline.rows.left_only == 1
    _same(inline, isolated)
    assert "spawn" in isolated.execution.timings
    assert "spawn" not in inline.execution.timings
    assert any("processo isolado" in note for note in isolated.execution.notes)
    assert not any("processo isolado" in note for note in inline.execution.notes)
    assert isolated.execution.total_seconds >= isolated.execution.timings["spawn"] > 0


def test_worker_error_same_type(tmp_path: Path) -> None:
    missing = tmp_path / "nao_existe.parquet"
    with pytest.raises(Exception) as inline:  # noqa: PT011 - o tipo é justamente o que se compara
        dc.compare(missing, missing, on="id", isolation="inline")
    with pytest.raises(Exception) as isolated:  # noqa: PT011
        dc.compare(missing, missing, on="id", isolation="process")
    assert type(isolated.value) is type(inline.value)
    assert str(isolated.value) == str(inline.value)


def test_worker_reemits_warnings(pair: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(HERE), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep))
    kwargs = {"on": "id", "comparators": {"v": double_is_equal}, "window_rows": 1_000}
    with pytest.warns(dc.PerformanceWarning, match="comparadores em Python") as inline:
        dc.compare(*pair, isolation="inline", **kwargs)
    with pytest.warns(dc.PerformanceWarning, match="comparadores em Python") as isolated:
        result = dc.compare(*pair, isolation="process", **kwargs)
    assert [str(w.message) for w in isolated if w.category is dc.PerformanceWarning] == [
        str(w.message) for w in inline if w.category is dc.PerformanceWarning
    ]
    assert result.execution.notes


def test_worker_cleans_temp_dir(pair: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    dc.compare(*pair, on="id", isolation="process")
    assert list(scratch.iterdir()) == []
    with pytest.raises(Exception):  # noqa: B017, PT011 - o tipo vem do worker
        dc.compare(tmp_path / "x.parquet", tmp_path / "y.parquet", on="id", isolation="process")
    assert list(scratch.iterdir()) == []


def test_worker_start_failure_falls_back_inline(pair: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess  # noqa: PLC0415

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("sem processos")

    monkeypatch.setattr(subprocess, "run", boom)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # sem aviso: só a nota
        result = dc.compare(*pair, on="id", isolation="process", window_rows=700)
    assert result.rows.mismatched > 0
    assert any("próprio processo" in note for note in result.execution.notes)
    assert "spawn" not in result.execution.timings


@pytest.mark.skipif(dc.ALLOCATOR_STATUS != "default", reason="o auto só isola com o alocador padrão")
def test_auto_goes_to_worker_when_row_group_does_not_fit(tmp_path: Path) -> None:
    n = 20_000
    df = pl.DataFrame({"id": range(n), **{f"s{j}": [f"{i:06d}" + "x" * 500 for i in range(n)] for j in range(4)}})
    path = tmp_path / "wide.parquet"
    df.write_parquet(path, row_group_size=5_000)
    auto = dc.compare(path, path, on="id", memory_limit="600MiB")
    assert auto.is_match
    assert "spawn" in auto.execution.timings
    assert any("processo isolado" in note for note in auto.execution.notes)
    # com um DataFrame de um lado, nunca isola (e diz por quê)
    mixed = dc.compare(path, df, on="id", memory_limit="600MiB")
    assert mixed.is_match
    assert "spawn" not in mixed.execution.timings
    assert any("próprio processo" in note for note in mixed.execution.notes)
    inline = dc.compare(path, path, on="id", memory_limit="600MiB", isolation="inline")
    assert "spawn" not in inline.execution.timings


@pytest.mark.skipif(dc.ALLOCATOR_STATUS != "default", reason="o auto só isola com o alocador padrão")
def test_auto_goes_to_worker_when_staging_is_needed(tmp_path: Path) -> None:
    n = 20_000
    df = pl.DataFrame({"id": range(n), "v": [i % 11 for i in range(n)]}).sample(fraction=1.0, shuffle=True, seed=3)
    path = tmp_path / "shuffled.parquet"
    df.write_parquet(path, row_group_size=100, statistics=True)  # 200 row groups, ids espalhados em todos
    kwargs = {"on": "id", "window_rows": 500, "memory_limit": "1GB"}  # 40 janelas: amplificação real > 15
    auto = dc.compare(path, path, **kwargs)
    inline = dc.compare(path, path, isolation="inline", **kwargs)
    assert auto.is_match
    assert "spawn" in auto.execution.timings
    assert "spawn" not in inline.execution.timings
    for result in (auto, inline):
        assert any("fontes reparticionadas por janela" in note for note in result.execution.notes)
    assert (auto.rows.common, auto.rows.left_only, auto.rows.right_only) == (
        inline.rows.common,
        inline.rows.left_only,
        inline.rows.right_only,
    )
    # com um lado DataFrame o staging segue neste processo
    mixed = dc.compare(path, df, **kwargs)
    assert mixed.is_match
    assert "spawn" not in mixed.execution.timings


@pytest.mark.skipif(dc.ALLOCATOR_STATUS != "default", reason="o auto só isola com o alocador padrão")
def test_auto_skips_host_plan_when_schema_proves_no_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    n = 200_000
    path = tmp_path / "ints.parquet"
    pl.DataFrame({"id": range(n), **{f"c{j}": range(n) for j in range(9)}}).write_parquet(path, row_group_size=n)

    def boom(*_: object, **__: object) -> None:
        raise AssertionError("o host não deve planejar")

    monkeypatch.setattr(pipeline, "plan_windows", boom)
    result = dc.compare(path, path, on="id", memory_limit="600MiB")
    assert result.is_match
    assert "spawn" in result.execution.timings


def test_result_pickles(pair: tuple[Path, Path]) -> None:
    result = dc.compare(*pair, on="id", isolation="inline", window_rows=700)
    again = pickle.loads(pickle.dumps(result))
    assert again.to_dict() == result.to_dict()
    assert again.columns["v"].samples.equals(result.columns["v"].samples)
    assert again.created_at == result.created_at
