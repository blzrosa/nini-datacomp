"""Argumentos do bench_hash: janela, saída JSON fora de ``docs/`` e os casos ``NxMw`` / ``NxMwu``."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

import bench_hash
from make_data import Spec, generate, load_expected


def _new(a: int, *, memory_limit: str, window_rows: int | None = None, prefetch: bool = True) -> None: ...


def _old(a: int, *, memory_limit: str) -> None: ...


def test_window_rows_and_out_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # --window-rows chega ao worker, e só como argumento que a versão de compare_rows aceita
    spec: dict[str, Any] = {"left": "l", "right": "r", "memory_limit": "1GB", "abs_tol": 0.0, "window_rows": 5_000}
    assert bench_hash.compare_kwargs(spec, _new)["window_rows"] == 5_000
    assert "window_rows" not in bench_hash.compare_kwargs(spec, _old)
    assert "window_rows" not in bench_hash.compare_kwargs({**spec, "window_rows": None}, _new)

    # --out-json grava onde se pede e não toca em docs/
    root = generate(Spec(500, 3, divergence=0.1), tmp_path / "data", log=lambda _: None)
    exp = load_expected(root)
    seen: list[dict[str, Any]] = []

    def fake_measure(s: dict[str, Any]) -> dict[str, Any]:
        seen.append(s)
        return {
            "seconds": 1.0, "peak_mb": 1.0, "common": exp.common, "left_only": exp.left_only,
            "right_only": exp.right_only, "mismatched": exp.mismatched,
            "columns": {c: n for c, n in exp.column_mismatches.items() if n}, "windows": 1, "timings": {},
        }  # fmt: skip

    docs = tmp_path / "docs"
    out = tmp_path / "out" / "r.json"
    monkeypatch.setattr(bench_hash, "DOCS", docs)
    monkeypatch.setattr(bench_hash, "measure_once", fake_measure)
    monkeypatch.setattr(bench_hash, "case_dataset", lambda case: root)
    monkeypatch.setattr(
        sys, "argv", ["bench_hash", "--cases", "500x3", "--runs", "1", "--window-rows", "123", "--out-json", str(out)]
    )
    bench_hash.main()
    assert seen
    assert all(s["window_rows"] == 123 for s in seen)
    assert not docs.exists()
    assert list(json.loads(out.read_text(encoding="utf-8"))) == ["500x3"]


@pytest.mark.parametrize("flag", [["--stage-divisor", "20"], ["--no-staging"]])
def test_via_compare_rejects_flags_it_would_ignore(
    flag: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["bench_hash", "--cases", "500x3", "--via-compare", *flag])
    with pytest.raises(SystemExit) as exc:
        bench_hash.main()
    assert exc.value.code == 2
    assert flag[0] in capsys.readouterr().err


@pytest.mark.parametrize("flag", [["--stage-divisor", "20"], ["--no-staging"]])
def test_c2_flags_accepted_without_via_compare(
    flag: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = generate(Spec(500, 3, divergence=0.1), tmp_path / "data", log=lambda _: None)
    exp = load_expected(root)
    seen: list[dict[str, Any]] = []

    def fake_measure(s: dict[str, Any]) -> dict[str, Any]:
        seen.append(s)
        return {
            "seconds": 1.0, "peak_mb": 1.0, "common": exp.common, "left_only": exp.left_only,
            "right_only": exp.right_only, "mismatched": exp.mismatched,
            "columns": {c: n for c, n in exp.column_mismatches.items() if n}, "windows": 1, "timings": {},
        }  # fmt: skip

    monkeypatch.setattr(bench_hash, "measure_once", fake_measure)
    monkeypatch.setattr(bench_hash, "case_dataset", lambda case: root)
    monkeypatch.setattr(
        sys, "argv", ["bench_hash", "--cases", "500x3", "--runs", "1", "--out-json", str(tmp_path / "r.json"), *flag]
    )
    bench_hash.main()
    assert seen


def test_combine_peak_inline_is_max_parent() -> None:
    assert bench_hash.combine_peak([(100, 0), (400, 0), (250, 0)], parent_base=300) == 400


def test_combine_peak_subtracts_parent_base() -> None:
    samples = [(300, 0), (300, 500), (300, 900), (300, 100)]
    assert bench_hash.combine_peak(samples, parent_base=300) == 900


def test_combine_peak_counts_parent_growth() -> None:
    assert bench_hash.combine_peak([(300, 600), (350, 900)], parent_base=300) == 950
    # o crescimento só conta na mesma amostra do filho
    assert bench_hash.combine_peak([(300, 900), (350, 600)], parent_base=300) == 900


def test_combine_peak_parent_below_base_not_negative() -> None:
    assert bench_hash.combine_peak([(200, 900), (250, 800)], parent_base=300) == 900


def test_split_rss_ignores_launcher_above_host() -> None:
    procs = [(1, 0, 5), (2, 1, 300)]  # launcher(5) -> host(300)
    assert bench_hash.split_rss(procs, host_pid=2) == (300, 0)


def test_split_rss_counts_worker_and_its_launcher() -> None:
    procs = [(1, 0, 5), (2, 1, 300), (3, 2, 5), (4, 3, 700)]  # launcher -> host -> launcher -> worker
    assert bench_hash.split_rss(procs, host_pid=2) == (300, 705)


def test_split_rss_without_host_pid_is_all_host() -> None:
    procs = [(1, 0, 5), (2, 1, 300)]
    assert bench_hash.split_rss(procs, host_pid=None) == (305, 0)


def test_split_rss_host_gone() -> None:
    assert bench_hash.split_rss([(1, 0, 5), (3, 1, 700)], host_pid=2) == (0, 0)


def test_parse_base_reads_pid_and_rss() -> None:
    assert bench_hash.parse_base("123 456") == (123, 456)
    assert bench_hash.parse_base("") is None
    assert bench_hash.parse_base("abc") is None


def test_unordered_wide_case() -> None:
    spec, unordered = bench_hash.parse_case("22000000x84w")
    assert not unordered
    assert (spec.rows, spec.cols, spec.profile) == (22_000_000, 84, "wide84")
    assert spec.divergence == 0.05
    assert spec.only_left == 0.00027
    assert spec.only_right == 0.0000182
    uspec, uunordered = bench_hash.parse_case("2000000x84wu")
    assert uunordered
    assert (uspec.rows, uspec.cols, uspec.profile) == (2_000_000, 84, "wide84")
    root = bench_hash.case_root("2000000x84w")
    assert bench_hash.case_root("2000000x84wu") == root.with_name(root.name + "_unordered")
    # os casos antigos seguem iguais: ordenado e fora de ordem do perfil padrão
    plain, plain_u = bench_hash.parse_case("1000000x50u")
    assert plain_u
    assert (plain.rows, plain.cols, plain.profile) == (1_000_000, 50, "default")
    assert bench_hash.case_root("1000000x50").name == "1000000x50"
    with pytest.raises(ValueError, match="84"):
        bench_hash.parse_case("1000x10w")


def test_parse_case_text_key() -> None:
    spec, unordered = bench_hash.parse_case("10000000x10t")
    assert not unordered
    assert (spec.rows, spec.cols, spec.profile) == (10_000_000, 10, "default")
    assert spec.label == "10000000x10_texto"
    assert bench_hash.case_root("10000000x10t").name == "10000000x10_texto"
    for bad in ("1000000x50tu", "2000000x84wt"):
        with pytest.raises(ValueError, match="texto"):
            bench_hash.parse_case(bad)
    # os casos antigos seguem iguais
    for case, label, u in (("1000000x50", "1000000x50", False), ("1000000x50u", "1000000x50", True)):
        s, got_u = bench_hash.parse_case(case)
        assert (s.label, got_u) == (label, u)
    w, wu = bench_hash.parse_case("2000000x84wu")
    assert (w.label, wu) == ("2000000x84_wide84", True)
    assert bench_hash.parse_case("2000000x84w")[0].label == "2000000x84_wide84"
