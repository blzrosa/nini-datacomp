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
