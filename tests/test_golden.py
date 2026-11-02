"""Gabaritos dos casos de borda (`tools/cases.py`, `tools/make_golden.py`), gerados com o datacompolars 0.4.4.

Além de conferir que os JSON estão atualizados, alguns casos têm o resultado conferido à mão aqui: o gabarito não é
aceito só porque o datacompolars o produziu.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

import make_golden
from cases import CASES


def _golden(name: str) -> dict[str, Any]:
    return json.loads((make_golden.GOLDEN / f"{name}.json").read_text(encoding="utf-8"))


def test_every_case_has_a_golden_file() -> None:
    names = [c.name for c in CASES]
    assert len(names) == len(set(names))
    files = {p.stem for p in make_golden.GOLDEN.glob("*.json")}
    assert files == set(names), "rode: uv run python tools/make_golden.py"


def test_golden_files_are_up_to_date() -> None:
    assert make_golden.stale() == [], "rode: uv run python tools/make_golden.py e confira o diff"


def test_columnwise_is_forced() -> None:
    assert make_golden.columnwise({"join_columns": ["id"]})["abs_tol"] == make_golden.FORCE_COLUMNWISE
    assert make_golden.columnwise({"abs_tol": 0.5})["abs_tol"] == 0.5
    assert make_golden.columnwise({"abs_tol": {"a": 0.5}})["abs_tol"] == {"a": 0.5, **make_golden.FORCE_COLUMNWISE}
    for case in CASES:
        assert _golden(case.name)["path"] in ("columnwise", None)  # None: abortado antes de escolher o caminho


def test_basic_by_hand() -> None:
    g = _golden("basic")
    assert g["rows"] == {
        "left_total": 8,
        "right_total": 8,
        "common": 7,
        "left_only": 1,
        "right_only": 1,
        "mismatched": 6,  # ids 2 (i), 3 (s), 4 (d), 5 (b), 6 (i nulo de um lado só), 7 (f)
    }
    assert g["column_mismatches"] == {"i": 2, "f": 1, "s": 1, "b": 1, "d": 1}


def test_shuffled_is_the_same_as_basic() -> None:
    a, b = _golden("basic"), _golden("shuffled")
    assert a["rows"] == b["rows"]
    assert a["column_mismatches"] == b["column_mismatches"]
    assert a["samples"] == b["samples"]  # amostras na ordem da chave, qualquer que seja a ordem das linhas


def test_null_keys_by_hand() -> None:
    g = _golden("null_keys")
    assert g["rows"]["common"] == 3  # nulo, 1, 2
    assert g["rows"]["left_only"] == 1
    assert g["rows"]["right_only"] == 1
    assert g["rows"]["mismatched"] == 1  # "nulo" x "nulo!"


def test_floats_by_hand() -> None:
    g = _golden("floats")
    # iguais: 0.0 x -0.0 (as duas ordens), NaN x NaN de outro sinal e de outro payload, inf x inf, 1.0 x 1.0
    # diferentes: -inf x inf, nulo x 1.0, 1e-300 x 0.0, 0.1 + 0.2 x 0.3
    assert g["column_mismatches"]["f64"] == 4


@pytest.mark.parametrize("name", ["duplicate_keys"])
def test_fatal(name: str) -> None:
    g = _golden(name)
    assert g["aborted"] is True
    assert g["fatal_errors"] >= 1


def test_tolerance_edges_by_hand() -> None:
    # a: |1 - 1,5| = 0,5 <= 0,5 (igual); |1 - 1,75| = 0,75 (diferente); |1 - 0,5| = 0,5 (igual); 100 x 100,5 (igual)
    assert _golden("tolerance_abs")["column_mismatches"].get("a", 0) == 1
    # r: |8 - 9| = 1 <= 0,125 * 9 (igual); |8 - 7| = 1 > 0,875 (diferente); |8 - 10| = 2 > 1,25 (diferente)
    assert _golden("tolerance_rel")["column_mismatches"].get("r", 0) == 2


def test_wide_by_hand() -> None:
    g = _golden("wide_300")
    assert g["rows"]["mismatched"] == 3  # linhas 3, 10 e 39
    assert g["column_mismatches"] == {"c000": 1, "c007": 1, "c150": 1, "c299": 1}
