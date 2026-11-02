"""cases.py -- casos de borda pequenos e determinísticos, com os gabaritos em ``tests/golden/<nome>.json``.

Cada caso tem as duas tabelas (montadas aqui, sem arquivos) e as configurações no formato do datacompolars 0.4.4, que
gera o gabarito (`make_golden.py`). Os testes do nini-datacomp vão rodar os mesmos casos e comparar com o gabarito.

Mudou um caso? Rode ``uv run python tools/make_golden.py`` e confira o diff dos JSON antes de commitar.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

import numpy as np
import polars as pl

__all__ = ["CASES", "Case"]


@dataclass(frozen=True)
class Case:
    name: str
    description: str
    build: Callable[[], tuple[pl.DataFrame, pl.DataFrame]]
    settings: dict[str, Any] = field(default_factory=dict)  # CompareSettings do datacompolars (com join_columns)


def _nan(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def _basic() -> tuple[pl.DataFrame, pl.DataFrame]:
    left = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "i": [10, 20, None, 40, 50, 60, 70, 80],
            "f": [1.5, 2.5, 3.5, None, 5.5, 6.5, 7.5, 8.5],
            "s": ["a", "b", "c", "d", None, "f", "g", "h"],
            "b": [True, False, True, None, True, False, True, False],
            "d": [date(2026, 1, d) for d in range(1, 9)],
        }
    )
    right = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 9],
            "i": [10, 21, None, 40, 50, None, 70, 90],  # 2: valor; 6: nulo de um lado só
            "f": [1.5, 2.5, 3.5, None, 5.5, 6.5, 7.25, 9.5],  # 7: valor
            "s": ["a", "b", "C", "d", None, "f", "g", "i"],  # 3: caixa
            "b": [True, False, True, None, False, False, True, True],  # 5: valor
            "d": [date(2026, 1, d) for d in (1, 2, 3, 5, 5, 6, 7, 9)],  # 4: data
        }
    )
    return left, right


def _shuffled() -> tuple[pl.DataFrame, pl.DataFrame]:
    left, right = _basic()
    return left.sample(fraction=1.0, shuffle=True, seed=1), right.sample(fraction=1.0, shuffle=True, seed=2)


def _composite() -> tuple[pl.DataFrame, pl.DataFrame]:
    left = pl.DataFrame({"k1": [1, 1, 2, 2, 3], "k2": ["a", "b", "a", "b", "a"], "v": [1.0, 2.0, 3.0, 4.0, 5.0]})
    right = pl.DataFrame({"k1": [1, 1, 2, 3, 3], "k2": ["a", "b", "b", "a", "b"], "v": [1.0, 2.5, 4.0, 5.0, 6.0]})
    return left, right


def _text_key() -> tuple[pl.DataFrame, pl.DataFrame]:
    left = pl.DataFrame({"codigo": ["A-01", "A-02", "B-10", "ção", ""], "v": [1, 2, 3, 4, 5]})
    right = pl.DataFrame({"codigo": ["A-02", "A-01", "B-11", "ção", ""], "v": [2, 1, 3, 40, 5]})
    return left, right


def _sparse() -> tuple[pl.DataFrame, pl.DataFrame]:
    ids_l = [-5, 0, 1, 10, 1_000_000, 10**12, 2**62]
    ids_r = [-5, 1, 10, 999_999, 1_000_000, 10**12, 2**62]
    left = pl.DataFrame({"id": ids_l, "v": [1, 2, 3, 4, 5, 6, 7]})
    right = pl.DataFrame({"id": ids_r, "v": [1, 3, 4, 0, 5, 60, 7]})
    return left, right


def _null_keys() -> tuple[pl.DataFrame, pl.DataFrame]:
    schema = {"id": pl.Int64, "v": pl.String}
    left = pl.DataFrame({"id": [None, 1, 2, 3], "v": ["nulo", "a", "b", "c"]}, schema=schema)
    right = pl.DataFrame({"id": [1, None, 2, 4], "v": ["a", "nulo!", "b", "d"]}, schema=schema)
    return left, right


def _duplicates() -> tuple[pl.DataFrame, pl.DataFrame]:
    left = pl.DataFrame({"id": [1, 2, 2, 3], "v": [1, 2, 2, 3]})
    right = pl.DataFrame({"id": [1, 2, 3], "v": [1, 2, 3]})
    return left, right


def _empty_left() -> tuple[pl.DataFrame, pl.DataFrame]:
    right = pl.DataFrame({"id": [1, 2, 3], "v": [1.0, 2.0, 3.0]})
    return right.clear(), right


def _no_common() -> tuple[pl.DataFrame, pl.DataFrame]:
    return pl.DataFrame({"id": [1, 2, 3], "v": [1, 2, 3]}), pl.DataFrame({"id": [4, 5], "v": [4, 5]})


def _floats() -> tuple[pl.DataFrame, pl.DataFrame]:
    inf = float("inf")
    left_v = [0.0, -0.0, _nan(0x7FF8000000000000), _nan(0x7FF8000000000000), inf, -inf, 1.0, None, 1e-300, 0.1 + 0.2]
    right_v = [-0.0, 0.0, _nan(0xFFF8000000000000), _nan(0x7FF8000000000001), inf, inf, 1.0, 1.0, 0.0, 0.3]
    ids = list(range(len(left_v)))
    left = pl.DataFrame({"id": ids, "f64": left_v, "f32": left_v}, schema_overrides={"f32": pl.Float32})
    right = pl.DataFrame({"id": ids, "f64": right_v, "f32": right_v}, schema_overrides={"f32": pl.Float32})
    return left, right


def _text() -> tuple[pl.DataFrame, pl.DataFrame]:
    left_s = ["São Paulo", "praia grande", "a b", "x ", " nbsp", "", None, "emoji 😀", "um texto bem maior que doze"]
    right_s = ["SÃO PAULO", "Praia Grande", "ab", "x", "nbsp", None, None, "emoji 😀", "um texto bem maior que doze!"]
    ids = list(range(len(left_s)))
    return pl.DataFrame({"id": ids, "s": left_s}), pl.DataFrame({"id": ids, "s": right_s})


def _categorical() -> tuple[pl.DataFrame, pl.DataFrame]:
    left = pl.DataFrame({"id": [1, 2, 3], "c": ["x", "y", "z"]}, schema_overrides={"c": pl.Categorical})
    right = pl.DataFrame({"id": [1, 2, 3], "c": ["x", "y", "w"]})
    return left, right


def _int_widths() -> tuple[pl.DataFrame, pl.DataFrame]:
    left = pl.DataFrame({"id": [1, 2, 3], "v": [1, 2, 3]}, schema_overrides={"v": pl.Int32})
    right = pl.DataFrame({"id": [1, 2, 3], "v": [1, 2, 4]}, schema_overrides={"v": pl.Int64})
    return left, right


def _tolerance() -> tuple[pl.DataFrame, pl.DataFrame]:
    # diferenças exatas em binário (potências de 2): a borda da tolerância não depende de arredondamento
    left = pl.DataFrame({"id": [1, 2, 3, 4, 5], "a": [1.0, 1.0, 1.0, 100.0, None], "r": [8.0, 8.0, 8.0, 0.0, 1.0]})
    right = pl.DataFrame({"id": [1, 2, 3, 4, 5], "a": [1.5, 1.75, 0.5, 100.5, None], "r": [9.0, 7.0, 10.0, 0.0, 1.0]})
    return left, right


def _wide() -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(7)
    data = {f"c{j:03d}": rng.integers(0, 5, 40) for j in range(300)}
    left = pl.DataFrame({"id": np.arange(40), **data})
    changed = {k: v.copy() for k, v in data.items()}
    for j, row in [(0, 3), (150, 3), (299, 39), (7, 10)]:
        changed[f"c{j:03d}"][row] += 1
    right = pl.DataFrame({"id": np.arange(40), **changed})
    return left, right


def _temporal() -> tuple[pl.DataFrame, pl.DataFrame]:
    left = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "dt": [datetime(2026, 1, 1, 12), datetime(2026, 1, 2), None],
            "du": [timedelta(seconds=5), timedelta(0), timedelta(days=1)],
            "tm": [time(12, 30), time(0), time(23, 59, 59)],
            "dec": [Decimal("1.25"), Decimal("2.50"), None],
        },
        schema_overrides={"dec": pl.Decimal(10, 2)},
    )
    right = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "dt": [datetime(2026, 1, 1, 12), datetime(2026, 1, 2, 0, 0, 1), None],
            "du": [timedelta(seconds=5), timedelta(microseconds=1), timedelta(days=1)],
            "tm": [time(12, 30), time(0), time(23, 59, 58)],
            "dec": [Decimal("1.25"), Decimal("2.51"), None],
        },
        schema_overrides={"dec": pl.Decimal(10, 2)},
    )
    return left, right


ID = ["id"]
CASES: tuple[Case, ...] = (
    Case("basic", "tipos básicos com nulos, divergências e linhas exclusivas", _basic, {"join_columns": ID}),
    Case("shuffled", "o mesmo do basic, com as linhas fora de ordem nos dois lados", _shuffled, {"join_columns": ID}),
    Case("composite_key", "chave composta (inteiro + texto)", _composite, {"join_columns": ["k1", "k2"]}),
    Case("text_key", "chave de texto (inclui vazio e acentos)", _text_key, {"join_columns": ["codigo"]}),
    Case("sparse_keys", "ids esparsos, negativos e enormes", _sparse, {"join_columns": ID}),
    Case("null_keys", "chave nula casa com chave nula", _null_keys, {"join_columns": ID}),
    Case("duplicate_keys", "chave repetida à esquerda: erro fatal", _duplicates, {"join_columns": ID}),
    Case("empty_left", "lado esquerdo vazio", _empty_left, {"join_columns": ID}),
    Case("no_common_keys", "nenhuma chave em comum", _no_common, {"join_columns": ID}),
    Case("floats", "-0.0 = 0.0, NaN = NaN (qualquer payload), infinitos, nulo x valor", _floats, {"join_columns": ID}),
    Case("text_exact", "texto: caixa, espaços, NBSP, vazio x nulo, emoji, longo", _text, {"join_columns": ID}),
    Case(
        "text_ignore",
        "o mesmo texto com ignore_case e ignore_spaces",
        _text,
        {"join_columns": ID, "ignore_case": True, "ignore_spaces": True},
    ),
    Case("categorical_vs_string", "Categorical x String, sem conversão", _categorical, {"join_columns": ID}),
    Case(
        "categorical_cast_right",
        "Categorical x String convertendo a esquerda para o tipo da direita",
        _categorical,
        {"join_columns": ID, "casting": "right"},
    ),
    Case("int_widths_none", "Int32 x Int64 sem conversão", _int_widths, {"join_columns": ID}),
    Case(
        "int_widths_left",
        "Int32 x Int64, direita convertida para Int32",
        _int_widths,
        {"join_columns": ID, "casting": "left"},
    ),
    Case("tolerance_abs", "tolerância absoluta exatamente na borda", _tolerance, {"join_columns": ID, "abs_tol": 0.5}),
    Case(
        "tolerance_rel", "tolerância relativa: |l - r| <= rel * |r|", _tolerance, {"join_columns": ID, "rel_tol": 0.125}
    ),
    Case(
        "tolerance_per_column",
        "tolerância só numa coluna",
        _tolerance,
        {"join_columns": ID, "abs_tol": {"a": 0.5}, "rel_tol": {"r": 0.125}},
    ),
    Case("wide_300", "300 colunas, 4 células diferentes", _wide, {"join_columns": ID}),
    Case("temporal", "Datetime, Duration, Time e Decimal", _temporal, {"join_columns": ID}),
)
