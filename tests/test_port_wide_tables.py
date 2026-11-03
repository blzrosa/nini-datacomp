"""Portado do datacompolars 0.4.4 (tests/test_wide_tables.py): tabelas largas. Toda coluna muda o hash de linha
(qualquer que seja a posição), trocar valores entre colunas é divergência, e as contagens e amostras por coluna não
dependem dos grupos de colunas nem das janelas.
"""

import polars as pl
import pytest

from port_helpers import ALL_MODES, EXACT, run

N_COLS = 120


def _cols(n_rows: int, n_cols: int = N_COLS) -> dict:
    data = {"id": list(range(n_rows))}
    for j in range(n_cols):
        data[f"c{j:03d}"] = [(i * 7 + j) % 11 for i in range(n_rows)]
    return data


def _frame(data: dict) -> pl.DataFrame:
    return pl.DataFrame(data)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_identical_wide(mode):
    d = _frame(_cols(10))
    r = run(d, d, **mode)
    assert r.rows.common == 10
    assert r.rows.mismatched == 0
    assert r.is_match


@pytest.mark.parametrize("col", [0, 1, 2, 3, 4, 59, 60, 61, 117, 118, 119])
@pytest.mark.parametrize("mode", ALL_MODES)
def test_every_column_position_changes_the_result(mode, col):
    left = _cols(6)
    right = _cols(6)
    right[f"c{col:03d}"][3] += 100
    r = run(_frame(left), _frame(right), **mode)
    assert r.rows.mismatched == 1
    assert [(s.column, s.mismatches) for s in r.column_stats] == [(f"c{col:03d}", 1)]


@pytest.mark.parametrize("a,b", [(0, 1), (1, 2), (2, 3), (5, 6), (58, 59), (118, 119)])
@pytest.mark.parametrize("mode", EXACT)
def test_swapping_values_between_two_columns_is_mismatch(mode, a, b):
    left = _cols(6)
    right = _cols(6)
    ca, cb = f"c{a:03d}", f"c{b:03d}"
    left[ca][2], left[cb][2] = 1, 2
    right[ca][2], right[cb][2] = 2, 1
    r = run(_frame(left), _frame(right), **mode)
    assert r.rows.mismatched == 1
    assert sorted(s.column for s in r.column_stats) == [ca, cb]


@pytest.mark.parametrize("mode", ALL_MODES)
def test_counts_and_samples_per_column_across_batches_and_windows(mode):
    left = _cols(10)
    right = _cols(10)
    for j in range(N_COLS):
        for i in (3, 7):
            right[f"c{j:03d}"][i] = left[f"c{j:03d}"][i] + 100
    r = run(_frame(left), _frame(right), **mode)
    assert r.rows.mismatched == 2
    assert len(r.column_stats) == N_COLS
    assert all(s.mismatches == 2 for s in r.column_stats)
    by_col = {t.column: t for t in r.samples}
    assert len(by_col) == N_COLS
    for j in range(N_COLS):
        t = by_col[f"c{j:03d}"]
        assert t.total_mismatches == 2
        want = [[i, left[f"c{j:03d}"][i], right[f"c{j:03d}"][i]] for i in (3, 7)]
        assert [list(row) for row in t.rows] == want


@pytest.mark.parametrize("mode", ALL_MODES)
def test_samples_respect_sample_count_across_windows(mode):
    left = _cols(20, 60)
    right = _cols(20, 60)
    for j in range(60):
        for i in range(20):
            right[f"c{j:03d}"][i] = left[f"c{j:03d}"][i] + 100
    r = run(_frame(left), _frame(right), sample_count=3, **mode)
    assert r.rows.mismatched == 20
    for t in r.samples:
        assert t.total_mismatches == 20
        assert [row[0] for row in t.rows] == [0, 1, 2]  # as menores chaves, mesmo com várias janelas
