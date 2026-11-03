"""Portado do datacompolars 0.4.4 (tests/test_composite_keys.py): chave composta com tipos mistos, nulos por
componente, repetições, ordem e caixa das colunas e janelas.

As janelas aqui dividem por qualquer chave (também texto, e também abaixo de um valor da 1ª coluna): nos testes que
lá documentavam o número de janelas da limitação antiga, só o resultado é conferido. Colunas aninhadas são
comparadas pelo valor (fase 6b), em vez de forçar o caminho coluna a coluna.
"""

from datetime import date, datetime

import polars as pl
import pytest

from port_helpers import ALL_MODES, EXACT, df, run

PK = ["k1", "k2"]


@pytest.mark.parametrize("mode", ALL_MODES)
def test_composite_keys_all_modes(mode):
    left = df(k1=[1, 1, 2, 2, 3, 3], k2=["a", "b", "a", "b", "a", "b"], v=[1, 2, 3, 4, 5, 6])
    right = df(k1=[1, 1, 2, 2, 3, 4], k2=["a", "b", "a", "b", "a", "a"], v=[1, 2, 3, 99, 5, 7])
    r = run(left, right, join_columns=PK, **mode)
    assert (r.rows.common, r.rows.mismatched, r.rows.left_only, r.rows.right_only) == (5, 1, 1, 1)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_composite_components_are_not_concatenated(mode):
    # ("a|b", "c") e ("a", "b|c") NÃO são a mesma chave (protege contra um futuro __PK_COMPARE__ por concatenação)
    left = df(k1=["a|b"], k2=["c"], v=[1])
    right = df(k1=["a"], k2=["b|c"], v=[1])
    r = run(left, right, join_columns=PK, **mode)
    assert (r.rows.common, r.rows.left_only, r.rows.right_only) == (0, 1, 1)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_composite_keys_with_nulls_in_either_component(mode):
    left = df(k1=[1, None, 2, None], k2=["a", "b", None, None], v=[1, 2, 3, 4])
    right = df(k1=[1, None, 2, None], k2=["a", "b", None, None], v=[1, 2, 3, 5])
    r = run(left, right, join_columns=PK, **mode)
    assert (r.rows.common, r.rows.mismatched, r.rows.left_only, r.rows.right_only) == (4, 1, 0, 0)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_composite_duplicates_only_when_the_full_key_repeats(mode):
    ok = df(k1=[1, 1], k2=["a", "b"], v=[1, 2])  # k1 repete, a chave completa não
    assert not run(ok, ok, join_columns=PK, **mode).aborted

    dup = df(k1=[1, 1], k2=["a", "a"], v=[1, 2])
    r = run(dup, ok, join_columns=PK, **mode)
    assert r.aborted
    assert any("left" in e.lower() and "duplicate" in e.lower() for e in r.fatal_errors)


def test_key_case_and_order_do_not_change_the_result():
    left = df(K1=[1, 1, 2], K2=["a", "b", "a"], v=[1, 2, 3])
    right = df(k1=[1, 1, 2], k2=["a", "b", "a"], v=[1, 2, 9])
    a = run(left, right, join_columns=["K1", "k2"], window_rows=2)
    b = run(left, right, join_columns=["k2", "K1"], window_rows=2)
    assert (a.rows.common, a.rows.mismatched) == (b.rows.common, b.rows.mismatched) == (3, 1)


def test_type_mismatch_in_second_key_component():
    left = df(k1=[1], k2=[1], v=[1])
    right = df(k1=[1], k2=["1"], v=[1])
    assert run(left, right, join_columns=PK, casting="none").aborted
    assert run(left, right, join_columns=PK, casting="left").rows.common == 1


def test_samples_carry_every_key_column():
    left = df(k1=[1, 1], k2=["a", "b"], v=[1, 2])
    right = df(k1=[1, 1], k2=["a", "b"], v=[1, 3])
    r = run(left, right, join_columns=PK, sample_count=2)
    table = r.samples[0]
    assert table.headers[:2] == PK
    assert table.rows[0][:2] == [1, "b"]


def test_low_cardinality_first_key():
    n = 1_000
    left = df(k1=[i % 2 for i in range(n)], k2=list(range(n)), v=list(range(n)))
    right = left.with_columns(pl.when(pl.col("k2") == 7).then(-1).otherwise(pl.col("v")).alias("v"))
    r = run(left, right, join_columns=PK, window_rows=100)
    assert (r.rows.common, r.rows.mismatched) == (n, 1)


# ------------------------------------------------------------------ hash: outros dtypes e aninhados
@pytest.mark.parametrize("mode", EXACT)
def test_hash_date_datetime_bool(mode):
    left = df(
        id=[1, 2],
        d=[date(2020, 1, 1), None],
        t=[datetime(2020, 1, 1, 1), datetime(2021, 1, 1)],
        b=[True, None],
    )
    assert run(left, left, **mode).rows.mismatched == 0
    changed = left.with_columns(pl.Series("b", [False, None]))
    assert run(left, changed, **mode).rows.mismatched == 1


def test_nested_columns_are_compared():
    d = df(id=[1, 2], items=[[1, 2], [3]])
    assert run(d, d).is_match
