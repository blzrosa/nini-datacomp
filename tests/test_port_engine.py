"""Portado do datacompolars 0.4.4 (tests/test_engine.py): o comportamento do motor, em todos os modos (hash e coluna
a coluna, janela única e várias janelas), pelo adaptador `port_helpers`.

Fora daqui, por serem diferenças de propósito ou detalhes internos do motor antigo (tabela de migração do README):
``test_duplicate_check_can_be_disabled`` (``check_duplicate_keys`` não existe: chave repetida sempre aborta),
``test_path_selection`` (não há "caminho" único: cada coluna vai pelo hash ou coluna a coluna) e
``test_columns_per_batch_does_not_change_result`` (não há lotes de colunas para escolher). Em
``test_non_integer_first_key_falls_back_to_single_window``, a chave de texto também é dividida em janelas aqui: o
teste confere o resultado, não o número de janelas.
"""

import random

import polars as pl
import pytest

from port_helpers import ALL_MODES, EXACT, NAN, df, mode_of, run


# ------------------------------------------------------------------ nulos
@pytest.mark.parametrize("mode", ALL_MODES)
def test_identical(mode):
    d = df(id=[1, 2, 3], a=[1, 2, 3], b=["x", "y", "z"])
    r = run(d, d, **mode)
    assert r.rows.common == 3
    assert r.rows.mismatched == 0
    assert r.is_match


@pytest.mark.parametrize("mode", ALL_MODES)
def test_null_on_one_side_is_mismatch(mode):
    assert run(df(id=[1, 2, 3], a=[1, None, 3]), df(id=[1, 2, 3], a=[1, 2, 3]), **mode).rows.mismatched == 1
    assert run(df(id=[1, 2, 3], a=[1, 2, 3]), df(id=[1, 2, 3], a=[1, None, 3]), **mode).rows.mismatched == 1


@pytest.mark.parametrize("mode", ALL_MODES)
def test_null_on_both_sides_is_match(mode):
    left = df(id=[1, 2], a=[1, None], b=[None, "x"])
    assert run(left, left, **mode).rows.mismatched == 0


@pytest.mark.parametrize("mode", ALL_MODES)
def test_string_null_vs_value(mode):
    r = run(df(id=[1, 2], s=["a", None]), df(id=[1, 2], s=["a", "b"]), **mode)
    assert r.rows.mismatched == 1


@pytest.mark.parametrize("mode", EXACT)
def test_null_in_one_column_not_masked_by_other_columns(mode):
    r = run(df(id=[1, 2], a=[None, 5], b=[10, 20]), df(id=[1, 2], a=[7, 5], b=[10, 20]), **mode)
    assert r.rows.mismatched == 1


# ------------------------------------------------------------------ hash: casos críticos
@pytest.mark.parametrize("mode", EXACT)
def test_swapping_values_between_columns_is_mismatch(mode):
    left = df(id=[1, 2, 3], a=[1, 5, 7], b=[2, 6, 8])
    right = df(id=[1, 2, 3], a=[2, 5, 7], b=[1, 6, 8])
    assert run(left, right, **mode).rows.mismatched == 1


@pytest.mark.parametrize("mode", EXACT)
def test_same_value_in_different_columns_is_distinguished(mode):
    assert run(df(id=[1], a=[7], b=[0]), df(id=[1], a=[0], b=[7]), **mode).rows.mismatched == 1


@pytest.mark.parametrize("mode", EXACT)
def test_many_columns_detects_single_change(mode):
    n_cols = 200
    base = {f"c{i:03d}": list(range(i, i + 50)) for i in range(n_cols)}
    changed = {k: list(v) for k, v in base.items()}
    changed["c199"][10] = -1
    changed["c000"][20] = -1
    r = run(df(id=list(range(50)), **base), df(id=list(range(50)), **changed), **mode)
    assert r.rows.mismatched == 2


# ------------------------------------------------------------------ NaN / -0.0 / categóricos
@pytest.mark.parametrize("mode", EXACT)
def test_nan_equals_nan_regardless_of_sign(mode):
    assert run(df(id=[1, 2], x=[NAN, 1.0]), df(id=[1, 2], x=[-NAN, 1.0]), **mode).rows.mismatched == 0


@pytest.mark.parametrize("mode", EXACT)
def test_nan_vs_value_is_mismatch(mode):
    assert run(df(id=[1, 2], x=[NAN, 1.0]), df(id=[1, 2], x=[2.0, 1.0]), **mode).rows.mismatched == 1


@pytest.mark.parametrize("mode", EXACT)
def test_negative_zero_equals_zero(mode):
    assert run(df(id=[1], x=[-0.0]), df(id=[1], x=[0.0]), **mode).rows.mismatched == 0


@pytest.mark.parametrize("mode", EXACT)
def test_float_null_vs_nan_is_mismatch(mode):
    left = df(id=[1], x=[None]).with_columns(pl.col("x").cast(pl.Float64))
    assert run(left, df(id=[1], x=[NAN]), **mode).rows.mismatched == 1


@pytest.mark.parametrize("mode", EXACT)
def test_multiple_float_columns_and_float32(mode):
    left = df(id=[1, 2, 3], x=[1.0, NAN, -0.0], y=[2.5, 3.5, NAN], z=[0.1, 0.2, 0.3])
    same = df(id=[1, 2, 3], x=[1.0, -NAN, 0.0], y=[2.5, 3.5, NAN], z=[0.1, 0.2, 0.3])
    diff = df(id=[1, 2, 3], x=[1.0, NAN, -0.0], y=[2.5, 3.5, 9.9], z=[0.1, 0.2, 0.3])
    assert run(left, same, **mode).rows.mismatched == 0
    assert run(left, diff, **mode).rows.mismatched == 1

    f32 = left.with_columns(pl.col("x", "y", "z").cast(pl.Float32))
    assert run(f32, f32, **mode).rows.mismatched == 0


@pytest.mark.parametrize("mode", EXACT)
def test_categorical_across_frames(mode):
    def cat(values):
        return df(id=[1, 2], c=values).with_columns(pl.col("c").cast(pl.Categorical))

    assert run(cat(["a", "b"]), cat(["b", "a"]), **mode).rows.mismatched == 2
    assert run(cat(["a", "b"]), cat(["a", "b"]), **mode).rows.mismatched == 0


# ------------------------------------------------------------------ linhas exclusivas
@pytest.mark.parametrize("mode", ALL_MODES)
def test_only_left_only_right(mode):
    r = run(df(id=[1, 2, 3], a=[1, 2, 3]), df(id=[2, 3, 4, 5], a=[2, 3, 4, 5]), **mode)
    assert (r.rows.left_only, r.rows.right_only, r.rows.common, r.rows.mismatched) == (1, 2, 2, 0)
    assert (r.rows.left_total, r.rows.right_total) == (3, 4)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_empty_intersection(mode):
    r = run(df(id=[1, 2], a=[1, 2]), df(id=[3, 4], a=[1, 2]), **mode)
    assert (r.rows.common, r.rows.left_only, r.rows.right_only) == (0, 2, 2)


# ------------------------------------------------------------------ chaves
@pytest.mark.parametrize("mode", ALL_MODES)
def test_duplicate_keys_are_fatal(mode):
    left = df(id=[1, 1, 1, 2], a=[1, 1, 1, 2])
    right = df(id=[1, 2], a=[1, 2])
    r = run(left, right, **mode)
    assert r.aborted
    assert any("left" in e.lower() and "duplicate" in e.lower() for e in r.fatal_errors)
    r = run(right, left, **mode)
    assert r.aborted
    assert any("right" in e.lower() and "duplicate" in e.lower() for e in r.fatal_errors)


@pytest.mark.parametrize("mode", EXACT)
def test_key_dtype_mismatch_with_casting(mode):
    left = df(id=[1, 2], a=[1, 2]).with_columns(pl.col("id").cast(pl.Int64))
    right = df(id=[1, 2], a=[1, 2]).with_columns(pl.col("id").cast(pl.Int32))
    assert run(left, right, casting="left", **mode).rows.common == 2


def test_key_dtype_mismatch_without_casting_is_fatal():
    left = df(id=[1, 2], a=[1, 2]).with_columns(pl.col("id").cast(pl.Int64))
    right = df(id=[1, 2], a=[1, 2]).with_columns(pl.col("id").cast(pl.Int32))
    assert run(left, right, casting="none").aborted


def test_missing_join_key_is_fatal():
    assert run(df(id=[1], a=[1]), df(other=[1], a=[1])).aborted


@pytest.mark.parametrize("mode", EXACT)
def test_null_keys_match_each_other(mode):
    left = df(id=[None, 1], a=[5, 6])
    r = run(left, left, **mode)
    assert (r.rows.common, r.rows.mismatched) == (2, 0)


def test_null_keys_with_windows():
    ids = [None] + list(range(11))  # a PK nula não pode sumir quando há várias janelas
    left = df(id=ids, a=list(range(12)))
    right = df(id=ids, a=list(range(11)) + [99])
    r = run(left, right, window_rows=3)
    assert r.execution.windows > 1
    assert (r.rows.common, r.rows.mismatched) == (12, 1)


def test_non_integer_first_key_is_windowed_too():
    left = df(k=["a", "b", "c"], v=[1, 2, 3])
    r = run(left, left, join_columns=["k"], window_rows=1)
    assert r.is_match


@pytest.mark.parametrize("mode", EXACT)
def test_composite_keys(mode):
    left = df(k1=[1, 1, 2], k2=["a", "b", "a"], v=[10, 20, 30])
    right = df(k1=[1, 1, 2], k2=["a", "b", "a"], v=[10, 99, 30])
    r = run(left, right, join_columns=["k1", "k2"], **mode)
    assert (r.rows.common, r.rows.mismatched) == (3, 1)


# ------------------------------------------------------------------ schema
def test_column_names_are_lowercased():
    r = run(df(ID=[1, 2], A=[1, 2]), df(id=[1, 2], a=[1, 3]), join_columns=["Id"])
    assert r.rows.mismatched == 1


def test_dtype_mismatch_column_is_reported_and_skipped():
    r = run(df(id=[1, 2], a=[1, 2], b=[1, 2]), df(id=[1, 2], a=[1, 2], b=["1", "2"]))
    assert [t.column for t in r.columns.type_mismatches] == ["b"]
    assert r.columns.compared == ["a"]
    assert not r.is_match


def test_exclusive_columns_are_reported():
    r = run(df(id=[1], a=[1], x=[1]), df(id=[1], a=[1], y=[1]))
    assert (r.columns.left_only, r.columns.right_only) == (["x"], ["y"])
    assert not r.is_match


def test_no_common_columns_besides_key():
    r = run(df(id=[1, 2], a=[1, 2]), df(id=[1, 2], b=[1, 2]))
    assert r.columns.compared == []
    assert (r.rows.common, r.rows.mismatched) == (2, 0)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_forced_windows_split_the_work(mode):
    n = 50
    left = df(id=list(range(n)), a=list(range(n)))
    right = df(id=list(range(n)), a=[x if x % 10 else -1 for x in range(n)])
    r = run(left, right, **{**mode, "window_rows": 7})
    assert r.execution.windows > 1
    assert (r.rows.common, r.rows.mismatched) == (50, 5)


def test_abs_and_rel_tolerance():
    left = df(id=[1, 2, 3, 4], a=[1.0, 2.0, None, 100.0])
    right = df(id=[1, 2, 3, 4], a=[1.05, 2.5, None, 100.5])
    assert run(left, right, abs_tol=0.1).rows.mismatched == 2
    assert run(left, right, abs_tol={"a": 0.1}).rows.mismatched == 2
    assert run(left, right, rel_tol=0.01).rows.mismatched == 2  # só 100 -> 100.5 passa (0,5%)
    assert run(left, right, abs_tol=0.1, rel_tol=0.01).rows.mismatched == 1


def test_ignore_case_and_spaces():
    left = df(id=[1, 2, 3], s=["ABC", "x y", "q"])
    right = df(id=[1, 2, 3], s=["abc", "xy", "z"])
    assert run(left, right, ignore_case=True).rows.mismatched == 2
    assert run(left, right, ignore_case=True, ignore_spaces=True).rows.mismatched == 1


@pytest.mark.filterwarnings("ignore::nini_datacomp.PerformanceWarning")  # esperado: comparador em Python
def test_custom_comparator_is_used():
    class AlwaysEqual:  # um comparador do datacompolars (BaseComparator): compare(col_l, col_r, **kwargs)
        def compare(self, col_l, col_r, **kwargs):
            return pl.col(col_l).is_not_null() | pl.col(col_l).is_null()

    r = run(df(id=[1, 2], a=[1, 2]), df(id=[1, 2], a=[9, 9]), custom_comparators={"a": AlwaysEqual()})
    assert r.rows.mismatched == 0


# ------------------------------------------------------------------ detalhes por coluna e amostras
@pytest.mark.parametrize("mode", ALL_MODES)
def test_column_stats_and_samples(mode):
    left = df(id=list(range(20)), a=list(range(20)), b=["x"] * 20)
    right = df(id=list(range(20)), a=[v + 1 if v < 8 else v for v in range(20)], b=["x"] * 19 + ["y"])
    r = run(left, right, sample_count=3, **mode)
    stats = {s.column: s.mismatches for s in r.column_stats}
    assert stats == {"a": 8, "b": 1}
    by_col = {s.column: s for s in r.samples}
    assert by_col["a"].total_mismatches == 8
    assert len(by_col["a"].rows) == 3  # respeita sample_count
    assert len(by_col["b"].rows) == 1


def test_column_details_off_still_counts_rows():
    left = df(id=[1, 2], a=[1, 2])
    r = run(left, df(id=[1, 2], a=[1, 3]), column_details=False)
    assert r.rows.mismatched == 1
    assert r.column_stats == [] and r.samples == []
    assert not r.execution.details_computed


def test_sample_count_zero_disables_samples():
    r = run(df(id=[1, 2], a=[1, 2]), df(id=[1, 2], a=[1, 3]), sample_count=0)
    assert r.samples == []
    assert r.rows.mismatched == 1


# ------------------------------------------------------------------ paridade entre todos os modos
def _random_frames(n: int = 5_000, seed: int = 42):
    rnd = random.Random(seed)

    def col(vals, p_null=0.1):
        return [None if rnd.random() < p_null else v for v in vals]

    base = {
        "id": list(range(n)),
        "i": col([rnd.randint(0, 10) for _ in range(n)]),
        "f": col([rnd.random() for _ in range(n)]),
        "s": col([rnd.choice("abcde") for _ in range(n)]),
    }
    mutated = {k: list(v) for k, v in base.items()}
    for _ in range(400):
        r, c = rnd.randrange(n), rnd.choice(["i", "f", "s"])
        mutated[c][r] = rnd.choice([None, mutated[c][r]])
    for _ in range(400):
        mutated["i"][rnd.randrange(n)] = rnd.randint(100, 200)
    right = pl.DataFrame(mutated).filter(pl.col("id") % 97 != 0)
    return pl.DataFrame(base), right


def test_all_modes_agree_on_random_data():
    left, right = _random_frames()
    results = {}
    for param in ALL_MODES:
        mode = mode_of(param)
        r = run(left, right, **{**mode, "window_rows": 1_200 if mode["window_rows"] else None})
        results[param.id] = (
            r.rows.common,
            r.rows.mismatched,
            r.rows.left_only,
            r.rows.right_only,
            tuple(sorted((s.column, s.mismatches) for s in r.column_stats)),  # hash e columnwise batem por coluna
        )
    assert len(set(results.values())) == 1, results
