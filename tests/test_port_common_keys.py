"""Portado do datacompolars 0.4.4 (tests/test_common_keys.py): ``common_keys_only`` (aqui ``only_common_keys``) dá
o mesmo resultado que o modo padrão, em todos os modos; só muda o custo.

Os testes de planejamento do topo (``key_overlap``, ``rows_in_range``, ``plan_windows`` com ``bounds``) são das
funções internas do motor antigo e não têm equivalente; o de ``execution.common_keys_only`` também não (a opção não
vai para o resultado). Em ``test_flag_runs_fewer_windows_when_ids_only_partly_overlap``, as janelas aqui são
planejadas sobre os dois lados inteiros: o teste confere o resultado, não o número de janelas.
"""

import random

import polars as pl
import pytest

from port_helpers import ALL_MODES, EXACT, df, mode_of, run


def summary(r):
    return (
        r.rows.common,
        r.rows.mismatched,
        r.rows.left_only,
        r.rows.right_only,
        r.rows.left_total,
        r.rows.right_total,
        tuple(sorted((s.column, s.mismatches) for s in r.column_stats)),
    )


def small_windows(mode, rows=40):
    """Os modos com janela passam a usar janelas pequenas o bastante para gerar várias; os demais ficam iguais."""
    return {**mode, "window_rows": rows if mode["window_rows"] else None}


# ------------------------------------------------------------------ linhas exclusivas
@pytest.mark.parametrize("mode", ALL_MODES)
def test_only_left_only_right_with_flag(mode):
    r = run(df(id=[1, 2, 3], a=[1, 2, 3]), df(id=[2, 3, 4, 5], a=[2, 3, 4, 5]), common_keys_only=True, **mode)
    assert (r.rows.left_only, r.rows.right_only, r.rows.common, r.rows.mismatched) == (1, 2, 2, 0)
    assert (r.rows.left_total, r.rows.right_total) == (3, 4)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_disjoint_ids_with_flag(mode):
    left = df(id=list(range(10)), a=list(range(10)))
    right = df(id=list(range(100, 110)), a=list(range(10)))
    r = run(left, right, common_keys_only=True, **mode)
    assert (r.rows.common, r.rows.left_only, r.rows.right_only) == (0, 10, 10)
    assert (r.rows.left_total, r.rows.right_total) == (10, 10)
    assert not r.is_match


def test_empty_side_with_flag():
    empty = df(id=[], a=[]).cast({"id": pl.Int64, "a": pl.Int64})
    r = run(df(id=[1, 2], a=[1, 2]), empty, common_keys_only=True)
    assert (r.rows.common, r.rows.left_only, r.rows.right_only) == (0, 2, 0)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_full_overlap_with_flag_is_identical(mode):
    d = df(id=list(range(20)), a=list(range(20)), b=["x"] * 20)
    r = run(d, d, common_keys_only=True, **mode)
    assert r.is_match and r.rows.common == 20


# ------------------------------------------------------------------ nulos, duplicatas, chaves
@pytest.mark.parametrize("mode", ALL_MODES)
def test_null_keys_with_flag(mode):
    left = df(id=[None, 1, 2, 3], a=[5, 6, 7, 8])
    right = df(id=[None, 2, 3, 4], a=[9, 7, 7, 0])
    r = run(left, right, common_keys_only=True, **mode)
    assert (r.rows.common, r.rows.mismatched, r.rows.left_only, r.rows.right_only) == (3, 2, 1, 1)
    assert summary(r) == summary(run(left, right, **mode))


@pytest.mark.parametrize("mode", ALL_MODES)
def test_duplicates_outside_the_overlap_are_still_fatal(mode):
    left = df(id=[1, 1, 10, 11], a=[1, 1, 10, 11])  # a chave 1 repetida cai fora da interseção [10, 11]
    right = df(id=[10, 11, 12], a=[10, 11, 12])
    r = run(left, right, common_keys_only=True, **mode)
    assert r.aborted
    assert any("left" in e.lower() and "duplicate" in e.lower() for e in r.fatal_errors)
    r = run(right, left, common_keys_only=True, **mode)
    assert r.aborted
    assert any("right" in e.lower() and "duplicate" in e.lower() for e in r.fatal_errors)


@pytest.mark.parametrize("mode", EXACT)
def test_composite_keys_with_flag(mode):
    left = df(k1=[1, 1, 2, 3], k2=["a", "b", "a", "z"], v=[10, 20, 30, 1])
    right = df(k1=[1, 1, 2, 9], k2=["a", "b", "a", "q"], v=[10, 99, 30, 5])
    r = run(left, right, join_columns=["k1", "k2"], common_keys_only=True, **mode)
    assert (r.rows.common, r.rows.mismatched, r.rows.left_only, r.rows.right_only) == (3, 1, 1, 1)


def test_non_integer_key_with_flag_still_restricts_to_common_keys():
    left = df(k=["a", "b", "c"], v=[1, 2, 3])
    right = df(k=["b", "c", "d"], v=[2, 30, 4])
    r = run(left, right, join_columns=["k"], common_keys_only=True, window_rows=1)
    assert (r.rows.common, r.rows.mismatched, r.rows.left_only, r.rows.right_only) == (2, 1, 1, 1)


# ------------------------------------------------------------------ janelas
def test_flag_runs_fewer_windows_when_ids_only_partly_overlap():
    left = df(id=list(range(100)), a=list(range(100)))
    right = df(id=list(range(50, 150)), a=list(range(50, 150)))
    base = run(left, right, window_rows=10)
    flagged = run(left, right, window_rows=10, common_keys_only=True)
    assert summary(flagged) == summary(base)
    assert (flagged.rows.common, flagged.rows.left_only, flagged.rows.right_only) == (50, 50, 50)


# ------------------------------------------------------------------ detalhes por coluna
@pytest.mark.parametrize("mode", ALL_MODES)
def test_column_stats_and_samples_with_flag(mode):
    left = df(id=list(range(30)), a=list(range(30)), b=["x"] * 30)
    # b diverge no id 39, que está fora da interseção (10..29) e portanto não pode contar
    right = df(id=list(range(10, 40)), a=[v + 1 if v < 18 else v for v in range(10, 40)], b=["x"] * 29 + ["y"])
    r = run(left, right, common_keys_only=True, sample_count=3, **mode)
    assert (r.rows.common, r.rows.left_only, r.rows.right_only) == (20, 10, 10)
    assert {s.column: s.mismatches for s in r.column_stats} == {"a": 8}
    by_col = {s.column: s for s in r.samples}
    assert by_col["a"].total_mismatches == 8
    assert len(by_col["a"].rows) == 3


# ------------------------------------------------------------------ paridade com o modo padrão
def _frames_with_partial_overlap(n: int = 2_000, seed: int = 7):
    """Esquerda com ids [0, n); direita com ids [n/4, n + n/4): 3/4 de sobreposição contígua."""
    rnd = random.Random(seed)
    shift = n // 4
    total = n + shift

    def col(vals, p_null=0.1):
        return [None if rnd.random() < p_null else v for v in vals]

    universe = {
        "id": list(range(total)),
        "i": col([rnd.randint(0, 10) for _ in range(total)]),
        "f": col([rnd.random() for _ in range(total)]),
        "s": col([rnd.choice("abcde") for _ in range(total)]),
    }
    left = pl.DataFrame({k: v[:n] for k, v in universe.items()})
    mutated = {k: list(v[shift:]) for k, v in universe.items()}
    for _ in range(300):
        r, c = rnd.randrange(total - shift), rnd.choice(["i", "s"])
        mutated[c][r] = rnd.choice([None, "zz" if c == "s" else 999])
    return left, pl.DataFrame(mutated)


def test_flag_gives_the_same_result_as_default_in_every_mode():
    left, right = _frames_with_partial_overlap()
    n = left.height
    results = {}
    for param in ALL_MODES:
        mode = small_windows(mode_of(param), rows=300)
        base = summary(run(left, right, **mode))
        flagged = summary(run(left, right, common_keys_only=True, **mode))
        assert flagged == base, param.id
        results[param.id] = flagged
    assert len(set(results.values())) == 1, results  # e os quatro modos concordam entre si
    common, _, left_only, right_only, *_ = next(iter(results.values()))
    assert (common, left_only, right_only) == (n - n // 4, n // 4, n // 4)
