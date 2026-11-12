"""O caminho completo: janelas lidas pelo Polars, chaves alinhadas no C++, contagens somadas (`count_rows`, fase 1) e
linhas divergentes pelo hash de linha (`compare_rows`, fase 2). Conferido contra os gabaritos do datacompolars e contra
o esperado pela construção dos dados."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import polars as pl
import pytest

import make_data
import make_golden
from cases import CASES
from make_data import Spec, generate, load_expected
from nini_datacomp import io, pipeline, windows
from nini_datacomp.errors import UnsupportedTypeError
from nini_datacomp.io import open_source, partition_source
from nini_datacomp.pipeline import WindowReader, compare_rows, count_rows, prepare_keys
from nini_datacomp.readers import ReadOptions, stage_partitioned
from nini_datacomp.windows import plan_windows


@pytest.mark.parametrize("window_rows", [None, 2])
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_golden_row_counts(case: Any, window_rows: int | None) -> None:
    """Com uma janela só e com janelas de 2 linhas, as contagens batem com o gabarito do datacompolars."""
    golden = json.loads((make_golden.GOLDEN / f"{case.name}.json").read_text(encoding="utf-8"))
    left, right = case.build()
    res = count_rows(left, right, case.settings["join_columns"], window_rows=window_rows)
    if golden["aborted"]:
        assert res.aborted
        return
    rows = golden["rows"]
    got = (res.left_total, res.right_total, res.common, res.left_only, res.right_only)
    want = (rows["left_total"], rows["right_total"], rows["common"], rows["left_only"], rows["right_only"])
    assert got == want


SETTINGS = {"casting", "abs_tol", "rel_tol", "ignore_case", "ignore_spaces"}  # as do datacompolars que valem aqui


@pytest.mark.parametrize("window_rows", [None, 2])
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_golden_mismatched_rows(case: Any, window_rows: int | None) -> None:
    """Linhas divergentes, divergências por coluna, colunas comparadas e tipos divergentes iguais aos do gabarito, com
    uma janela e com janelas de 2 linhas (inclusive tolerâncias e opções de texto)."""
    golden = json.loads((make_golden.GOLDEN / f"{case.name}.json").read_text(encoding="utf-8"))
    left, right = case.build()
    options = {k: v for k, v in case.settings.items() if k in SETTINGS}
    try:
        res = compare_rows(left, right, case.settings["join_columns"], window_rows=window_rows, **options)
    except UnsupportedTypeError as e:
        if "Decimal" in str(e) or "dec" in str(e):  # o Polars 1.30 não exporta Decimal (veja test_arrow)
            pytest.skip(f"Decimal não exportado pelo Polars {pl.__version__}")
        raise
    if golden["aborted"]:
        assert res.aborted
        assert res.mismatched == 0
        return
    assert res.mismatched == golden["rows"]["mismatched"]
    assert res.equal == golden["rows"]["common"] - golden["rows"]["mismatched"]
    # o datacompolars lista só as colunas com divergência; o nini-datacomp lista todas as comparadas (as outras com 0)
    assert {c: n for c, n in res.column_mismatches.items() if n} == golden["column_mismatches"]
    assert set(res.column_mismatches) == set(res.compared_columns)
    assert list(res.compared_columns) == golden["columns"]["compared"]
    got = [[c, make_golden._type_name(a), make_golden._type_name(b)] for c, a, b in res.type_mismatches]
    assert got == golden["columns"]["type_mismatches"]
    verified = compare_rows(left, right, case.settings["join_columns"], verify_hash=True, **options)
    assert verified.mismatched == res.mismatched
    assert verified.column_mismatches == res.column_mismatches
    assert verified.hash_collisions == 0


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    mp = pytest.MonkeyPatch()
    mp.setattr(make_data, "CELLS_PER_FILE", 13 * 4_000)  # 12 colunas + id: arquivos de 4 mil linhas
    try:
        spec = Spec(20_000, 12, divergence=0.05, only_left=0.02, only_right=0.02, row_group_size=1_000)
        return generate(spec, tmp_path_factory.mktemp("data"), log=lambda _: None)
    finally:
        mp.undo()


def _check(res: Any, root: Path) -> None:
    exp = load_expected(root)
    assert not res.aborted
    assert (res.left_total, res.right_total) == (exp.left_rows, exp.right_rows)
    assert (res.common, res.left_only, res.right_only) == (exp.common, exp.left_only, exp.right_only)


@pytest.mark.parametrize("prefetch", [True, False])
@pytest.mark.parametrize("window_rows", [None, 3_000, 777])
def test_parquet_dataset(dataset: Path, window_rows: int | None, prefetch: bool) -> None:
    res = count_rows(dataset / "left", dataset / "right", ["id"], window_rows=window_rows, prefetch=prefetch)
    _check(res, dataset)
    if window_rows:
        assert res.windows > 1
        # Parquet em ordem: merge. Na última janela, os arquivos extra-* (ids acima de todos) vêm antes dos part-* na
        # ordem dos nomes, então esse lado chega fora de ordem e vai pelo hash
        assert res.strategies.get("merge", 0) >= res.windows - 1
    if window_rows == 3_000:
        assert res.aligned_to_row_groups


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_engine(dataset: Path, engine: Any) -> None:
    res = count_rows(dataset / "left", dataset / "right", ["id"], window_rows=3_000, engine=engine)
    _check(res, dataset)


def test_memory_limit_makes_windows(dataset: Path) -> None:
    res = count_rows(dataset / "left", dataset / "right", ["id"], memory_limit="64MiB")
    _check(res, dataset)
    assert res.memory_limit == 64 * 1024**2
    assert res.window_bytes > 0


def test_only_key_columns(dataset: Path) -> None:
    res = count_rows(dataset / "left", dataset / "right", ["id"], columns=[], window_rows=5_000)
    _check(res, dataset)


def test_text_key(dataset: Path) -> None:
    """A mesma tabela com a chave em texto (fora de ordem numérica) e embaralhada: as janelas funcionam igual."""

    def text(side: str) -> pl.DataFrame:
        df = pl.read_parquet(dataset / side)
        df = df.with_columns(pl.format("chave-{}", pl.col("id")).alias("id"))
        return df.sample(fraction=1.0, shuffle=True, seed=7)

    res = count_rows(text("left"), text("right"), ["id"], window_rows=2_500)
    _check(res, dataset)
    assert res.windows > 1
    assert res.strategies.get("hash", 0) > 0  # embaralhada: hash join


def test_composite_key(dataset: Path) -> None:
    def composite(side: str) -> pl.DataFrame:
        df = pl.read_parquet(dataset / side)
        return df.with_columns((pl.col("id") % 7).alias("k1"), (pl.col("id") // 7).alias("k2")).drop("id")

    res = count_rows(composite("left"), composite("right"), ["k1", "k2"], window_rows=1_500)
    _check(res, dataset)


def test_duplicates_abort() -> None:
    left = pl.DataFrame({"id": [1, 2, 2, 3, 4, 5], "v": range(6)})
    right = pl.DataFrame({"id": [1, 2, 3, 3, 3, 6], "v": range(6)})
    res = count_rows(left, right, ["id"], window_rows=2)
    assert res.aborted
    assert (res.dup_left, res.dup_right) == (1, 2)
    assert res.common == 0


def test_categorical_key() -> None:
    left = pl.DataFrame({"k": ["a", "b", "c", "d"]}).with_columns(pl.col("k").cast(pl.Categorical))
    right = pl.DataFrame({"k": ["b", "c", "e"]})
    res = count_rows(left, right, ["k"], window_rows=2)
    assert (res.common, res.left_only, res.right_only) == (2, 2, 1)


def test_key_errors() -> None:
    a = pl.DataFrame({"id": [1, 2]})
    with pytest.raises(ValueError, match="faltam"):
        count_rows(a, pl.DataFrame({"x": [1]}), ["id"])
    with pytest.raises(ValueError, match="tipos diferentes"):
        count_rows(a, pl.DataFrame({"id": ["1", "2"]}), ["id"])


@pytest.mark.parametrize("window_rows", [None, 3_000])
def test_mismatched_rows_on_the_dataset(dataset: Path, window_rows: int | None) -> None:
    exp = load_expected(dataset)
    res = compare_rows(dataset / "left", dataset / "right", ["id"], window_rows=window_rows, threads=4)
    _check(res, dataset)
    assert res.mismatched == exp.mismatched
    assert res.equal == exp.common - exp.mismatched
    assert {c: n for c, n in res.column_mismatches.items() if n} == {
        c: n for c, n in exp.column_mismatches.items() if n
    }
    assert res.timings["hash"] > 0
    verified = compare_rows(dataset / "left", dataset / "right", ["id"], window_rows=window_rows, verify_hash=True)
    assert (verified.mismatched, verified.hash_collisions) == (exp.mismatched, 0)


def test_compare_rows_columns_and_casting() -> None:
    left = pl.DataFrame({"id": [1, 2, 3], "a": [1, 2, 3], "w": [1, 2, 3], "c": ["x", "y", "z"], "only_l": [0, 0, 0]})
    right = pl.DataFrame(
        {"id": [1, 2, 3], "a": [1, 2, 4], "w": [1, 2, 3], "c": ["x", "q", "z"], "only_r": [0, 0, 0]},
        schema_overrides={"w": pl.Int32},
    )
    left = left.with_columns(pl.col("c").cast(pl.Categorical))
    right = right.with_columns(pl.col("c").cast(pl.Categorical))
    res = compare_rows(left, right, ["id"])
    assert res.compared_columns == ("a", "c")  # w: Int64 x Int32, fora; only_l/only_r: de um lado só
    assert res.type_mismatches == (("w", "Int64", "Int32"),)
    assert res.mismatched == 2  # id 2 (c, Categorical comparado como texto) e id 3 (a)
    cast = compare_rows(left, right, ["id"], casting="left")
    assert cast.compared_columns == ("a", "w", "c")
    assert cast.mismatched == 2
    only_a = compare_rows(left, right, ["id"], columns=["a"])
    assert (only_a.compared_columns, only_a.mismatched) == (("a",), 1)
    with pytest.raises(ValueError, match="não estão nos dois lados"):
        compare_rows(left, right, ["id"], columns=["only_l"])
    with pytest.raises(ValueError, match="casting"):
        compare_rows(left, right, ["id"], casting="both")  # type: ignore[arg-type]


def test_compare_rows_key_cast_and_nested() -> None:
    left = pl.DataFrame({"id": [1, 2], "v": [1, 2]}, schema_overrides={"id": pl.Int32})
    right = pl.DataFrame({"id": [1, 2], "v": [1, 3]})
    with pytest.raises(ValueError, match="tipos diferentes"):
        compare_rows(left, right, ["id"])
    assert compare_rows(left, right, ["id"], casting="right").mismatched == 1
    nested = pl.DataFrame({"id": [1], "l": [[1, 2]]})
    assert compare_rows(nested, nested, ["id"]).mismatched == 0  # aninhadas: pelo hash do valor (fase 6b)


def test_chunked_windows_are_hashed_without_copies() -> None:
    """Frames em vários pedaços de memória (como o Polars entrega uma janela de vários row groups): mesmo resultado,
    lido pedaço a pedaço."""

    def side(seed: int) -> pl.DataFrame:
        parts = [
            pl.DataFrame({"id": range(i, i + 1_000), "v": [(j * seed) % 7 for j in range(1_000)]})
            for i in range(0, 5_000, 1_000)
        ]
        return pl.concat(parts, rechunk=False)

    left, right = side(1), side(1).with_columns(pl.when(pl.col("id") % 500 == 0).then(99).otherwise("v").alias("v"))
    right = pl.concat([right.slice(0, 2_500), right.slice(2_500)], rechunk=False)
    res = compare_rows(left, right, ["id"])
    assert res.mismatched == 10
    assert res.hash_pieces >= 5 + 2
    assert compare_rows(left.rechunk(), right.rechunk(), ["id"]).mismatched == 10


@pytest.mark.parametrize("groups", [2, 5, 50])
def test_column_groups_give_the_same_result(dataset: Path, groups: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cada janela lida em grupos de colunas (o hash continua de um grupo para o outro): mesmo resultado, também com
    o verify_hash (que confere célula a célula, grupo a grupo)."""
    from nini_datacomp import pipeline  # noqa: PLC0415

    real = plan_windows  # o original (o pipeline o importou de windows)

    def plan_in_groups(*args: Any, **kwargs: Any) -> Any:
        return dataclasses.replace(real(*args, **kwargs), column_groups=groups)

    exp = load_expected(dataset)
    monkeypatch.setattr(pipeline, "plan_windows", plan_in_groups)
    res = compare_rows(dataset / "left", dataset / "right", ["id"], window_rows=3_000)
    assert res.column_groups == min(groups, len(res.compared_columns))
    assert res.mismatched == exp.mismatched
    verified = compare_rows(dataset / "left", dataset / "right", ["id"], window_rows=3_000, verify_hash=True)
    assert (verified.mismatched, verified.hash_collisions) == (exp.mismatched, 0)
    # com tolerância (as numéricas comparadas uma a uma, em todos os pares), como o caminho columnwise do baseline
    tol = compare_rows(dataset / "left", dataset / "right", ["id"], window_rows=3_000, abs_tol=1e-12)
    assert tol.mismatched == exp.mismatched
    assert tol.column_mismatches == res.column_mismatches


def test_tolerance_and_text_options() -> None:
    left = pl.DataFrame({"id": [1, 2, 3, 4], "f": [1.0, 2.0, 3.0, 4.0], "s": ["Ana", "b c", "X", "y"]})
    right = pl.DataFrame({"id": [1, 2, 3, 4], "f": [1.4, 2.6, 3.0, 4.0], "s": ["ana", "bc", "X ", "Y"]})
    exact = compare_rows(left, right, ["id"])
    assert exact.column_mismatches == {"f": 2, "s": 4}
    assert exact.mismatched == 4
    loose = compare_rows(left, right, ["id"], abs_tol={"f": 0.5}, ignore_case=True, ignore_spaces=True)
    assert loose.column_mismatches == {"f": 1, "s": 0}  # 2,6 - 2 = 0,6 > 0,5
    assert loose.mismatched == 1
    rel = compare_rows(left, right, ["id"], rel_tol=0.25, ignore_case=True)
    # |l - r| <= 0,25 * |r|: 0,4 > 0,35 (diverge), 0,6 <= 0,65; "b c"/"bc" e "X"/"X " seguem diferentes
    assert rel.column_mismatches == {"f": 1, "s": 2}


def test_count_rows_has_no_mismatched(dataset: Path) -> None:
    res = count_rows(dataset / "left", dataset / "right", ["id"], window_rows=5_000)
    assert res.mismatched is None
    assert res.equal is None


def test_window_reader_keeps_two_windows(dataset: Path) -> None:
    """A leitura em paralelo entrega as janelas na ordem do plano, com as colunas pedidas."""
    ls, rs = open_source(dataset / "left"), open_source(dataset / "right")
    ls, rs, dtypes = prepare_keys(ls, rs, ["id"])
    plan = plan_windows(ls, rs, ["id"], window_rows=4_000)
    reader = WindowReader(ls, rs, plan, ["id"], dtypes, columns_left=["aa"], columns_right=["aa"])
    seen = [(w.index, lf.columns, rf.height) for w, lf, rf in reader]
    assert [s[0] for s in seen] == [w.index for w in plan.windows]
    assert all(s[1] == ["id", "aa"] for s in seen)
    assert "read_wait" in reader.timer.seconds


# ----------------------------------------------------------------------------- amplificação de leitura e staging
def _write_pair(tmp_path: Path, *, shuffled: bool, n: int = 4000, groups: int = 100) -> tuple[Path, Path]:
    """Duas tabelas em Parquet com vários row groups: a direita perde os 100 primeiros ids, ganha 100 novos no fim e
    tem ``v`` diferente nos ids múltiplos de 7."""
    left = pl.DataFrame({"id": range(n), "v": [i * 3 for i in range(n)]})
    right = pl.DataFrame({"id": range(100, n + 100), "v": [i * 3 + (i % 7 == 0) for i in range(100, n + 100)]})
    if shuffled:
        left, right = left.sample(fraction=1.0, shuffle=True, seed=1), right.sample(fraction=1.0, shuffle=True, seed=2)
    lp, rp = tmp_path / "left.parquet", tmp_path / "right.parquet"
    left.write_parquet(lp, row_group_size=groups, statistics=True)
    right.write_parquet(rp, row_group_size=groups, statistics=True)
    return lp, rp


def test_read_amplification_sorted(tmp_path: Path) -> None:
    lp, rp = _write_pair(tmp_path, shuffled=False)
    plan = plan_windows(open_source(lp), open_source(rp), ["id"], window_rows=1000)
    assert len(plan.windows) > 2
    assert plan.read_amplification == pytest.approx((1.0, 1.0), abs=0.05)


def test_read_amplification_shuffled(tmp_path: Path) -> None:
    lp, rp = _write_pair(tmp_path, shuffled=True)
    plan = plan_windows(open_source(lp), open_source(rp), ["id"], window_rows=1000)
    assert len(plan.windows) >= 4
    assert all(a >= len(plan.windows) * 0.8 for a in plan.read_amplification)


def test_read_amplification_without_statistics_is_nan() -> None:
    df = pl.DataFrame({"id": range(4000)})
    plan = plan_windows(open_source(df), open_source(df), ["id"], window_rows=1000)
    assert all(math.isnan(a) for a in plan.read_amplification)  # não é Parquet


def test_partitioned_staging_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import gc  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    temp = tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    monkeypatch.setattr(pipeline, "PARTITION_THRESHOLD", 3.0)  # amplificação ~4 aqui: abaixo do limiar de produção
    lp, rp = _write_pair(tmp_path, shuffled=True)
    got = compare_rows(lp, rp, ["id"], window_rows=1000)
    gc.collect()
    assert any("reparticionadas por janela" in n for n in got.notes)
    assert list(temp.iterdir()) == []  # a pasta temporária sumiu
    assert got.windows > 2

    base = compare_rows(pl.read_parquet(lp), pl.read_parquet(rp), ["id"], window_rows=1000)  # em memória: sem staging
    assert not any("reparticionadas" in n for n in base.notes)
    for field in ("left_total", "right_total", "common", "left_only", "right_only", "mismatched"):
        assert getattr(got, field) == getattr(base, field), field
    assert (got.common, got.left_only, got.right_only) == (3900, 100, 100)
    assert got.mismatched == len([i for i in range(100, 4000) if i % 7 == 0])


def test_stage_sides_match_repartition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`_stage_sides` é True exatamente nos lados que `_repartition` troca por fonte em staging."""
    lp, rp = _write_pair(tmp_path, shuffled=True)
    (tmp_path / "ord").mkdir()
    ordered_l, _ = _write_pair(tmp_path / "ord", shuffled=False)

    def sides(left: Path, right: Path, window_rows: int) -> tuple[tuple[bool, bool], tuple[bool, bool]]:
        ls, rs = open_source(left), open_source(right)
        plan = plan_windows(ls, rs, ["id"], window_rows=window_rows)
        want = pipeline._stage_sides(plan, ls, rs)
        _new_plan, nl, nr, owned = pipeline._repartition(plan, ls, rs, ["id"], None)
        try:
            return want, (nl is not ls, nr is not rs)
        finally:
            for src in owned:
                src.cleanup()

    monkeypatch.setattr(pipeline, "PARTITION_THRESHOLD", 3.0)
    want, got = sides(lp, rp, 1000)
    assert want == got == (True, True)
    # um lado em ordem tem amplificação ~1: só o fora de ordem é trocado
    want, got = sides(ordered_l, rp, 1000)
    assert want == got
    assert want[1]
    assert not want[0]
    # 2 janelas: sem staging
    want, got = sides(lp, rp, 2000)
    assert want == got == (False, False)
    # amplificação abaixo do limiar
    monkeypatch.setattr(pipeline, "PARTITION_THRESHOLD", 1000.0)
    want, got = sides(lp, rp, 1000)
    assert want == got == (False, False)


def test_sliced_read_same_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A leitura em fatias de row groups dá as mesmas contagens da comparação em memória, com `steps` (cast) e 2
    arquivos por lado; `Source.scan` recebe `rows`."""
    n = 4000
    for side, offset in (("l", 0), ("r", 100)):
        folder = tmp_path / side
        folder.mkdir()
        df = pl.DataFrame(
            {
                "id": range(offset, n + offset),
                "w": [i * 3 + (i % 7 == 0) * (side == "r") for i in range(offset, n + offset)],
            }
        )
        df = df.sample(fraction=1.0, shuffle=True, seed=5 if side == "l" else 6)
        if side == "r":
            df = df.with_columns(pl.col("w").cast(pl.Int32))
        df.slice(0, n // 2).write_parquet(folder / "a.parquet", row_group_size=100, statistics=True)
        df.slice(n // 2).write_parquet(folder / "b.parquet", row_group_size=100, statistics=True)
    lp, rp = tmp_path / "l", tmp_path / "r"

    seen: list[tuple[int, int] | None] = []
    real_scan = io.Source.scan

    def spy(self: Any, files: Any, rows: Any = None) -> Any:
        seen.append(rows)
        return real_scan(self, files, rows)

    monkeypatch.setattr(io.Source, "scan", spy)
    base = compare_rows(
        pl.read_parquet(lp / "*.parquet"), pl.read_parquet(rp / "*.parquet"), ["id"], window_rows=1000, casting="left"
    )
    assert base.mismatched is not None
    assert base.mismatched > 0
    for share in (1e-9, 1e9):  # m = 1 (uma fatia por row group) e fatia que cobre o arquivo inteiro
        monkeypatch.setattr(windows, "SLICE_SHARE", share)
        seen.clear()
        got = compare_rows(lp, rp, ["id"], window_rows=1000, casting="left")
        assert any("em fatias" in note for note in got.notes)
        slices = [r for r in seen if r is not None]
        assert slices
        if share < 1:
            assert {n for _, n in slices} == {100}
        else:
            assert {a for a, _ in slices} == {0}
        for field in ("left_total", "right_total", "common", "left_only", "right_only", "mismatched"):
            assert getattr(got, field) == getattr(base, field), field
        assert got.column_mismatches == base.column_mismatches


def test_repartition_clears_slice_sides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lp, rp = _write_pair(tmp_path, shuffled=True)
    (tmp_path / "ord").mkdir()
    ordered_l, _ = _write_pair(tmp_path / "ord", shuffled=False)
    monkeypatch.setattr(pipeline, "PARTITION_THRESHOLD", 3.0)
    for left in (lp, ordered_l):  # os dois lados regravados / só o direito (o esquerdo, em ordem, não é fatiado)
        ls, rs = open_source(left), open_source(rp)
        plan = plan_windows(ls, rs, ["id"], window_rows=1000)
        assert plan.slice_sides[1]
        new_plan, nl, nr, owned = pipeline._repartition(plan, ls, rs, ["id"], None)
        try:
            staged = (nl is not ls, nr is not rs)
            assert staged[1]
            assert new_plan.slice_sides == (plan.slice_sides[0] and not staged[0], False)
        finally:
            for src in owned:
                src.cleanup()


def test_repartition_uses_plan_stage_chunk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "PARTITION_THRESHOLD", 3.0)
    seen: list[int] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs["chunk_bytes"])
        return partition_source(*args, **kwargs)

    monkeypatch.setattr(pipeline, "partition_source", spy)
    lp, rp = _write_pair(tmp_path, shuffled=True)
    compare_rows(lp, rp, ["id"], window_rows=1000, memory_limit="1GiB")
    plan = plan_windows(open_source(lp), open_source(rp), ["id"], window_rows=1000, memory_limit="1GiB")
    assert seen
    assert set(seen) == {plan.stage_chunk_bytes}


def test_partitioned_staging_skipped_below_threshold(tmp_path: Path) -> None:
    """Amplificação baixa (~4, abaixo do limiar): regravar a fonte custa mais que ler a mais; não entra staging."""
    lp, rp = _write_pair(tmp_path, shuffled=True)
    got = compare_rows(lp, rp, ["id"], window_rows=1000)
    assert got.windows > 2
    assert all(3.0 < a < windows.PARTITION_THRESHOLD for a in got.read_amplification)
    assert not any("reparticionadas" in n for n in got.notes)
    assert (got.common, got.left_only, got.right_only) == (3900, 100, 100)
    assert got.mismatched == len([i for i in range(100, 4000) if i % 7 == 0])


def test_stage_partitioned_one_file_per_window(tmp_path: Path) -> None:
    """Vários arquivos de origem e fatias pequenas: cada janela acaba com os arquivos dela (um por fatia e origem, sem
    juntar), todos fora das subpastas, e o total de linhas é o da fonte."""
    lake = _write_text_lake(tmp_path / "lake", shuffled=True, text=False)
    src = open_source(lake)
    plan = plan_windows(src, src, ["id"], window_rows=500)
    staged = stage_partitioned(src, plan.windows, ["id"], ReadOptions(temp_dir=tmp_path), chunk_bytes=1)
    assert len(staged.groups) == len(plan.windows)
    assert all(len(g) >= 1 for g in staged.groups)
    assert len(staged.files) == sum(len(g) for g in staged.groups)
    assert sorted(p.name for p in staged.path.rglob("*.parquet")) == sorted(f.name for f in staged.files)
    ids: list[int] = []
    for w, group in zip(plan.windows, staged.groups, strict=True):
        got = pl.concat([pl.read_parquet(staged.files[i])["id"] for i in group])
        pred = w.predicate(["id"], [pl.Int64()])
        assert pred is None or got.len() == got.to_frame().filter(pred).height
        ids += got.to_list()
    assert sorted(ids) == list(range(4000))
    staged.cleanup()


def test_stage_partitioned_merge_keeps_rows_and_order_free(tmp_path: Path) -> None:
    """Fatias minúsculas (muitos arquivos por janela) e fatias padrão dão as mesmas linhas por janela, e a pasta
    temporária fica só com os ``w*.parquet`` (sem subpastas com arquivos)."""
    lake = _write_text_lake(tmp_path / "lake", shuffled=True, text=False)
    src = open_source(lake)
    plan = plan_windows(src, src, ["id"], window_rows=500)
    tiny = stage_partitioned(src, plan.windows, ["id"], ReadOptions(temp_dir=tmp_path), chunk_bytes=1)
    default = stage_partitioned(src, plan.windows, ["id"], ReadOptions(temp_dir=tmp_path))
    try:
        assert tiny.groups == default.groups
        for a, b in zip(tiny.groups, default.groups, strict=True):
            for i, j in zip(a, b, strict=True):
                left = pl.read_parquet(tiny.files[i]).sort("id")
                assert left.equals(pl.read_parquet(default.files[j]).sort("id"))
        for staged in (tiny, default):
            assert sorted(p.name for p in staged.path.iterdir()) == sorted(f.name for f in staged.files)
            assert all(p.name.startswith("w") and p.suffix == ".parquet" for p in staged.path.iterdir())
    finally:
        tiny.cleanup()
        default.cleanup()


@pytest.mark.parametrize("chunk_bytes", [1, 64 * 1024**2])
def test_stage_same_rows_per_window(tmp_path: Path, chunk_bytes: int) -> None:
    """O staging entrega, por janela, as mesmas linhas que a leitura da fonte original; a pasta temporária some."""
    lake = _write_text_lake(tmp_path / "lake", shuffled=True, text=False)
    src = open_source(lake)
    plan = plan_windows(src, src, ["id"], window_rows=500)
    assert len(plan.windows) > 4
    staged_src, groups = partition_source(
        src, plan.windows, ["id"], ReadOptions(temp_dir=tmp_path), chunk_bytes=chunk_bytes
    )
    folder = staged_src.staged.path if staged_src.staged is not None else None
    try:
        assert len(groups) == len(plan.windows)
        staged_plan = dataclasses.replace(plan, files=(groups, groups))
        want = [(lf.sort("id"), rf.sort("id")) for _, lf, rf in WindowReader(src, src, plan, ["id"], [pl.Int64()])]
        reader = WindowReader(staged_src, staged_src, staged_plan, ["id"], [pl.Int64()])
        got = [(lf.sort("id"), rf.sort("id")) for _, lf, rf in reader]
        assert len(got) == len(want)
        for (la, ra), (lb, rb) in zip(got, want, strict=True):
            assert la.columns == ["id", "v"]
            assert la.equals(lb)
            assert ra.equals(rb)
        assert sum(lf.height for lf, _ in got) == 4000
        reader_cols = WindowReader(
            staged_src, staged_src, staged_plan, ["id"], [pl.Int64()], columns_left=["v"], columns_right=["v"]
        )
        assert all(lf.columns == ["id", "v"] for _, lf, _ in reader_cols)
    finally:
        staged_src.cleanup()
    assert folder is not None
    assert not folder.exists()


def test_stage_partitioned_empty_window(tmp_path: Path) -> None:
    """Uma janela sem linhas não tem arquivo (``groups[i] == ()``); as outras têm um só."""
    lake = _write_text_lake(tmp_path / "lake", shuffled=True, text=False)
    src = open_source(lake)
    wins = [
        windows.Window(0, hi=(2000,)),
        windows.Window(1, lo=(5000,), hi=(6000,)),  # fora do intervalo das chaves (0..3999)
        windows.Window(2, lo=(2000,)),
    ]
    staged = stage_partitioned(src, wins, ["id"], ReadOptions(temp_dir=tmp_path), chunk_bytes=1)
    try:
        assert staged.groups[1] == ()
        assert len(staged.files) == sum(len(g) for g in staged.groups)
        assert [len(g) > 0 for g in staged.groups] == [True, False, True]
        assert sum(pl.read_parquet(f).height for f in staged.files) == 4000
    finally:
        staged.cleanup()


def test_stage_partitioned_cleanup(tmp_path: Path) -> None:
    lp, rp = _write_pair(tmp_path, shuffled=True)
    src = open_source(lp)
    plan = plan_windows(src, open_source(rp), ["id"], window_rows=1000)
    staged = stage_partitioned(src, plan.windows, ["id"], ReadOptions(temp_dir=tmp_path))
    assert len(staged.files) == len(plan.windows)
    assert sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in staged.files) == 4000
    folder = staged.path
    assert folder.exists()
    staged.cleanup()
    assert not folder.exists()


def _write_text_lake(folder: Path, *, shuffled: bool, n: int = 4000, files: int = 8, text: bool = True) -> Path:
    """``n`` linhas em ``files`` arquivos de 1 row group: faixas contíguas de chave, ou (``shuffled``) cada arquivo com
    chaves de toda a faixa. A chave é texto (``K0000042``) ou inteira."""
    folder.mkdir()
    for f in range(files):
        ids = [i for i in range(n) if (i % files == f if shuffled else i * files // n == f)]
        key = [f"K{i:07d}" for i in ids] if text else ids
        pl.DataFrame({"id": key, "v": [i * 3 for i in ids]}).write_parquet(
            folder / f"p{f:02d}.parquet", row_group_size=n, statistics=True
        )
    return folder


@pytest.mark.parametrize("text", [True, False])
def test_read_amplification_multi_file_text_key(tmp_path: Path, text: bool) -> None:
    """Chave de texto: sem estatística de inteiro, a amplificação vem das faixas dos arquivos (arquivo inteiro lido)."""
    ordered = _write_text_lake(tmp_path / "o", shuffled=False, text=text)
    shuffled = _write_text_lake(tmp_path / "s", shuffled=True, text=text)
    po = plan_windows(open_source(ordered), open_source(ordered), ["id"], window_rows=500)
    ps = plan_windows(open_source(shuffled), open_source(shuffled), ["id"], window_rows=500)
    assert len(po.windows) >= 4
    assert all(1.0 <= a <= 2.0 for a in po.read_amplification)  # cada janela encosta em 1-2 arquivos
    assert all(a > 2.5 for a in ps.read_amplification)  # as janelas encostam em muitos arquivos
    assert all(a > 1.5 * b for a, b in zip(ps.read_amplification, po.read_amplification, strict=True))


def test_read_amplification_single_text_file_is_nan(tmp_path: Path) -> None:
    """Um arquivo só e chave de texto: sem faixa por arquivo para estimar, continua desconhecida."""
    lake = _write_text_lake(tmp_path / "um", shuffled=False, files=1)
    plan = plan_windows(open_source(lake), open_source(lake), ["id"], window_rows=500)
    assert all(math.isnan(a) for a in plan.read_amplification)


def test_text_key_amplification_does_not_trigger_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A estimativa por arquivo (chave de texto) é só informação: o staging pediria row groups com estatística."""
    monkeypatch.setattr(windows, "PARTITION_THRESHOLD", 3.0)
    monkeypatch.setattr(pipeline, "PARTITION_THRESHOLD", 3.0)
    lake = _write_text_lake(tmp_path / "t", shuffled=True)
    got = compare_rows(lake, lake, ["id"], window_rows=500)
    assert all(a > 3.0 for a in got.read_amplification)
    assert not any("reparticionadas" in n for n in got.notes)
    assert (got.common, got.left_only, got.right_only, got.mismatched) == (4000, 0, 0, 0)


def test_partitioned_staging_other_side_when_aligned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Plano alinhado pelos row groups da esquerda (em ordem); a direita, em vários arquivos embaralhados, é
    reparticionada sozinha: a amplificação dela passa do limiar e a da esquerda não."""
    monkeypatch.setattr(windows, "PARTITION_THRESHOLD", 3.0)
    monkeypatch.setattr(pipeline, "PARTITION_THRESHOLD", 3.0)
    n = 4000
    pl.DataFrame({"id": range(n), "v": [i * 3 for i in range(n)]}).write_parquet(
        tmp_path / "left.parquet", row_group_size=250, statistics=True
    )
    right = _write_text_lake(tmp_path / "right", shuffled=True, n=n, text=False)
    got = compare_rows(tmp_path / "left.parquet", right, ["id"], window_rows=500)
    assert got.aligned_to_row_groups
    assert got.read_amplification[0] == pytest.approx(1.0, abs=0.05)
    assert got.read_amplification[1] > 3.0
    assert any("reparticionadas" in n for n in got.notes)
    assert (got.common, got.left_only, got.right_only, got.mismatched) == (n, 0, 0, 0)


def test_partitioned_staging_skipped_when_aligned(tmp_path: Path) -> None:
    lp, rp = _write_pair(tmp_path, shuffled=False)
    got = compare_rows(lp, rp, ["id"], window_rows=1000)
    assert got.aligned_to_row_groups
    assert not any("reparticionadas" in n for n in got.notes)
    assert got.read_amplification == pytest.approx((1.0, 1.0), abs=0.05)
