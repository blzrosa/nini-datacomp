"""O caminho completo da fase 1 (`pipeline.count_rows`): janelas lidas pelo Polars, chaves alinhadas no C++, contagens
somadas. Conferido contra os gabaritos do datacompolars e contra o esperado pela construção dos dados."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

import make_data
import make_golden
from cases import CASES
from make_data import Spec, generate, load_expected
from nini_datacomp.io import open_source
from nini_datacomp.pipeline import WindowReader, count_rows, prepare_keys
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
