"""Planejamento das janelas (`windows.py`): cada chave cai em exatamente uma janela, dos dois lados."""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from nini_datacomp.io import open_source, row_groups
from nini_datacomp.windows import MIN_MEMORY, Plan, Window, parse_memory_limit, plan_windows


def _rows_per_window(plan: Plan, df: pl.DataFrame, keys: list[str]) -> list[int]:
    dtypes = [df.schema[k] for k in keys]
    out = []
    for w in plan.windows:
        pred = w.predicate(keys, dtypes)
        out.append(df.height if pred is None else df.filter(pred).height)
    return out


def assert_partition(plan: Plan, df: pl.DataFrame, keys: list[str]) -> None:
    """Cada linha de ``df`` está em exatamente uma janela."""
    dtypes = [df.schema[k] for k in keys]
    indexed = df.with_row_index("__i")
    seen: list[int] = []
    for w in plan.windows:
        pred = w.predicate(keys, dtypes)
        seen += (indexed if pred is None else indexed.filter(pred))["__i"].to_list()
    assert sorted(seen) == list(range(df.height))


def test_parse_memory_limit() -> None:
    assert parse_memory_limit("2GB") == 2_000_000_000
    assert parse_memory_limit("1.5GiB") == int(1.5 * 1024**3)
    assert parse_memory_limit(3 * 1024**3) == 3 * 1024**3
    assert parse_memory_limit(1) == MIN_MEMORY
    assert MIN_MEMORY <= parse_memory_limit("auto") <= 2 * 1024**3
    with pytest.raises(ValueError, match="memory_limit"):
        parse_memory_limit("2 potes")


def test_small_tables_are_one_window() -> None:
    df = pl.DataFrame({"id": range(1000), "v": range(1000)})
    plan = plan_windows(open_source(df), open_source(df), ["id"])
    assert len(plan.windows) == 1
    assert plan.windows[0].is_full
    assert plan.windows[0].predicate(["id"], [pl.Int64()]) is None


def test_budget_decides_rows_per_window() -> None:
    df = pl.DataFrame({"id": range(200_000), "s": [f"texto {i}" for i in range(200_000)]})
    small = plan_windows(open_source(df), open_source(df), ["id"], memory_limit="64MiB")
    large = plan_windows(open_source(df), open_source(df), ["id"], memory_limit="1GiB")
    assert small.rows_per_window < large.rows_per_window
    assert small.bytes_per_row[0] > 8  # id + texto
    assert len(small.windows) >= 1


@pytest.mark.parametrize("shuffle", [False, True])
def test_int_key_partition(shuffle: bool) -> None:
    ids = list(range(-5_000, 5_000, 3))
    if shuffle:
        random.Random(1).shuffle(ids)
    left = pl.DataFrame({"id": ids})
    right = pl.DataFrame({"id": [i + 1 for i in ids[: len(ids) // 2]] + [10**9, -(10**9)]})
    plan = plan_windows(open_source(left), open_source(right), ["id"], window_rows=500)
    assert len(plan.windows) == -(-left.height // 500)
    assert_partition(plan, left, ["id"])
    assert_partition(plan, right, ["id"])  # inclusive os ids fora da faixa da esquerda
    sizes = _rows_per_window(plan, left, ["id"])
    assert all(s == 500 for s in sizes[:-1])


def test_text_key_partition() -> None:
    words = [f"chave-{i:06d}" for i in random.Random(2).sample(range(10**6), 3_000)] + ["", "ção", "Z"]
    left = pl.DataFrame({"k": words, "v": range(len(words))})
    right = pl.DataFrame({"k": [*words[::2], "novo"], "v": range(len(words[::2]) + 1)})
    plan = plan_windows(open_source(left), open_source(right), ["k"], window_rows=400)
    assert len(plan.windows) > 5
    assert_partition(plan, left, ["k"])
    assert_partition(plan, right, ["k"])


def test_composite_key_with_nulls() -> None:
    rng = random.Random(3)
    texts = [None, "a", "b", "c", "texto longo"]
    rows = list({(rng.choice([None, *range(20)]), rng.choice(texts)) for _ in range(500)})
    rows = rows * 3  # chaves repetidas também têm de cair numa janela só
    df = pl.DataFrame({"a": [r[0] for r in rows], "b": [r[1] for r in rows]}, schema={"a": pl.Int32, "b": pl.String})
    plan = plan_windows(open_source(df), open_source(df), ["a", "b"], window_rows=30)
    assert len(plan.windows) > 3
    assert plan.windows[0].with_nulls
    assert_partition(plan, df, ["a", "b"])
    # a mesma chave na mesma janela
    dtypes = [pl.Int32(), pl.String()]
    for w in plan.windows:
        pred = w.predicate(["a", "b"], dtypes)
        part = df if pred is None else df.filter(pred)
        others = df.join(part.unique(), on=["a", "b"], how="semi", nulls_equal=True)
        assert others.height == part.height


def test_date_key_partition() -> None:
    days = [date(2020, 1, 1) + timedelta(days=i) for i in range(2_000)]
    df = pl.DataFrame({"d": days, "v": range(2_000)})
    plan = plan_windows(open_source(df), open_source(df.slice(100, 1_500)), ["d"], window_rows=300)
    assert_partition(plan, df, ["d"])


def test_categorical_key_is_compared_as_text() -> None:
    df = pl.DataFrame({"c": [f"v{i:04d}" for i in range(1_000)]})
    cat = df.with_columns(pl.col("c").cast(pl.Categorical))
    plan = plan_windows(open_source(cat), open_source(df), ["c"], window_rows=100)
    assert_partition(plan, df, ["c"])  # as fronteiras são texto


def test_aligned_to_row_groups(tmp_path: Path) -> None:
    n, rg = 20_000, 1_000
    df = pl.DataFrame({"id": range(n), "v": [i * 0.5 for i in range(n)]})
    for part in range(4):  # quatro arquivos, em ordem
        df.slice(part * 5_000, 5_000).write_parquet(tmp_path / f"part-{part:05d}.parquet", row_group_size=rg)
    plan = plan_windows(open_source(tmp_path), open_source(tmp_path), ["id"], window_rows=2_600)
    assert plan.aligned_to_row_groups
    assert_partition(plan, df, ["id"])
    for w in plan.windows[1:]:
        assert w.lo is not None
        assert w.lo[0] % rg == 0  # cada janela começa num row group


def test_unsorted_parquet_is_not_aligned(tmp_path: Path) -> None:
    df = pl.DataFrame({"id": list(range(5_000))[::-1]})
    df.write_parquet(tmp_path / "x.parquet", row_group_size=500)
    plan = plan_windows(open_source(tmp_path), open_source(df), ["id"], window_rows=1_200)
    assert not plan.aligned_to_row_groups
    assert_partition(plan, df, ["id"])


def test_describe_and_errors() -> None:
    assert Window(0, with_nulls=True).describe() == "todas as linhas"
    assert "chaves nulas" in Window(0, None, (5,), with_nulls=True).describe()
    df = pl.DataFrame({"id": [1]})
    with pytest.raises(ValueError, match="window_rows"):
        plan_windows(open_source(df), open_source(df), ["id"], window_rows=0)
    with pytest.raises(ValueError, match="ao menos uma"):
        plan_windows(open_source(df), open_source(df), [])


def test_window_is_never_smaller_than_a_row_group(tmp_path: Path) -> None:
    """Linhas largas e memory_limit pequeno pedem janelas menores que um row group: cada janela leva um row group
    inteiro (o Polars o decodifica de qualquer jeito), alinhada, e a memória cabe lendo as colunas em grupos."""
    n, rg = 20_000, 5_000
    df = pl.DataFrame({"id": range(n), **{f"s{j}": [f"{i:06d}" + "x" * 500 for i in range(n)] for j in range(4)}})
    df.write_parquet(tmp_path / "x.parquet", row_group_size=rg)
    groups = row_groups(tmp_path / "x.parquet")  # o Polars 1.30 nem sempre corta no tamanho pedido
    if len(groups) < 2 or len(set(groups[:-1])) != 1:
        pytest.skip(f"row groups de tamanhos {groups} no Polars {pl.__version__}")
    plan = plan_windows(open_source(tmp_path), open_source(tmp_path), ["id"], memory_limit=MIN_MEMORY)
    assert plan.rows_per_window == max(groups)
    assert plan.aligned_to_row_groups
    assert len(plan.windows) == len(groups)
    assert plan.column_groups >= 2
    assert any("grupos de colunas" in note for note in plan.notes)
    assert set(plan.column_bytes) == {"s0", "s1", "s2", "s3"}  # sem a chave
    split = plan.split_columns(["s0", "s1", "s2", "s3"])
    assert [c for g in split for c in g] == ["s0", "s1", "s2", "s3"]  # grupos seguidos, na ordem
    assert len(split) == min(plan.column_groups, 4)
    assert_partition(plan, df, ["id"])
    fixed = plan_windows(open_source(tmp_path), open_source(tmp_path), ["id"], window_rows=1_000)
    assert (fixed.rows_per_window, fixed.column_groups) == (1_000, 1)  # window_rows explícito não é corrigido


def test_each_window_reads_only_its_files(tmp_path: Path) -> None:
    """Com vários arquivos, cada janela lê só os que encostam na sua faixa de chaves (o Polars não abre o rodapé de
    todos a cada janela), e nenhuma linha fica de fora."""
    n = 20_000
    df = pl.DataFrame({"id": range(n), "v": [i % 13 for i in range(n)]})
    for part in range(8):
        df.slice(part * 2_500, 2_500).write_parquet(tmp_path / f"part-{part:05d}.parquet", row_group_size=500)
    src = open_source(tmp_path)
    plan = plan_windows(src, src, ["id"], window_rows=2_000)
    left, right = plan.files
    assert left is not None
    assert right == left
    assert len(left) == len(plan.windows)
    assert all(0 < len(files) <= 2 for files in left)  # uma janela de 2.000 linhas cai em até 2 arquivos de 2.500
    dtypes = [df.schema["id"]]
    for w, files in zip(plan.windows, left, strict=True):
        pred = w.predicate(["id"], dtypes)
        want = df.height if pred is None else df.filter(pred).height
        got = src.scan([src.files[i] for i in files])
        assert (got.collect() if pred is None else got.filter(pred).collect()).height == want
    single = plan_windows(open_source(df), open_source(df), ["id"], window_rows=2_000)
    assert single.files == (None, None)  # DataFrame: não há arquivos


@pytest.mark.parametrize("nulls", [False, True])
def test_footer_statistics_give_the_same_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nulls: bool) -> None:
    """Com uma chave inteira, o planejamento usa as estatísticas dos rodapés (nulos, começo dos row groups, faixa de
    cada arquivo) em vez de ler a chave; o plano sai o mesmo que lendo."""
    from nini_datacomp import windows  # noqa: PLC0415

    ids: list[int | None] = list(range(60_000))
    if nulls:
        ids[123] = None
    df = pl.DataFrame({"id": ids, "v": [i % 7 for i in range(60_000)]}, schema={"id": pl.Int64, "v": pl.Int64})
    for i in range(3):
        df.slice(i * 20_000, 20_000).write_parquet(tmp_path / f"p{i}.parquet", row_group_size=5_000)
    src = open_source(tmp_path)
    fk = windows._footer_keys(src, ["id"])
    assert fk is not None
    assert fk.nulls == nulls
    assert fk.ranges is not None
    assert len(fk.ranges) == 3
    if not nulls:
        assert fk.firsts is not None
        assert fk.firsts["id"].to_list() == list(range(0, 60_000, 5_000))
    fast = plan_windows(src, src, ["id"], window_rows=10_000)
    monkeypatch.setattr(windows, "_footer_keys", lambda *_: None)
    slow = plan_windows(src, src, ["id"], window_rows=10_000)
    assert fast.windows == slow.windows
    assert fast.files == slow.files
    assert fast.aligned_to_row_groups == slow.aligned_to_row_groups
    assert_partition(fast, pl.read_parquet(tmp_path), ["id"])


def test_footer_statistics_skip_renamed_and_cast_keys(tmp_path: Path) -> None:
    from nini_datacomp import windows  # noqa: PLC0415

    pl.DataFrame({"ID": range(100), "s": ["a"] * 100}).write_parquet(tmp_path / "a.parquet")
    src = open_source(tmp_path / "a.parquet")
    renamed = src.rename({"ID": "id"})
    fk = windows._footer_keys(renamed, ["id"])
    assert fk is not None  # renomeada: as estatísticas são as da coluna "ID" do arquivo
    assert fk.ranges is None  # um arquivo só
    cast = renamed.with_columns([pl.col("id").cast(pl.Int32)])
    assert windows._footer_keys(cast, ["id"]) is None  # convertida depois da leitura: não vale
    assert windows._footer_keys(open_source(tmp_path / "a.parquet"), ["s"]) is None  # texto
    assert windows._footer_keys(open_source(tmp_path / "a.parquet"), ["ID", "s"]) is None  # composta
