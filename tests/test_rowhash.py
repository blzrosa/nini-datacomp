"""Hash de linha e pares divergentes (`core/rowhash.py`, `cpp/rowhash.cpp`)."""

from __future__ import annotations

import struct
from datetime import date, timedelta
from itertools import pairwise
from typing import Any

import numpy as np
import polars as pl
import pytest

from nini_datacomp.core.arrow import Column, import_column, import_frame
from nini_datacomp.core.rowhash import diff_pairs, row_hash, row_hash_parts


def _cols(df: pl.DataFrame) -> list[Column]:
    return [import_column(s) for s in df.get_columns()]


def _h(*series: pl.Series) -> list[int]:
    return row_hash([import_column(s) for s in series]).tolist()


def _nan(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def test_values_are_pinned() -> None:
    """O hash não depende de plataforma, compilador nem versão do Polars: estes valores saíram do g++ e do clang no
    Linux e têm de sair iguais no Windows (clang-cl) e no macOS."""
    df = pl.DataFrame(
        {
            "i": [0, 1, -1, None],
            "f": [0.0, float("nan"), 1.5, -2.25],
            "s": ["", "abc", "a" * 20, "ação"],
            "b": [True, False, True, False],
        }
    )
    assert [hex(x) for x in row_hash(_cols(df)).tolist()] == [
        "0xfe4810159a67b015",
        "0xa4d58b4043b057",
        "0xcb3d30d01a9bbc70",
        "0xeee31d36f114edbb",
    ]
    text = ["0x307f2d35f6b05b5e", "0x4133295aeef03e12", "0x1c2405f51e6efd1", "0xb1b28d8c0bfa848"]
    assert [hex(x) for x in _h(df["s"])] == text


def test_what_eq_missing_calls_equal_hashes_equal() -> None:
    a = pl.Series("f", [0.0, _nan(0x7FF8000000000000), 1.0, None], dtype=pl.Float64)
    b = pl.Series("f", [-0.0, _nan(0xFFF8000000000001), 1.0, None], dtype=pl.Float64)
    assert _h(a) == _h(b)
    assert _h(a.cast(pl.Float32)) == _h(b.cast(pl.Float32))


def test_null_is_not_a_value() -> None:
    h = _h(pl.Series("i", [0, None, 0]))
    assert h[0] == h[2] != h[1]
    assert len(set(_h(pl.Series("s", ["", None])))) == 2


def test_validity_and_offsets_do_not_matter() -> None:
    """Com ou sem bitmap de validade, e com a coluna começando no meio de um byte: mesmos valores, mesmo hash."""
    plain = pl.Series("x", [3, 4, 5, 6, 7])
    sliced = pl.Series("x", [None, 9, 9, 3, 4, 5, 6, 7]).slice(3)  # validade presente, deslocada, sem nulos aqui
    assert _h(plain) == _h(sliced)
    bools = pl.Series("b", [True, False, False, True, True])
    assert _h(bools) == _h(pl.Series("b", [None, False, True, True, False, False, True, True]).slice(3))


def test_text_inline_and_in_buffers() -> None:
    values = ["curto", "um texto bem mais longo que doze bytes", "", "emoji 😀 e acentos ção", "x" * 12, "y" * 13]
    a = pl.Series("s", values)
    b = pl.concat([pl.Series("s", values[:3]), pl.Series("s", values[3:])], rechunk=True)  # outros buffers
    assert _h(a) == _h(b)
    assert len(set(_h(a))) == len(values)


def test_column_order_and_position_matter() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [2, 1]})
    h = row_hash(_cols(df)).tolist()
    assert h[0] != h[1]
    assert h != row_hash(_cols(df.select("b", "a"))).tolist()


@pytest.mark.parametrize(
    "dtype", [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Date, pl.Datetime]
)
def test_integer_like_types(dtype: Any) -> None:
    a = pl.Series("v", [1, 2, None, 3]).cast(dtype)
    b = pl.Series("v", [1, 2, None, 4]).cast(dtype)
    ha, hb = _h(a), _h(b)
    assert ha[:3] == hb[:3]
    assert ha[3] != hb[3]
    assert len(set(ha)) == 4


def test_no_columns() -> None:
    h = row_hash([], 3).tolist()
    assert len(set(h)) == 1
    with pytest.raises(ValueError, match="número de linhas"):
        row_hash([])


def test_threads_do_not_change_the_hash() -> None:
    rng = np.random.default_rng(5)
    n = 300_000  # vários blocos
    df = pl.DataFrame({"a": rng.integers(0, 9, n), "b": rng.random(n), "s": rng.integers(0, 99, n).astype(str)})
    df = df.with_columns(pl.when(pl.col("a") == 3).then(None).otherwise(pl.col("b")).alias("b"))
    cols = _cols(df)
    assert (row_hash(cols, threads=1) == row_hash(cols, threads=8)).all()


def _pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n, dtype=np.uint32)
    return idx, idx.copy()


def test_diff_pairs_finds_the_changed_rows() -> None:
    rng = np.random.default_rng(9)
    n = 200_000
    left = pl.DataFrame({"a": rng.integers(0, 5, n), "s": rng.integers(0, 5, n).astype(str)})
    changed = np.sort(rng.choice(n, 1_000, replace=False))
    a = left["a"].to_numpy().copy()
    a[changed] += 10
    right = left.with_columns(pl.Series("a", a))
    lc, rc = _cols(left), _cols(right)
    li, ri = _pairs(n)
    d = diff_pairs(lc, rc, row_hash(lc), row_hash(rc), left_rows=li, right_rows=ri, threads=4)
    assert d.pairs.tolist() == changed.tolist()
    assert d.collisions == 0
    # pares em outra ordem: as posições são as dos vetores de pares
    perm = rng.permutation(n).astype(np.uint32)
    d2 = diff_pairs(lc, rc, row_hash(lc), row_hash(rc), left_rows=perm, right_rows=perm)
    assert sorted(perm[d2.pairs].tolist()) == changed.tolist()


def test_verify_hash_catches_collisions() -> None:
    """Com os mesmos hashes nos dois lados (uma colisão em todo par), só o verify acha as linhas diferentes."""
    left = pl.DataFrame({"a": [1, 2, 3, 4], "f": [0.0, 1.0, float("nan"), None]})
    right = pl.DataFrame({"a": [1, 2, 9, 4], "f": [-0.0, 1.5, float("nan"), None]})
    lc, rc = _cols(left), _cols(right)
    li, ri = _pairs(4)
    h = row_hash(lc)
    assert len(diff_pairs(lc, rc, h, h, left_rows=li, right_rows=ri)) == 0
    d = diff_pairs(lc, rc, h, h, left_rows=li, right_rows=ri, verify=True)
    assert d.pairs.tolist() == [1, 2]
    assert d.collisions == 2
    real = diff_pairs(lc, rc, row_hash(lc), row_hash(rc), left_rows=li, right_rows=ri, verify=True)
    assert real.pairs.tolist() == [1, 2]
    assert real.collisions == 0


def test_diff_pairs_errors() -> None:
    lc = _cols(pl.DataFrame({"a": [1, 2]}))
    rc = _cols(pl.DataFrame({"a": ["1", "2"]}))
    li, ri = _pairs(2)
    with pytest.raises(ValueError, match="tipos diferentes"):
        diff_pairs(lc, rc, row_hash(lc), row_hash(rc), left_rows=li, right_rows=ri)
    with pytest.raises(ValueError, match="fora do hash"):
        diff_pairs(lc, lc, row_hash(lc), row_hash(lc), left_rows=li + 5, right_rows=ri)
    assert len(diff_pairs(lc, lc, row_hash(lc), row_hash(lc), left_rows=li[:0], right_rows=ri[:0])) == 0


def test_parts_give_the_same_hash() -> None:
    """Uma coluna em vários pedaços de memória, lida pedaço a pedaço (sem rechunk): o mesmo hash da coluna inteira."""
    rng = np.random.default_rng(11)
    df = pl.DataFrame({"a": rng.integers(0, 9, 150_000), "s": rng.integers(0, 99, 150_000).astype(str)})
    whole = row_hash(_cols(df))
    cuts = [0, 1, 70_000, 70_001, 150_000]
    parts = [(a, _cols(df.slice(a, b - a))) for a, b in pairwise(cuts)]
    assert (row_hash_parts(parts, df.height, threads=4) == whole).all()
    with pytest.raises(ValueError, match="cobrir"):
        row_hash_parts(parts[1:], df.height)


def _old_polars() -> bool:
    return tuple(int(x) for x in pl.__version__.split(".")[:2]) < (1, 31)


def test_import_frame_matches_column_by_column() -> None:
    """Todas as colunas numa exportação só (o caminho rápido das tabelas largas): mesmos hashes da leitura coluna a
    coluna, também numa fatia (deslocamento) e com nulos, texto longo e booleanos."""
    rng = np.random.default_rng(13)
    n = 50_000
    df = pl.DataFrame(
        {
            "i": rng.integers(-5, 5, n),
            "u": rng.integers(0, 9, n).astype(np.uint16),
            "f": rng.random(n),
            "s": [("abc" * (i % 9)) for i in range(n)],
            "b": rng.random(n) > 0.5,
            "d": pl.date_range(date(2000, 1, 1), date(2000, 1, 1) + timedelta(days=n - 1), eager=True),
        }
    ).with_columns(pl.when(pl.col("i") == 0).then(None).otherwise(pl.col("f")).alias("f"))
    for frame in (df, df.slice(1_234, 30_000)):
        fc = import_frame(frame)
        if fc is None:
            if _old_polars():
                pytest.skip(f"o Polars {pl.__version__} não exporta este DataFrame num lote só")
            pytest.fail("import_frame devolveu None num DataFrame simples")
        assert fc.names == tuple(frame.columns)
        assert fc.length == frame.height
        by_column = row_hash(_cols(frame))
        assert (row_hash_parts([(0, fc)], frame.height, threads=4) == by_column).all()


def test_import_frame_falls_back() -> None:
    cat = pl.DataFrame({"c": ["a", "b"]}).with_columns(pl.col("c").cast(pl.Categorical))
    assert import_frame(cat) is None  # códigos de dicionário: o hash compara o texto
    chunked = pl.concat([pl.DataFrame({"a": [1]}), pl.DataFrame({"a": [2]})], rechunk=False)
    fc = import_frame(chunked)  # dois lotes (None: quem chama lê coluna a coluna), ou o Polars juntou num só
    assert fc is None or (fc.length == 2 and (row_hash_parts([(0, fc)], 2) == row_hash(_cols(chunked))).all())
    assert import_frame(pl.DataFrame()) is None
