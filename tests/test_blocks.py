"""Blocos de linhas e pool de threads (`core.blocks`)."""

from __future__ import annotations

import threading

import pytest

from nini_datacomp.core.blocks import BLOCK_ROWS, THREADS_ENV, map_blocks, parallel_map, resolve_threads, split


def test_split_covers_all_rows() -> None:
    n = 3 * BLOCK_ROWS + 17
    blocks = split(n)
    assert [b.index for b in blocks] == [0, 1, 2, 3]
    assert sum(b.rows for b in blocks) == n
    assert blocks[-1].start == 3 * BLOCK_ROWS
    assert blocks[-1].rows == 17
    assert split(0) == []


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_split_rejects(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        split(bad)  # type: ignore[arg-type]


def test_resolve_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_threads(3) == 3
    monkeypatch.setenv(THREADS_ENV, "2")
    assert resolve_threads() == 2
    monkeypatch.setenv(THREADS_ENV, "x")
    with pytest.raises(ValueError, match=THREADS_ENV):
        resolve_threads()
    with pytest.raises(ValueError, match=">= 1"):
        resolve_threads(0)


def test_parallel_map_keeps_order_and_uses_threads() -> None:
    seen: set[str] = set()

    def f(x: int) -> int:
        seen.add(threading.current_thread().name)
        return x * x

    assert parallel_map(f, list(range(200)), threads=4) == [x * x for x in range(200)]
    assert any(name.startswith("nini-datacomp") for name in seen)


def test_nested_call_runs_serially() -> None:
    def inner(x: int) -> list[int]:
        return parallel_map(lambda y: y + x, [1, 2, 3], threads=4)

    assert parallel_map(inner, [10, 20], threads=2) == [[11, 12, 13], [21, 22, 23]]


def test_map_blocks() -> None:
    blocks = split(2 * BLOCK_ROWS + 1)
    assert map_blocks(lambda b: b.rows, blocks, threads=2) == [BLOCK_ROWS, BLOCK_ROWS, 1]
