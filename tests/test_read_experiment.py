"""Plano de janelas do experimento de leitura (`tools/read_experiment.py`); a medida em si roda à mão."""

from __future__ import annotations

from itertools import pairwise

from read_experiment import WINDOW_CELLS, windows


def test_small_table_is_one_window() -> None:
    assert windows(10_000, 10, 100_000) == [(0, 10_000 + 100 + 1)]


def test_windows_cover_all_ids_aligned_to_row_groups() -> None:
    rows, cols, rg = 10_000_000, 300, 100_000
    plan = windows(rows, cols, rg)
    assert plan[0][0] == 0
    assert plan[-1][1] > rows  # inclui os ids só da direita, acima de `rows`
    assert all(a[1] == b[0] for a, b in pairwise(plan))
    assert all((hi - lo) % rg == 0 for lo, hi in plan[:-1])
    # ~WINDOW_CELLS células por janela, nunca menos de um row group
    assert all((hi - lo) == rg or (hi - lo) * (cols + 1) <= WINDOW_CELLS for lo, hi in plan[:-1])
    assert len(windows(rows, 10, rg)) < len(plan)  # menos colunas: janelas maiores
