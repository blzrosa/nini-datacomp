"""diag_read.py -- fase 2: quanto custa ler uma janela de uma tabela larga, e em que parte.

    uv run python tools/diag_read.py                                   # 10M x 300, as 10 primeiras janelas
    uv run python tools/diag_read.py --case 1000000x300 --windows 5

O diag_memory mostrou que, em 10M x 300, só a leitura das janelas leva ~44 s (o hash fica escondido atrás dela), e
que janelas maiores (memory_limit 4GB) caem para ~31 s: cada leitura tem um custo fixo alto. Aqui, num processo só,
cada janela do lado esquerdo é lida de vários jeitos (o melhor de ``--repeat`` tentativas), para separar esse custo:

- ``filtro``: como o pacote lê hoje (os arquivos da janela, com o filtro da faixa de chaves);
- ``filtro_streaming``: o mesmo com o engine streaming;
- ``sem_filtro``: os mesmos arquivos inteiros, sem filtro (o filtro copia as 300 colunas?);
- ``slice``: as mesmas linhas pela posição (``slice``), sem filtro;
- ``read_parquet``: ``pl.read_parquet`` direto, sem o LazyFrame;
- ``10_colunas``: só 10 das colunas, com o filtro (o custo cresce com as colunas?);
- ``todas_juntas``: as N janelas numa leitura só, dividida por N (o piso sem o custo por leitura).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

os.environ.setdefault("MIMALLOC_PURGE_DELAY", "0")  # o alocador como o pacote o configura
os.environ.setdefault("MIMALLOC_RESET_DELAY", "0")

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_rows import dataset
from nini_datacomp.io import Source, open_source
from nini_datacomp.pipeline import prepare_keys
from nini_datacomp.windows import plan_windows


def best(fn: Callable[[], pl.DataFrame], repeat: int) -> tuple[float, int]:
    times, rows = [], 0
    for _ in range(repeat):
        t = time.perf_counter()
        rows = fn().height
        times.append(time.perf_counter() - t)
    return min(times), rows


def _window(
    ls: Source, paths: list[Path], pred: pl.Expr | None, *, offset: int, few: list[str], repeat: int
) -> tuple[dict[str, float], int]:
    """Os tempos (melhor de ``repeat``) de cada jeito de ler uma janela, e as linhas dela."""

    def scan() -> pl.LazyFrame:
        lf = ls.scan(paths)
        return lf if pred is None else lf.filter(pred)

    rows_in = int(scan().select(pl.len()).collect().item())
    variants: dict[str, Callable[[], pl.DataFrame]] = {
        "filtro": lambda: scan().collect(),
        "filtro_streaming": lambda: scan().collect(engine="streaming"),
        "sem_filtro": lambda: ls.scan(paths).collect(),
        "slice": lambda: ls.lazy.slice(offset, rows_in).collect(),
        "read_parquet": lambda: pl.read_parquet([str(p) for p in paths]),
        "10_colunas": lambda: scan().select(few).collect(),
    }
    return {name: best(fn, repeat)[0] for name, fn in variants.items()}, rows_in


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="10000000x300")
    ap.add_argument("--windows", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--memory-limit", default="1GB")
    args = ap.parse_args()

    root = dataset(args.case)
    ls, rs = open_source(root / "left"), open_source(root / "right")
    ls, rs, dtypes = prepare_keys(ls, rs, ["id"])
    t = time.perf_counter()
    plan = plan_windows(ls, rs, ["id"], args.memory_limit)
    print(
        f"plano: {len(plan.windows)} janelas de ~{plan.rows_per_window:,} linhas em {time.perf_counter() - t:.2f} s; "
        f"{len(ls.files)} arquivos à esquerda; alinhado: {plan.aligned_to_row_groups}"
    )
    cols = ls.lazy.collect_schema().names()
    few = cols[:11]

    totals: dict[str, float] = {}
    offset = 0
    windows = plan.windows[: args.windows]
    all_files: set[int] = set()
    for w in windows:
        files_l, _ = plan.files_for(w.index)
        idx = files_l if files_l is not None else tuple(range(len(ls.files)))
        all_files |= set(idx)
        times, rows_in = _window(
            ls, [ls.files[i] for i in idx], w.predicate(["id"], dtypes), offset=offset, few=few, repeat=args.repeat
        )
        for name, secs in times.items():
            totals[name] = totals.get(name, 0.0) + secs
        line = "  ".join(f"{name} {secs * 1000:6.0f} ms" for name, secs in times.items())
        print(f"janela {w.index:>3} ({len(idx)} arq., {rows_in:,} linhas): {line}", flush=True)
        offset += rows_in

    paths = [ls.files[i] for i in sorted(all_files)]
    secs, rows = best(lambda: ls.scan(paths).collect(), args.repeat)
    totals["todas_juntas"] = secs
    print(f"\ntodas as {len(windows)} janelas numa leitura só: {secs:.2f} s ({rows:,} linhas)")
    print(f"\nsoma de {len(windows)} janelas, lado esquerdo (s):")
    for name, secs in totals.items():
        print(f"  {name:<18} {secs:7.2f}   ({secs / len(windows) * 1000:6.0f} ms por janela)")


if __name__ == "__main__":
    main()
