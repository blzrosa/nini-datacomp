"""diag_profile.py -- onde vai o tempo de uma comparação (cProfile), para os casos pequenos em que cada etapa pesa.

    uv run python tools/diag_profile.py                          # 1M x 50, 5 execuções (a primeira aquece)
    uv run python tools/diag_profile.py --case 1000000x10 --top 40

Num processo só: uma execução de aquecimento e ``--runs`` medidas. Mostra o tempo de cada uma, as etapas
(``timings``) e as funções que mais pesam no total das medidas (tempo acumulado e tempo próprio). O tempo nas threads
do Polars e dos kernels C++ aparece como espera na função do Python que as chamou.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import sys
import time
from pathlib import Path

os.environ.setdefault("MIMALLOC_PURGE_DELAY", "0")  # o alocador como o pacote o configura
os.environ.setdefault("MIMALLOC_RESET_DELAY", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl

import nini_datacomp as dc
from bench_rows import dataset
from nini_datacomp import windows as w
from nini_datacomp.io import open_source, row_groups


def plan_parts(left: Path, right: Path, repeat: int) -> None:
    """Cada consulta do planejamento sozinha (o melhor de ``repeat``), e as duas rodadas como o plano as faz."""
    ls, rs = open_source(left), open_source(right)
    kl, _kr = w._key_lazy(ls, ["id"]), w._key_lazy(rs, ["id"])
    n = sum(n for f in ls.files for n in row_groups(f))
    starts = w._row_group_starts(ls)
    firsts = kl.with_row_index("__i").filter(pl.col("__i").is_in(starts))
    nulls = kl.select(w._any_null(["id"]).any())
    parts = {
        "rodapés (row groups)": lambda: [row_groups(f) for src in (ls, rs) for f in src.files],
        f"amostra head({w.SAMPLE_ROWS:,}) esquerda": lambda: ls.lazy.head(w.SAMPLE_ROWS).collect(engine="streaming"),
        "amostra head(10.000) esquerda": lambda: ls.lazy.head(10_000).collect(engine="streaming"),
        "chave nula, esquerda": lambda: nulls.collect(engine="streaming"),
        "chave no começo dos row groups": lambda: firsts.collect(engine="streaming"),
        "amostra só das colunas de texto, esquerda": lambda: w._collect(
            [w._sample(ls, w._variable_columns(ls), w.SAMPLE_ROWS)]
        ),
        "amostra só das colunas de texto, 10.000": lambda: w._collect([w._sample(ls, w._variable_columns(ls), 10_000)]),
        "esquema (collect_schema)": ls.lazy.collect_schema,
        "plan_windows inteiro": lambda: w.plan_windows(ls, rs, ["id"], "1GB"),
    }
    print(f"\nplanejamento ({n:,} linhas à esquerda, {len(starts)} row groups), melhor de {repeat}:")
    for name, fn in parts.items():
        best = float("inf")
        for _ in range(repeat):
            t = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t)
        print(f"  {name:<40} {best * 1000:7.1f} ms")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="1000000x50")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--memory-limit", default="1GB")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    root = dataset(args.case)
    left, right = root / "left", root / "right"

    def run() -> dc.ComparisonResult:
        return dc.compare(left, right, "id", memory_limit=args.memory_limit)

    run()  # aquecimento (cache de disco, threads)
    prof = cProfile.Profile()
    for i in range(args.runs):
        t = time.perf_counter()
        prof.enable()
        res = run()
        prof.disable()
        ex = res.execution
        phases = ", ".join(f"{k} {v:.3f}" for k, v in ex.timings.items() if v >= 0.0005)
        print(f"#{i + 1}: {time.perf_counter() - t:.3f} s, {ex.windows} janelas; {phases}", flush=True)
    plan_parts(left, right, args.runs)
    for order in ("cumulative", "tottime"):
        out = io.StringIO()
        pstats.Stats(prof, stream=out).strip_dirs().sort_stats(order).print_stats(args.top)
        print(f"\n===== por {order} (soma das {args.runs} execuções) =====")
        print(out.getvalue().split("\n", 4)[-1])


if __name__ == "__main__":
    main()
