"""diag_memory.py -- fase 1c: de onde vem o pico de memória do caminho por janelas.

    uv run python tools/diag_memory.py                                  # 10M x 10 e 10M x 50, todas as variantes
    uv run python tools/diag_memory.py --cases 10000000x10 --variants padrao sem_prefetch --runs 3

O bench_rows mostrou pico de ~1,4 GB com ``memory_limit = 1GB`` e a maior janela com ~200 MB: o que sobra não é a
janela. Aqui cada execução (processo novo) refaz o laço do ``count_rows`` anotando a memória do processo depois de
cada etapa (abrir as fontes, planejar, receber cada janela, alinhar) e roda variações que isolam uma causa por vez:

- ``padrao``: como o bench_rows, mas com o alocador do Polars sem configurar.
- ``sem_prefetch``: uma janela por vez (se o pico cair muito, é a janela seguinte lida em paralelo).
- ``threads4``: ``POLARS_MAX_THREADS=4`` (se cair, é o Polars lendo vários row groups ao mesmo tempo).
- ``streaming``: ``collect(engine="streaming")`` em cada janela.
- ``so_chave``: só a coluna da chave (o que sobra é o custo fixo: processo, planejamento, kernels).
- ``limite_256MB``: janelas 4x menores (se o pico não cair junto, ele não depende da janela).
- ``mimalloc_purge``: ``MIMALLOC_PURGE_DELAY=0`` (o alocador do Polars no Windows devolve a memória livre na hora).
- ``so_purge_delay``, ``so_reset_delay``: só uma das duas variáveis (qual delas o mimalloc do Polars lê).
- ``purge_1ms``, ``purge_10ms``, ``purge_100ms``: o mimalloc devolve a memória depois de esperar esse tempo (menos
  idas ao sistema operacional, talvez menos custo de tempo que o 0).
- ``purge_0_threads8``: purge na hora e o Polars com 8 threads.
- ``purge_0_segura``: purge na hora, segurando a janela anterior como o ``count_rows`` fazia (três janelas vivas).
- ``sem_alinhar``: só lê as janelas, sem o C++ (separa o Polars dos kernels).
- ``sem_pool``: ``NINI_DATACOMP_POOL=0`` (os buffers dos kernels sem o pool do C++).

Resultado em ``docs/benchmarks/diag_memoria.md`` e ``.json`` (não precisa entrar no commit).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# aqui se mede o Polars sem configurar: o nini_datacomp não mexe no alocador (nem avisa); só a variante
# mimalloc_purge liga a configuração. Os subprocessos herdam isto
os.environ.setdefault("NINI_DATACOMP_ALLOCATOR", "0")

import polars as pl
import psutil

from nini_datacomp import pool_stats
from nini_datacomp.core.arrow import import_column
from nini_datacomp.core.keys import align_keys
from nini_datacomp.io import open_source
from nini_datacomp.pipeline import WindowReader, prepare_keys
from nini_datacomp.windows import plan_windows

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_baseline import _self_peak_bytes
from bench_rows import dataset
from make_data import load_expected

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "benchmarks"
CASES = ("10000000x10", "10000000x50")
MB = 1024 * 1024

# nome -> (variáveis de ambiente, opções do worker)
VARIANTS: dict[str, tuple[dict[str, str], dict[str, Any]]] = {
    "padrao": ({}, {}),
    "sem_prefetch": ({}, {"prefetch": False}),
    "threads4": ({"POLARS_MAX_THREADS": "4"}, {}),
    "streaming": ({}, {"engine": "streaming"}),
    "so_chave": ({}, {"columns": []}),
    "limite_256MB": ({}, {"memory_limit": "256MB"}),
    "mimalloc_purge": ({"MIMALLOC_PURGE_DELAY": "0", "MIMALLOC_RESET_DELAY": "0"}, {}),
    "so_purge_delay": ({"MIMALLOC_PURGE_DELAY": "0"}, {}),
    "so_reset_delay": ({"MIMALLOC_RESET_DELAY": "0"}, {}),
    "purge_1ms": ({"MIMALLOC_PURGE_DELAY": "1"}, {}),
    "purge_10ms": ({"MIMALLOC_PURGE_DELAY": "10"}, {}),
    "purge_100ms": ({"MIMALLOC_PURGE_DELAY": "100"}, {}),
    "purge_0_threads8": ({"MIMALLOC_PURGE_DELAY": "0", "POLARS_MAX_THREADS": "8"}, {}),
    "purge_0_segura": ({"MIMALLOC_PURGE_DELAY": "0"}, {"hold": True}),
    "sem_alinhar": ({}, {"align": False}),
    "sem_pool": ({"NINI_DATACOMP_POOL": "0"}, {}),
}


# ----------------------------------------------------------------------------- execução (subprocesso)
def _mem() -> dict[str, float]:
    info = psutil.Process().memory_info()
    out = {"rss": info.rss / MB, "peak": _self_peak_bytes() / MB}
    private = getattr(info, "private", None)  # Windows: memória reservada (commit), mesmo sem estar no working set
    if private:
        out["private"] = private / MB
    return out


def _windows(reader: WindowReader, keys: list[str], align: bool, hold: bool) -> tuple[list[dict[str, Any]], list[int]]:
    """O laço do ``count_rows``, anotando a memória ao receber cada janela e depois de alinhá-la.

    ``hold``: segura a janela anterior até a próxima chegar (como o ``count_rows`` fazia antes da correção)."""
    rows: list[dict[str, Any]] = []
    counts = [0, 0, 0]
    held: list[Any] = []
    for w, lf, rf in reader:
        held.clear()
        got = _mem()  # esta janela e (com prefetch) a seguinte já sendo lida
        size = (lf.estimated_size() + rf.estimated_size()) / MB
        if align:
            a = align_keys([import_column(lf[k]) for k in keys], [import_column(rf[k]) for k in keys])
            counts = [counts[0] + a.common, counts[1] + a.only_left, counts[2] + a.only_right]
            del a
        after = _mem()
        rows.append(
            {
                "i": w.index,
                "rows": lf.height + rf.height,
                "window_mb": size,
                "rss_got": got["rss"],
                "rss_aligned": after["rss"],
                "peak": after["peak"],
                "private": after.get("private"),
                "pool_cached_mb": pool_stats()["cached"] / MB,
            }
        )
        if hold:
            held.extend((lf, rf))
        del lf, rf
    return rows, counts


def worker(spec: dict[str, Any]) -> None:
    marks: dict[str, Any] = {"imports": _mem(), "polars_threads": pl.thread_pool_size()}
    t0 = time.perf_counter()
    keys = ["id"]
    ls, rs = open_source(spec["left"]), open_source(spec["right"])
    ls, rs, dtypes = prepare_keys(ls, rs, keys)
    columns = spec.get("columns")
    if columns is None:
        rnames = set(rs.lazy.collect_schema().names())
        columns = [c for c in ls.lazy.collect_schema().names() if c in rnames]
    marks["open"] = _mem()
    plan = plan_windows(ls, rs, keys, spec.get("memory_limit", "1GB"))
    marks["plan"] = _mem()
    marks["plan_seconds"] = time.perf_counter() - t0
    reader = WindowReader(
        ls,
        rs,
        plan,
        keys,
        dtypes,
        columns_left=columns,
        columns_right=columns,
        prefetch=spec.get("prefetch", True),
        engine=spec.get("engine", "auto"),
    )
    per_window, counts = _windows(reader, keys, spec.get("align", True), spec.get("hold", False))
    marks["end"] = _mem()
    out = {
        "seconds": time.perf_counter() - t0,
        "windows": len(plan.windows),
        "rows_per_window": plan.rows_per_window,
        "bytes_per_row": plan.bytes_per_row,
        "aligned": plan.aligned_to_row_groups,
        "marks": marks,
        "per_window": per_window,
        "counts": counts,
        "pool": pool_stats(),
    }
    print("RESULT " + json.dumps(out))


def _base_env() -> dict[str, str]:
    """O ambiente dos subprocessos sem as variáveis do mimalloc (o bench_rows, importado aqui, liga uma delas)."""
    return {k: v for k, v in os.environ.items() if not k.startswith("MIMALLOC_")}


def measure_once(spec: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
    with tempfile.TemporaryFile("w+b") as out_f, tempfile.TemporaryFile("w+b") as err_f:
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(spec)],
            stdout=out_f,
            stderr=err_f,
            env={**_base_env(), **env},
        )
        ps = psutil.Process(proc.pid)
        peak = 0
        while proc.poll() is None:
            try:
                peak = max(peak, ps.memory_info().rss)
            except psutil.Error:
                break
            time.sleep(0.02)
        proc.wait()
        out_f.seek(0)
        err_f.seek(0)
        stdout = out_f.read().decode("utf-8", errors="replace")
        stderr = err_f.read().decode("utf-8", errors="replace")
    for line in stdout.splitlines():
        if line.startswith("RESULT "):
            r = json.loads(line[7:])
            r["peak_mb"] = max(r["marks"]["end"]["peak"], peak / MB)
            return r
    raise RuntimeError("a execução falhou:\n" + stderr[-3000:])


# ----------------------------------------------------------------------------- relatório
def _summary(r: dict[str, Any]) -> dict[str, float]:
    pw = r["per_window"]
    return {
        "imports": r["marks"]["imports"]["rss"],
        "plan_rss": r["marks"]["plan"]["rss"],
        "plan_peak": r["marks"]["plan"]["peak"],
        "win_max": max((w["rss_got"] for w in pw), default=0.0),
        "win_mb": max((w["window_mb"] for w in pw), default=0.0),
        "end": r["marks"]["end"]["rss"],
        "peak": r["peak_mb"],
        "seconds": r["seconds"],
    }


def report(results: dict[str, dict[str, list[dict[str, Any]]]]) -> Path:
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "diag_memoria.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Fase 1c: diagnóstico da memória por janela",
        "",
        "MB do processo (working set no Windows). Média das execuções. `plano` = depois de planejar as janelas;",
        "`janelas: máx.` = o maior valor logo depois de receber uma janela; `pico` = o máximo do processo inteiro.",
        "",
    ]
    for case, by_variant in results.items():
        lines += [
            f"## {case}",
            "",
            "| variante | janelas | imports | plano | pico no plano | janelas: máx. | maior janela | final | pico "
            "| tempo (s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for variant, runs in by_variant.items():
            s = [_summary(r) for r in runs]
            m = {k: sum(x[k] for x in s) / len(s) for k in s[0]}
            lines.append(
                f"| {variant} | {runs[0]['windows']} | {m['imports']:.0f} | {m['plan_rss']:.0f} | {m['plan_peak']:.0f} "
                f"| {m['win_max']:.0f} | {m['win_mb']:.0f} | {m['end']:.0f} | {m['peak']:.0f} | {m['seconds']:.2f} |"
            )
        first = by_variant.get("padrao", next(iter(by_variant.values())))[0]
        lines += [
            "",
            "Janela a janela (`padrao`, 1ª execução):",
            "",
            "| janela | linhas | tamanho | ao receber | depois de alinhar | pico até aqui | commit | pool |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for w in first["per_window"]:
            private = f"{w['private']:.0f}" if w.get("private") else "—"
            lines.append(
                f"| {w['i']} | {w['rows']:,} | {w['window_mb']:.0f} | {w['rss_got']:.0f} | {w['rss_aligned']:.0f} "
                f"| {w['peak']:.0f} | {private} | {w['pool_cached_mb']:.0f} |"
            )
        lines.append("")
    path = DOCS / "diag_memoria.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+", default=list(CASES))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()
    results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for case in args.cases:
        root = dataset(case)
        exp = load_expected(root)
        expected = [exp.common, exp.left_only, exp.right_only]
        base = {"left": str(root / "left"), "right": str(root / "right"), "memory_limit": "1GB"}
        measure_once(base, {})  # aquecimento (cache de disco), descartado
        results[case] = {}
        for variant in args.variants:
            env, options = VARIANTS[variant]
            results[case][variant] = []
            for i in range(args.runs):
                r = measure_once({**base, **options}, env)
                ok = not options.get("align", True) or r["counts"] == expected
                s = _summary(r)
                results[case][variant].append(r)
                print(
                    f"  {case:<12} {variant:<15} #{i + 1}: plano {s['plan_rss']:5.0f}  janelas {s['win_max']:5.0f}  "
                    f"pico {s['peak']:5.0f} MB  {s['seconds']:6.2f} s  {'ok' if ok else 'ERRADO'}",
                    flush=True,
                )
    print(f"relatório: {report(results).relative_to(ROOT)}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(json.loads(sys.argv[2]))
    else:
        main()
