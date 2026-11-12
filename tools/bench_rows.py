"""bench_rows.py -- fase 1c: memória e tempo do caminho completo (contagem de linhas por janelas) de 1M a 10M linhas.

    uv run python tools/bench_rows.py                       # casos padrão, 5 execuções cada
    uv run python tools/bench_rows.py --cases 1000000x10 10000000x10 --runs 10 --memory-limit 512MB

O que a fase 1 promete e este benchmark confere: o pico de memória não cresce com o número de linhas (as janelas
cabem no ``memory_limit``), também com chave de texto. Cada execução roda num processo novo, como no bench_baseline,
e lê todas as colunas de cada janela (como uma comparação de verdade), mas só conta as linhas: o tempo ainda não é
comparável com o do datacompolars, que compara os valores. Ele aparece ao lado só como referência.

Casos: ``<linhas>x<colunas>`` (chave inteira, dados do make_data) e ``<linhas>x<colunas>t`` (a mesma tabela com a
chave em texto, ``"id-0000001234"``, regravada uma vez em ``data/bench/<caso>_texto``). Resultado em
``docs/benchmarks/fase1.md`` e ``fase1.json``.
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

import polars as pl
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_baseline import _self_peak_bytes
from benchstats import Stats, summarize
from make_data import Spec, generate, load_expected

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bench"
DOCS = ROOT / "docs" / "benchmarks"
BASELINE = DOCS / "baseline.json"
CASES = ("1000000x10", "1000000x50", "10000000x10", "10000000x50", "1000000x10t", "10000000x10t")
MB = 1024 * 1024


# ----------------------------------------------------------------------------- execução (subprocesso)
def worker(spec: dict[str, Any]) -> None:
    from nini_datacomp.pipeline import count_rows  # noqa: PLC0415 - import fora do cronômetro, mas no subprocesso

    t0 = time.perf_counter()
    res = count_rows(spec["left"], spec["right"], ["id"], memory_limit=spec["memory_limit"])
    seconds = time.perf_counter() - t0
    out = {
        "seconds": seconds,
        "peak_mb": _self_peak_bytes() / MB,
        "common": res.common,
        "left_only": res.left_only,
        "right_only": res.right_only,
        "windows": res.windows,
        "rows_per_window": res.rows_per_window,
        "aligned": res.aligned_to_row_groups,
        "strategies": res.strategies,
        "window_mb": res.window_bytes / MB,
        "timings": res.timings,
    }
    print("RESULT " + json.dumps(out))


def child_env(allocator: str) -> dict[str, str]:
    """O ambiente da execução: ``default`` = o mimalloc do Polars como vem; ``tuned`` = devolvendo a memória em 10 ms
    (``core.allocator.SETTINGS``). O processo que roda a ferramenta não mexe no próprio ambiente."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("MIMALLOC_")}
    if allocator == "tuned":
        env.update(MIMALLOC_PURGE_DELAY="10", MIMALLOC_RESET_DELAY="10")
    return env


def measure_once(spec: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryFile("w+b") as out_f, tempfile.TemporaryFile("w+b") as err_f:
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(spec)],
            stdout=out_f,
            stderr=err_f,
            env=child_env(spec.get("allocator", "default")),
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
            r["peak_mb"] = max(r["peak_mb"], peak / MB)
            return r
    raise RuntimeError("a execução falhou:\n" + stderr[-3000:])


# ----------------------------------------------------------------------------- dados
def text_copy(root: Path) -> Path:
    """A mesma tabela com a chave em texto ("id-0000001234": a ordem do texto é a dos números), mesmos row groups."""
    out = root.with_name(root.name + "_texto")
    if (out / "expected.json").exists():
        return out
    spec = json.loads((root / "expected.json").read_text(encoding="utf-8"))["spec"]
    for side in ("left", "right"):
        (out / side).mkdir(parents=True, exist_ok=True)
        for path in sorted((root / side).glob("*.parquet")):
            key = pl.format("id-{}", pl.col("id").cast(pl.String).str.zfill(10)).alias("id")
            df = pl.read_parquet(path).with_columns(key)
            df.write_parquet(out / side / path.name, compression="zstd", row_group_size=spec["row_group_size"])
    (out / "expected.json").write_text((root / "expected.json").read_text(encoding="utf-8"), encoding="utf-8")
    return out


def dataset(case: str) -> Path:
    text = case.endswith("t")
    rows, cols = (int(x) for x in case.rstrip("t").split("x"))
    root = generate(Spec(rows, cols), DATA)
    return text_copy(root) if text else root


# ----------------------------------------------------------------------------- relatório
def _baseline(case: str) -> dict[str, Any] | None:
    if not BASELINE.exists():
        return None
    rows, cols = case.rstrip("t").split("x")
    return json.loads(BASELINE.read_text(encoding="utf-8")).get(f"hash/{rows}/{cols}")


def _cell(values: list[float], digits: int) -> str:
    s: Stats = summarize(values)
    return f"{s.mean:.{digits}f} ± {s.half_width:.{digits}f}" if s.n > 1 else f"{s.mean:.{digits}f}"


def report(results: dict[str, list[dict[str, Any]]], memory_limit: str, allocator: str = "default") -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "fase1.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Fase 1c: memória e tempo do caminho completo (contagem de linhas)",
        "",
        f"`memory_limit = {memory_limit}`. Média ± meia largura do IC de 95%, cada execução num processo novo.",
        f"Alocador do Polars: `{allocator}` ("
        + ("`MIMALLOC_PURGE_DELAY=10`, `MIMALLOC_RESET_DELAY=10`" if allocator == "tuned" else "o mimalloc como vem")
        + "; o datacompolars foi medido sem).",
        "O nini-datacomp lê todas as colunas de cada janela, mas só conta as linhas (os valores ainda não são",
        "comparados): o tempo do datacompolars (caminho hash, que compara tudo) é só referência. O que esta fase",
        "promete é a memória não crescer com as linhas. `t` = chave em texto.",
        "",
        "| caso | janelas | linhas/janela | alinhada | tempo (s) | pico (MB) | maior janela (MB) "
        "| datacompolars: tempo | datacompolars: pico | ok |",
        "|---|---:|---:|:---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for case, runs in results.items():
        last = runs[-1]
        base = _baseline(case)
        bt = f"{base['seconds']['mean']:.3f}" if base else "—"
        bp = f"{base['peak_mb']['mean']:.0f}" if base else "—"
        ok = "✓" if all(r["ok"] for r in runs) else "✗"
        lines.append(
            f"| {case} | {last['windows']} | {last['rows_per_window']:,} | {'sim' if last['aligned'] else 'não'} "
            f"| {_cell([r['seconds'] for r in runs], 3)} | {_cell([r['peak_mb'] for r in runs], 0)} "
            f"| {last['window_mb']:.0f} | {bt} | {bp} | {ok} |"
        )
    (DOCS / "fase1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"relatório: {(DOCS / 'fase1.md').relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+", default=list(CASES))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--memory-limit", default="1GB")
    ap.add_argument(
        "--allocator", choices=("default", "tuned"), default="default",
        help="o alocador do Polars nas execuções: o mimalloc como vem, ou com MIMALLOC_PURGE_DELAY=10",
    )  # fmt: skip
    args = ap.parse_args()
    results: dict[str, list[dict[str, Any]]] = {}
    for case in args.cases:
        root = dataset(case)
        exp = load_expected(root)
        spec = {
            "left": str(root / "left"), "right": str(root / "right"), "memory_limit": args.memory_limit,
            "allocator": args.allocator,
        }  # fmt: skip
        measure_once(spec)  # aquecimento (cache de disco), descartado
        results[case] = []
        for i in range(args.runs):
            r = measure_once(spec)
            r["ok"] = (r["common"], r["left_only"], r["right_only"]) == (exp.common, exp.left_only, exp.right_only)
            results[case].append(r)
            print(
                f"  {case:<14} #{i + 1}/{args.runs}: {r['seconds']:7.3f} s  {r['peak_mb']:7.0f} MB  "
                f"{r['windows']} janelas  {'ok' if r['ok'] else 'ERRADO'}",
                flush=True,
            )
    report(results, args.memory_limit, args.allocator)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(json.loads(sys.argv[2]))
    else:
        main()
