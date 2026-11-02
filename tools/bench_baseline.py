"""bench_baseline.py -- mede a linha de base (datacompolars 0.4.4) na grade do plano, com repetições e IC de 95%.

    uv run python tools/bench_baseline.py                          # grade inteira (demora horas; retoma de onde parou)
    uv run python tools/bench_baseline.py --rows 10000 1000000 --cols 10 300 --runs 5
    uv run python tools/bench_baseline.py --report                 # só refaz docs/benchmarks/baseline.{md,json}

Cada execução roda num subprocesso novo (o alocador do Polars não devolve memória ao sistema, e medir várias vezes no
mesmo processo acumularia memória), como no benchmark do datacompolars. Cronometrado: leitura + comparação + relatório
em texto; os imports ficam fora. Memória: o pico do próprio processo (``peak_wset`` no Windows, ``VmHWM`` no Linux),
conferido com o maior valor que o processo pai viu ao amostrar.

Modos:
    hash        datacompolars com as configurações padrão (janelas e cache automáticos: o hash_windows_cache)
    columnwise  o mesmo com ``abs_tol=1e-12``, que leva ao caminho coluna a coluna

Repetições: ``--runs`` (padrão 30 até 1 milhão de linhas e 10 acima), mais uma de aquecimento descartada. Cada
execução vai para ``benchmarks/baseline/runs.jsonl`` assim que termina: interrompido, o benchmark continua de onde
parou. Os dados saem de `make_data.py` (``data/bench``); os de 1 milhão de linhas ou mais são apagados depois de
medidos (``--keep-data`` guarda), porque 10M x 300 ocupa dezenas de GB.

No fim (ou com ``--report``), ``docs/benchmarks/baseline.md`` (tabela) e ``baseline.json`` (média, desvio e n por
caso), que os benchmarks do nini-datacomp usam como referência.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchstats import Stats, summarize
from make_data import Spec, generate, load_expected

ROOT = Path(__file__).resolve().parents[1]
RUNS_FILE = ROOT / "benchmarks" / "baseline" / "runs.jsonl"
DOCS = ROOT / "docs" / "benchmarks"
DATA = ROOT / "data" / "bench"
ROWS = (100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000)
COLS = (10, 50, 100, 150, 200, 250, 300)
MODES = ("hash", "columnwise")
MB = 1024 * 1024
BIG = 1_000_000  # a partir daqui: 10 repetições e dados apagados depois


# ----------------------------------------------------------------------------- execução (subprocesso)
def _self_peak_bytes() -> int:
    """Pico de memória do próprio processo, só da imagem atual (sem herdar o pico do pai depois do fork+exec)."""
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)  # Windows
    if peak:
        return int(peak)
    try:  # Linux
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    import resource  # noqa: PLC0415 - macOS (no Windows nem existe; getattr para o pyright de lá)

    getrusage = getattr(resource, "getrusage", None)
    if getrusage is None:
        return int(info.rss)
    raw = int(getrusage(getattr(resource, "RUSAGE_SELF", 0)).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def worker(spec: dict[str, Any]) -> None:
    """Uma execução: imports, cronômetro em volta de comparação + relatório, resultado em JSON na saída padrão."""
    from datacompolars import CompareSettings, ReportSettings, compare  # noqa: PLC0415

    extra: dict[str, Any] = {"abs_tol": 1e-12} if spec["mode"] == "columnwise" else {}
    settings = CompareSettings(join_columns=["id"], **extra)
    t0 = time.perf_counter()
    res = compare(spec["left"], spec["right"], settings)
    res.report(ReportSettings(print_output=False))
    seconds = time.perf_counter() - t0
    out = {
        "seconds": seconds,
        "peak_mb": _self_peak_bytes() / MB,
        "common": res.rows.common,
        "mismatched": res.rows.mismatched,
        "left_only": res.rows.left_only,
        "right_only": res.rows.right_only,
        "column_mismatches": {st.column: st.mismatches for st in res.column_stats},
        "path": res.execution.path,
        "windows": res.execution.windows,
    }
    print("RESULT " + json.dumps(out))


def measure_once(spec: dict[str, Any]) -> dict[str, Any]:
    """Roda `worker` num processo novo, amostrando a memória dele de fora a cada 20 ms."""
    with tempfile.TemporaryFile("w+b") as out_f, tempfile.TemporaryFile("w+b") as err_f:
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(spec)], stdout=out_f, stderr=err_f
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


# ----------------------------------------------------------------------------- grade
def _machine() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "threads": psutil.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "python": platform.python_version(),
        "polars": metadata.version("polars"),
        "datacompolars": metadata.version("datacompolars"),
    }


def _done() -> dict[tuple[str, int, int], int]:
    """Execuções já gravadas por (modo, linhas, colunas)."""
    out: dict[tuple[str, int, int], int] = {}
    if RUNS_FILE.exists():
        for line in RUNS_FILE.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            key = (r["mode"], r["rows"], r["cols"])
            out[key] = out.get(key, 0) + 1
    return out


def _ok(r: dict[str, Any], root: Path) -> bool:
    exp = load_expected(root)
    same_rows = all(r[k] == getattr(exp, k) for k in ("common", "mismatched", "left_only", "right_only"))
    same_cols = {c: n for c, n in exp.column_mismatches.items() if n} == r["column_mismatches"]
    return same_rows and same_cols


def run_case(mode: str, rows: int, cols: int, runs: int, have: int) -> None:
    root = generate(Spec(rows, cols), DATA)
    spec = {"mode": mode, "left": str(root / "left"), "right": str(root / "right")}
    if have == 0:
        measure_once(spec)  # aquecimento (cache de disco do sistema, compilação de expressões): descartado
    machine = _machine()
    with RUNS_FILE.open("a", encoding="utf-8") as f:
        for i in range(have, runs):
            r = measure_once(spec)
            r.update(mode=mode, rows=rows, cols=cols, run=i, ok=_ok(r, root), machine=machine)
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"  {mode} {rows:>10,} x {cols:<3} #{i + 1:>2}/{runs}: {r['seconds']:8.3f} s  {r['peak_mb']:8.0f} MB")


def run_grid(args: argparse.Namespace) -> None:
    RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    done = _done()
    for rows in args.rows:
        for cols in args.cols:
            runs = args.runs or (30 if rows < BIG else 10)
            for mode in args.modes:
                have = done.get((mode, rows, cols), 0)
                if have < runs:
                    print(f"{mode} {rows:,} x {cols}: {runs - have} execuções", flush=True)
                    run_case(mode, rows, cols, runs, have)
            root = DATA / Spec(rows, cols).label
            if rows >= BIG and not args.keep_data and root.exists():
                shutil.rmtree(root)


# ----------------------------------------------------------------------------- relatório
def summary() -> dict[str, dict[str, Any]]:
    """Por caso (``modo/linhas/colunas``): estatística do tempo e do pico de memória e se todas as contagens bateram."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for line in RUNS_FILE.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        groups.setdefault(f"{r['mode']}/{r['rows']}/{r['cols']}", []).append(r)
    return {
        key: {
            "seconds": summarize([r["seconds"] for r in rs]).as_dict(),
            "peak_mb": summarize([r["peak_mb"] for r in rs]).as_dict(),
            "ok": all(r["ok"] for r in rs),
            "windows": rs[-1]["windows"],
            "path": rs[-1]["path"],
            "machine": rs[-1]["machine"],
        }
        for key, rs in groups.items()
    }


def _fmt(d: dict[str, float], digits: int) -> str:
    s = Stats(d["mean"], d["sd"], int(d["n"]))
    return f"{s.mean:.{digits}f} ± {s.half_width:.{digits}f}" if s.n > 1 else f"{s.mean:.{digits}f}"


def report() -> None:
    data = summary()
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "baseline.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    machine = next(iter(data.values()))["machine"] if data else {}
    lines = [
        "# Linha de base: datacompolars 0.4.4",
        "",
        f"Máquina: {machine.get('platform', '?')}, {machine.get('threads', '?')} threads, "
        f"{machine.get('ram_gb', '?')} GB; Python {machine.get('python', '?')}, Polars {machine.get('polars', '?')}.",
        "Média ± meia largura do IC de 95% (n execuções, cada uma num processo novo). Tempo = leitura + comparação +",
        "relatório. `ok` = contagens de linhas e por coluna iguais às da construção dos dados (`tools/make_data.py`).",
        "",
        "| modo | linhas | colunas | tempo (s) | pico (MB) | n | janelas | ok |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for key in sorted(data, key=lambda k: (k.split("/")[0], int(k.split("/")[1]), int(k.split("/")[2]))):
        mode, rows, cols = key.split("/")
        d = data[key]
        n = int(d["seconds"]["n"])
        ok = "✓" if d["ok"] else "✗"
        lines.append(
            f"| {mode} | {int(rows):,} | {cols} | {_fmt(d['seconds'], 3)} | {_fmt(d['peak_mb'], 0)} | {n} "
            f"| {d['windows']} | {ok} |"
        )
    (DOCS / "baseline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"relatório: {(DOCS / 'baseline.md').relative_to(ROOT)} ({len(data)} casos)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=lambda s: int(s.replace("_", "")), nargs="+", default=list(ROWS))
    ap.add_argument("--cols", type=int, nargs="+", default=list(COLS))
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    ap.add_argument("--runs", type=int, default=None, help="repetições por caso (padrão: 30, ou 10 a partir de 1M)")
    ap.add_argument("--keep-data", action="store_true", help="não apaga os dados de 1M linhas ou mais")
    ap.add_argument("--report", action="store_true", help="só refaz o relatório com o que já foi medido")
    args = ap.parse_args()
    if not args.report:
        run_grid(args)
    if RUNS_FILE.exists():
        report()


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(json.loads(sys.argv[2]))
    else:
        main()
