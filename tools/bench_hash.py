"""bench_hash.py -- fase 2: o caminho exato (hash de linha) contra o caminho hash do datacompolars 0.4.4.

    uv run python tools/bench_hash.py                                   # casos padrão, 5 execuções cada
    uv run python tools/bench_hash.py --cases 10000000x300 --runs 3 --memory-limit 2GB
    uv run python tools/bench_hash.py --cases 10000000x50u 30000000x50u    # "u": chave embaralhada (fora de ordem)

Mesma medição do bench_baseline e do bench_rows: cada execução num processo novo, média e IC de 95%, pico de memória
amostrado de fora. A razão de tempo contra a linha de base (``docs/benchmarks/baseline.json``) usa a regra do plano:
``ok`` se o IC inteiro da razão fica em até 1,10. As contagens (em comum, só de um lado, divergentes) são conferidas
contra o ``expected.json`` dos dados, também por coluna. Resultado em ``docs/benchmarks/fase2.md`` e ``fase2.json``;
com ``--mode columnwise`` (``abs_tol=1e-12``, como o columnwise da linha de base), em ``fase3.md`` e ``fase3.json``.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl

from bench_baseline import _self_peak_bytes
from bench_rows import DATA, child_env, dataset
from benchstats import Stats, compare, summarize
from make_data import WIDE84, Spec, generate, load_expected

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "benchmarks"
BASELINE = DOCS / "baseline.json"
CASES = ("1000000x10", "1000000x50", "1000000x300", "10000000x10", "10000000x50", "10000000x300")
MB = 1024 * 1024
UNORDERED_CASES = ("10000000x50u", "30000000x50u")
TEXT_TAG = "texto"
_SCATTER = 999_983  # primo: id * _SCATTER % M é uma permutação dos ids (M primo maior que o maior id)


def parse_case(case: str) -> tuple[Spec, bool]:
    """``<linhas>x<colunas>[w][u]`` ou ``<linhas>x<colunas>t`` -> (especificação dos dados, fora de ordem?). ``w``:
    perfil wide84 (84 colunas, 48 exatas e 36 reais, ~6 mil linhas só da esquerda e ~400 só da direita em 22M,
    divergência de 5%); ``u``: chave embaralhada; ``t``: chave de texto (``tag`` "texto"; sem ``w``/``u``)."""
    unordered = case.endswith("u")
    base = case.removesuffix("u")
    text = base.endswith("t")
    if text:
        base = base.removesuffix("t")
        if unordered or base.endswith("w"):
            raise ValueError("a chave de texto (sufixo t) não combina com w nem u")
        rows, cols = (int(x) for x in base.split("x"))
        return Spec(rows, cols, tag=TEXT_TAG), False
    wide = base.endswith("w")
    rows, cols = (int(x) for x in base.removesuffix("w").split("x"))
    if wide:
        if cols != 84:
            raise ValueError("o perfil wide84 (sufixo w) tem 84 colunas")
        return Spec(
            rows, cols, divergence=0.05, only_left=0.00027, only_right=0.0000182, profile=WIDE84, tag=WIDE84
        ), unordered
    return Spec(rows, cols), unordered


def case_root(case: str) -> Path:
    """A pasta dos dados do caso (sem gerar nada)."""
    spec, unordered = parse_case(case)
    root = DATA / spec.label
    return root.with_name(root.name + "_unordered") if unordered else root


def case_dataset(case: str) -> Path:
    """Os dados do caso, gerados se ainda não existem (o caso ``u`` é derivado do ordenado)."""
    spec, unordered = parse_case(case)
    if spec.tag == TEXT_TAG:
        return dataset(f"{spec.rows}x{spec.cols}t")
    root = generate(spec, DATA) if spec.profile == WIDE84 else dataset(f"{spec.rows}x{spec.cols}")
    return unordered_dataset(root) if unordered else root


def unordered_dataset(root: Path) -> Path:
    """O mesmo par de tabelas com a chave embaralhada (a mesma permutação dos dois lados, então as contagens esperadas
    não mudam), arquivo a arquivo: cada row group passa a ter ids da faixa inteira."""
    out = root.with_name(root.name + "_unordered")
    if (out / "expected.json").exists():
        return out
    exp = load_expected(root)
    modulus = _next_prime(exp.left_rows + exp.right_only + 1)
    shutil.rmtree(out, ignore_errors=True)
    for side in ("left", "right"):
        (out / side).mkdir(parents=True)
        for f in sorted((root / side).glob("*.parquet")):
            df = pl.read_parquet(f).with_columns((pl.col("id") * _SCATTER % modulus).alias("id"))
            df.write_parquet(out / side / f.name, compression="zstd", statistics=True)
            print(f"  {out.name}/{side}/{f.name}", flush=True)
    shutil.copy(root / "expected.json", out / "expected.json")
    return out


def _next_prime(n: int) -> int:
    def prime(k: int) -> bool:
        return k > 1 and all(k % d for d in range(2, int(k**0.5) + 1))

    while not prime(n):
        n += 1
    return n


# ----------------------------------------------------------------------------- execução (subprocesso)
def compare_kwargs(spec: dict[str, Any], fn: Any) -> dict[str, Any]:
    """Os argumentos de ``compare_rows`` que a versão instalada aceita (a v0.3.0 não tem alguns)."""
    wanted = {
        "memory_limit": spec["memory_limit"], "abs_tol": spec["abs_tol"], "window_rows": spec.get("window_rows"),
    }  # fmt: skip
    accepted = inspect.signature(fn).parameters
    return {k: v for k, v in wanted.items() if k in accepted and v is not None}


def write_base(spec: dict[str, Any]) -> None:
    """Grava ``"<pid> <rss>"`` do Python que roda a comparação (o host): a base (Python + Polars importados) não é custo
    da comparação, e o pid separa o host do launcher do venv que o sobe (Windows)."""
    if spec.get("base_file"):
        Path(spec["base_file"]).write_text(f"{os.getpid()} {psutil.Process().memory_info().rss}", encoding="utf-8")


def worker_via_compare(spec: dict[str, Any]) -> None:
    """O caminho público (``dc.compare``, com ``isolation="auto"``): o que um usuário roda, com o processo isolado
    quando o plano pede. O pico do ``measure_once`` soma o processo e os filhos."""
    import nini_datacomp as dc  # noqa: PLC0415 - no subprocesso, fora do cronômetro

    write_base(spec)
    t0 = time.perf_counter()
    res = dc.compare(
        spec["left"], spec["right"], on="id", memory_limit=spec["memory_limit"], abs_tol=spec["abs_tol"],
        window_rows=spec.get("window_rows"), samples=0, isolation=spec.get("isolation", "auto"),
    )  # fmt: skip
    seconds = time.perf_counter() - t0
    ex = res.execution
    out = {
        "seconds": seconds,
        "peak_mb": _self_peak_bytes() / MB,
        "common": res.rows.common,
        "left_only": res.rows.left_only,
        "right_only": res.rows.right_only,
        "mismatched": res.rows.mismatched,
        "columns": {c: r.mismatches for c, r in res.columns.items() if r.mismatches},
        "windows": ex.windows,
        "groups": ex.column_groups,
        "rows_per_window": ex.rows_per_window,
        "window_mb": ex.window_bytes / MB,
        "timings": dict(ex.timings),
        "amplification": list(ex.read_amplification),
        "notes": list(ex.notes),
        "isolated": "spawn" in ex.timings,
    }
    print("RESULT " + json.dumps(out))


def worker(spec: dict[str, Any]) -> None:
    if spec.get("via_compare"):
        worker_via_compare(spec)
        return
    from nini_datacomp.pipeline import compare_rows  # noqa: PLC0415 - no subprocesso, fora do cronômetro

    if spec.get("stage_divisor") is not None:  # ... ou outra fatia (nos dois modos do alocador)
        from nini_datacomp import windows  # noqa: PLC0415

        windows.STAGE_DIVISOR = windows.DEFAULT_STAGE_DIVISOR = spec["stage_divisor"]
    if spec.get("partition") is False:  # sem o staging por janela (só para medir o que ele custa e ganha)
        from nini_datacomp import pipeline  # noqa: PLC0415

        if hasattr(pipeline, "PARTITION_THRESHOLD"):  # a v0.3.0 não tem staging: nada a desligar
            setattr(pipeline, "PARTITION_THRESHOLD", float("inf"))  # noqa: B010
    write_base(spec)
    t0 = time.perf_counter()
    res = compare_rows(spec["left"], spec["right"], ["id"], **compare_kwargs(spec, compare_rows))
    seconds = time.perf_counter() - t0
    out = {
        "seconds": seconds,
        "peak_mb": _self_peak_bytes() / MB,
        "common": res.common,
        "left_only": res.left_only,
        "right_only": res.right_only,
        "mismatched": res.mismatched,
        "columns": {c: n for c, n in res.column_mismatches.items() if n},
        "windows": res.windows,
        "pieces": getattr(res, "hash_pieces", 0),
        "groups": getattr(res, "column_groups", 1),
        "rows_per_window": getattr(res, "rows_per_window", 0),
        "window_mb": getattr(res, "window_bytes", 0) / MB,
        "timings": getattr(res, "timings", {}),
        "amplification": list(getattr(res, "read_amplification", ())),
        "notes": list(getattr(res, "notes", ())),
    }
    print("RESULT " + json.dumps(out))


def _private(process: psutil.Process) -> int:
    """Memória privada (commit) do processo: só o Windows a informa; nos outros, vale o RSS."""
    try:
        mi = process.memory_info()
    except psutil.Error:
        return 0
    return int(getattr(mi, "private", mi.rss))


def parse_base(text: str) -> tuple[int, int] | None:
    """``"<pid> <rss>"`` do arquivo de base; ``None`` se vazio ou inválido (ainda sendo escrito)."""
    parts = text.split()
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _descendants(procs: Sequence[tuple[int, int, int]], host_pid: int) -> set[int]:
    found: set[int] = set()
    frontier = {host_pid}
    while frontier:
        frontier = {pid for pid, ppid, _ in procs if ppid in frontier and pid not in found and pid != host_pid}
        found |= frontier
    return found


def split_rss(procs: Sequence[tuple[int, int, int]], host_pid: int | None) -> tuple[int, int]:
    """``(host, workers)`` em bytes, de ``procs`` = ``(pid, ppid, rss)`` da árvore inteira. Host = o rss do
    ``host_pid``; workers = a soma dos descendentes dele (com o launcher do worker). Os ancestrais do host (o
    launcher do venv do bench) ficam fora. Sem ``host_pid``: tudo é host."""
    if host_pid is None:
        return sum(rss for _, _, rss in procs), 0
    rss_of = {pid: rss for pid, _, rss in procs}
    if host_pid not in rss_of:
        return 0, 0
    return rss_of[host_pid], sum(rss_of[pid] for pid in _descendants(procs, host_pid))


def combine_peak(samples: list[tuple[int, int]], parent_base: int) -> int:
    """Pico da comparação a partir das amostras ``(pai, filhos)`` (bytes): o worker inteiro mais o que o pai CRESCE
    acima da ``parent_base`` na mesma amostra. A base do pai (Python + Polars) é do usuário, não da comparação.
    Sem filhos (inline) vale o maior ``pai``, como sempre."""
    if not any(children for _, children in samples):
        return max((parent for parent, _ in samples), default=0)
    return max(children + max(0, parent - parent_base) for parent, children in samples)


def _observe(ps: psutil.Process, host_pid: int | None, wset: dict[int, int]) -> tuple[int, int, int, int]:
    """Uma amostra da árvore de ``ps``: ``(rss do host, rss dos workers, rss da árvore, commit)``; atualiza em ``wset``
    o maior ``peak_wset`` visto por pid de worker. O commit é o dos workers somados ou, sem workers, o do host."""
    infos = {p.pid: (p, p.ppid(), p.memory_info()) for p in [ps, *ps.children(recursive=True)]}
    procs = [(pid, ppid, int(mi.rss)) for pid, (_, ppid, mi) in infos.items()]
    host, workers = split_rss(procs, host_pid)
    kids = _descendants(procs, host_pid) if host_pid is not None and host_pid in infos else set()
    for pid in kids:
        mi = infos[pid][2]
        wset[pid] = max(wset.get(pid, 0), int(getattr(mi, "peak_wset", mi.rss)))
    private = 0
    if kids:
        private = sum(_private(infos[pid][0]) for pid in kids)
    elif host_pid in infos:
        private = _private(infos[host_pid][0])
    return host, workers, sum(rss for _, _, rss in procs), private


def measure_once(spec: dict[str, Any]) -> dict[str, Any]:
    with (
        tempfile.TemporaryFile("w+b") as out_f,
        tempfile.TemporaryFile("w+b") as err_f,
        tempfile.TemporaryDirectory() as tmp,
    ):
        base_file = Path(tmp) / "parent_base"
        proc = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                json.dumps({**spec, "base_file": str(base_file)}),
            ],
            stdout=out_f,
            stderr=err_f,
            env=child_env(spec.get("allocator", "default")),
        )
        ps = psutil.Process(proc.pid)
        # No Windows, sys.executable é um launcher que sobe o Python real como filho: proc.pid NÃO é o host. O host é o
        # pid que ele grava no arquivo de base; os workers são os descendentes do host.
        samples: list[tuple[int, int]] = []  # (rss do host, rss dos workers)
        base: tuple[int, int] | None = None
        host_max = 0  # maior rss do host depois que a base existe
        total_peak = 0  # soma da árvore toda (regra antiga)
        private_peak = 0  # o maior commit: dos workers somados ou, sem workers, do host
        wset: dict[int, int] = {}  # por pid de worker, o último peak_wset visto
        while proc.poll() is None:
            try:
                if base is None:
                    base = parse_base(base_file.read_text(encoding="utf-8")) if base_file.exists() else None
                host_pid = base[0] if base else None
                host, workers, total, private = _observe(ps, host_pid, wset)
                samples.append((host, workers))
                total_peak = max(total_peak, total)
                private_peak = max(private_peak, private)
                if base:
                    host_max = max(host_max, host)
            except psutil.Error:
                break
            time.sleep(0.02)
        proc.wait()
        out_f.seek(0)
        err_f.seek(0)
        stdout = out_f.read().decode("utf-8", errors="replace")
        stderr = err_f.read().decode("utf-8", errors="replace")
        if base is None:  # o worker acabou antes da primeira amostra com a base
            try:
                base = parse_base(base_file.read_text(encoding="utf-8"))
            except OSError:
                base = None
    base_rss = base[1] if base else 0
    peak = combine_peak(samples, base_rss)
    worker_peak = max((w for _, w in samples), default=0)
    for line in stdout.splitlines():
        if line.startswith("RESULT "):
            r = json.loads(line[7:])
            r["peak_total_mb"] = max(r["peak_mb"], total_peak / MB)
            # com worker, o peak_wset do host inclui a base: só vale a regra de combine_peak
            r["peak_mb"] = peak / MB if worker_peak else max(r["peak_mb"], peak / MB)
            r["parent_base_mb"] = base_rss / MB
            r["parent_growth_mb"] = max(0, host_max - base_rss) / MB
            r["worker_peak_mb"] = worker_peak / MB
            r["worker_wset_peak_mb"] = max(wset.values(), default=0) / MB
            r["private_peak_mb"] = private_peak / MB
            return r
    raise RuntimeError("a execução falhou:\n" + stderr[-3000:])


# ----------------------------------------------------------------------------- relatório
def _baseline(case: str, mode: str) -> dict[str, Any] | None:
    if not BASELINE.exists():
        return None
    rows, cols = case.split("x")
    return json.loads(BASELINE.read_text(encoding="utf-8")).get(f"{mode}/{rows}/{cols}")


def _stats(d: dict[str, float]) -> Stats:
    return Stats(d["mean"], d["sd"], int(d["n"]))


def _cell(values: list[float], digits: int) -> str:
    s = summarize(values)
    return f"{s.mean:.{digits}f} ± {s.half_width:.{digits}f}" if s.n > 1 else f"{s.mean:.{digits}f}"


def _phases(runs: list[dict[str, Any]]) -> str:
    """Média de cada etapa (s): planejamento, espera pela leitura, alinhamento, hash, pares divergentes."""
    names = (
        "setup", "plan", "repartition", "read_wait", "read_background", "read", "align", "hash", "diff", "verify",
        "other",
    )  # fmt: skip
    parts = []
    for n in names:
        vals = [r["timings"].get(n) for r in runs if n in r["timings"]]
        if vals:
            parts.append(f"{n} {sum(vals) / len(vals):.2f}")
    return ", ".join(parts)


def report(results: dict[str, list[dict[str, Any]]], memory_limit: str, mode: str) -> Path:
    from nini_datacomp.windows import parse_memory_limit  # noqa: PLC0415

    limit_mib = parse_memory_limit(memory_limit) / MB
    DOCS.mkdir(parents=True, exist_ok=True)
    name = "fase2" if mode == "hash" else "fase3"
    if all(
        c.removesuffix(" sem staging").endswith("u") for c in results
    ):  # só casos fora de ordem: o relatório da fase 7
        name = "fase7"
    (DOCS / f"{name}.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    lines = [
        {
            "fase2": "# Fase 2: caminho exato (hash de linha)",
            "fase3": "# Fase 3: coluna a coluna (abs_tol = 1e-12)",
            "fase7": "# Fase 7: chave fora de ordem (staging por janela)",
        }[name],
        "",
        f"`memory_limit = {memory_limit}`. Média ± meia largura do IC de 95%, cada execução num processo novo. Razão =",
        "tempo do nini-datacomp / tempo do datacompolars 0.4.4 (caminho hash, `baseline.json`), com o IC de 95%",
        "(Welch, método delta); `ok` se o IC inteiro fica em até 1,10. Alocador do Polars: o mimalloc como vem ou",
        "com `MIMALLOC_PURGE_DELAY=10` (`--allocator`; o datacompolars foi medido sem).",
        "",
        f"Pico em MiB (1 MiB = 1.048.576 bytes); `cabe` = o maior pico das execuções até o memory_limit "
        f"({limit_mib:,.0f} MiB).",
        "",
        "Regra do pico: no Windows o `sys.executable` do venv é um launcher que sobe o Python real como filho, então",
        "o host é o pid gravado por ele mesmo (não o do processo lançado) e os workers são os descendentes do host",
        '(com o launcher do worker). Com worker (`isolation="process"`), pico = RSS dos workers + o quanto o host',
        "CRESCE acima da sua base (Python + Polars importados, medida antes da comparação) na mesma amostra; a base do",
        "host é do",
        "usuário, não da comparação. Sem worker (inline), pico = maior RSS do host (ou o `peak_wset` dele, se maior).",
        "`pico total` = soma do RSS da árvore inteira (conta duas vezes as DLLs do Polars e as páginas mapeadas).",
        "`worker (peak_wset)` = o maior `peak_wset` de um só processo de worker (sem a perda da amostragem a 20 ms).",
        "",
        "| caso | janelas | tempo (s) | pico (MiB) | host base | host cresc. | worker | worker (peak_wset) "
        "| pico total (MiB) | cabe | datacompolars: tempo | datacompolars: pico "
        "| razão (IC 95%) | situação | contagens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|:---:|:---:|",
    ]
    detail = ["", "Etapas (média, s; `read_wait` = a leitura que não ficou escondida atrás da comparação):", ""]
    for case, runs in results.items():
        base = _baseline(case, mode)
        seconds = [r["seconds"] for r in runs]
        ok = "✓" if all(r["ok"] for r in runs) else "✗"
        fits = "✓" if max(r["peak_mb"] for r in runs) <= limit_mib else "✗"
        if base:
            ratio, lo, hi, status = compare(summarize(seconds), _stats(base["seconds"]))
            bt, bp = f"{base['seconds']['mean']:.3f}", f"{base['peak_mb']['mean']:.0f}"
            rcell = f"{ratio:.2f} ({lo:.2f}–{hi:.2f})"
        else:
            bt = bp = rcell = status = "—"
        total_cell = _cell([r.get("peak_total_mb", r["peak_mb"]) for r in runs], 0)
        lines.append(
            f"| {case} | {runs[-1]['windows']} | {_cell(seconds, 3)} | {_cell([r['peak_mb'] for r in runs], 0)} "
            f"| {_cell([r.get('parent_base_mb', 0.0) for r in runs], 0)} "
            f"| {_cell([r.get('parent_growth_mb', 0.0) for r in runs], 0)} "
            f"| {_cell([r.get('worker_peak_mb', 0.0) for r in runs], 0)} "
            f"| {_cell([r.get('worker_wset_peak_mb', 0.0) for r in runs], 0)} "
            f"| {total_cell} | {fits} | {bt} | {bp} | {rcell} | {status} | {ok} |"
        )
        pieces = runs[-1].get("pieces", 0) / max(1, 2 * runs[-1]["windows"] * runs[-1].get("groups", 1))
        groups = runs[-1].get("groups", 1)
        amp = runs[-1].get("amplification")
        extra = f"; amplificação de leitura {amp[0]:.2f} / {amp[1]:.2f}" if amp else ""
        notes = "; ".join(runs[-1].get("notes", []))
        detail.append(
            f"- {case}: {_phases(runs)}; {groups} grupo(s) de colunas; {pieces:.1f} pedaços de memória por lado e parte"
            f"{extra}" + (f" ({notes})" if notes else "")
        )
    path = DOCS / f"{name}.md"
    path.write_text("\n".join(lines + detail) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+", default=list(CASES))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--memory-limit", default="1GB")
    ap.add_argument(
        "--no-staging", action="store_true",
        help="nos casos com chave fora de ordem, mede também sem o staging por janela (caso \"... sem staging\")",
    )  # fmt: skip
    ap.add_argument("--window-rows", type=int, default=None, help="linhas por janela (o padrão: o plano decide)")
    ap.add_argument(
        "--out-json", type=Path, default=None,
        help="grava os resultados neste JSON e não gera o relatório (não toca em docs/)",
    )  # fmt: skip
    ap.add_argument(
        "--mode", choices=("hash", "columnwise"), default="hash",
        help="columnwise: abs_tol=1e-12 (as numéricas comparadas uma a uma), contra o columnwise do datacompolars",
    )  # fmt: skip
    ap.add_argument(
        "--allocator", choices=("default", "tuned"), default="default",
        help="o alocador do Polars nas execuções: o mimalloc como vem, ou com MIMALLOC_PURGE_DELAY=10",
    )  # fmt: skip
    ap.add_argument(
        "--via-compare", action="store_true",
        help="roda pelo dc.compare (com isolation='auto': o worker, se o plano pedir) em vez de pipeline.compare_rows",
    )  # fmt: skip
    ap.add_argument(
        "--stage-divisor", type=int, default=None,
        help="o staging lê fatias de memory_limit / N (o padrão do pipeline é 10)",
    )  # fmt: skip
    args = ap.parse_args()
    if args.via_compare:  # o worker do dc.compare não aplica estas flags: ignorá-las em silêncio invalidaria a medição
        for flag, given in (
            ("--stage-divisor", args.stage_divisor is not None),
            ("--no-staging", args.no_staging),
        ):
            if given:
                ap.error(f"{flag} não vale com --via-compare (o dc.compare não aplica a flag)")
    abs_tol = 1e-12 if args.mode == "columnwise" else 0.0
    results: dict[str, list[dict[str, Any]]] = {}
    variants = [(c, True) for c in args.cases]
    if args.no_staging:
        variants += [(c, False) for c in args.cases if c.endswith("u")]
    for case, partition in variants:
        label = case if partition else f"{case} sem staging"
        root = case_dataset(case)
        exp = load_expected(root)
        want = (exp.common, exp.left_only, exp.right_only, exp.mismatched)
        spec = {
            "left": str(root / "left"), "right": str(root / "right"), "memory_limit": args.memory_limit,
            "abs_tol": abs_tol,
            "partition": partition, "window_rows": args.window_rows, "allocator": args.allocator,
            "via_compare": args.via_compare, "stage_divisor": args.stage_divisor,
        }  # fmt: skip
        want_columns = {c: n for c, n in exp.column_mismatches.items() if n}
        measure_once(spec)  # aquecimento (cache de disco), descartado
        results[label] = []
        for i in range(args.runs):
            r = measure_once(spec)
            r["ok"] = (r["common"], r["left_only"], r["right_only"], r["mismatched"]) == want
            r["ok"] = r["ok"] and r["columns"] == want_columns
            results[label].append(r)
            print(
                f"  {label:<24} #{i + 1}/{args.runs}: {r['seconds']:7.3f} s  {r['peak_mb']:7.0f} MB  "
                f"{r['windows']} janelas x {r.get('groups', 1)} grupos  {r['mismatched']:,} divergentes  "
                f"{'ok' if r['ok'] else 'ERRADO'}",
                flush=True,
            )
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"resultados: {args.out_json}")
        return
    print(f"relatório: {report(results, args.memory_limit, args.mode).relative_to(ROOT)}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(json.loads(sys.argv[2]))
    else:
        main()
