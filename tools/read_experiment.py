"""read_experiment.py -- fase 0c: quanto um leitor de Parquet em C++ poderia ganhar do Polars na leitura das janelas?

    uv sync --group bench                                   # pyarrow, só para este experimento
    uv run python tools/read_experiment.py                  # casos padrão (1M x 10, 50, 300 e 10M x 10, 50)
    uv run python tools/read_experiment.py --cases 1000000x10 10000000x300 --runs 5

Escrever um leitor de Parquet próprio (rodapé Thrift, páginas, dicionário, RLE, snappy, zstd) é trabalho de semanas.
Antes, este experimento mede o teto do ganho, lendo as MESMAS janelas da comparação (faixas de ids, os dois lados) de
quatro jeitos:

    polars        scan_parquet + filtro da faixa + collect: o que o nini-datacomp faria hoje
    arrow_cpp     o leitor C++ do Apache Arrow (pyarrow.dataset com o mesmo filtro, todas as threads): um leitor C++
                  maduro; se ele não ganha com folga do Polars, um leitor próprio dificilmente ganharia
    polars_raw    o Polars lendo os mesmos dados gravados sem compressão: quanto do tempo é descompressão
    bytes_floor   só ler do disco (cache do sistema) os bytes dos row groups da janela: o piso de qualquer leitor

Sugestão de regra (a decisão é da Nini, com os números): vale escrever o leitor C++ se o arrow_cpp for pelo menos 1,5x
mais rápido que o polars e a leitura for pelo menos 30% do tempo da comparação (``baseline.json``, fase 0b).

Resultado em ``docs/benchmarks/leitura.md`` e ``leitura.json``. Não rode junto com o bench_baseline: um atrapalha a
medida do outro.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import shutil
import sys
import time
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Any

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchstats import Stats, summarize
from make_data import KEY, Spec, generate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bench"
DOCS = ROOT / "docs" / "benchmarks"
BASELINE = DOCS / "baseline.json"
WINDOW_CELLS = 30_000_000  # o mesmo orçamento das janelas automáticas do datacompolars (window_rows="auto")
METHODS = ("polars", "arrow_cpp", "polars_raw", "bytes_floor")
# 10M x 300 fica de fora do padrão: são dezenas de GB, e a cópia sem compressão dobra isso
CASES = ("1000000x10", "1000000x50", "1000000x300", "10000000x10", "10000000x50")
SIDES = ("left", "right")


def windows(rows: int, cols: int, row_group: int) -> list[tuple[int, int]]:
    """Faixas ``[lo, hi)`` de ids, como as janelas automáticas do datacompolars, alinhadas aos row groups."""
    size = max(row_group, WINDOW_CELLS // (cols + 1) // row_group * row_group)
    top = rows + rows // 100 + 1  # as linhas só da direita têm ids acima de `rows` (~0,5% de `rows`)
    if rows * (cols + 1) <= WINDOW_CELLS:
        return [(0, top)]
    bounds = [*list(range(0, rows, size)), top]
    return list(pairwise(bounds))


def _arrow() -> tuple[Any, Any]:
    try:
        ds = importlib.import_module("pyarrow.dataset")
        pq = importlib.import_module("pyarrow.parquet")
    except ImportError as e:  # pragma: no cover
        raise SystemExit("este experimento precisa do pyarrow: uv sync --group bench") from e
    return ds, pq


# ----------------------------------------------------------------------------- os quatro jeitos de ler uma janela
def read_polars(folder: Path, lo: int, hi: int) -> int:
    df = pl.scan_parquet(str(folder / "*.parquet")).filter((pl.col(KEY) >= lo) & (pl.col(KEY) < hi)).collect()
    return df.height


def read_arrow(folder: Path, lo: int, hi: int) -> int:
    ds, _ = _arrow()
    field = ds.field(KEY)
    table = ds.dataset(str(folder), format="parquet").to_table(filter=(field >= lo) & (field < hi), use_threads=True)
    return table.num_rows


def byte_ranges(folder: Path, lo: int, hi: int) -> list[tuple[Path, int, int]]:
    """(arquivo, início, tamanho) das colunas de cada row group cujos ids cruzam ``[lo, hi)``, pelo rodapé."""
    _, pq = _arrow()
    out = []
    for path in sorted(folder.glob("*.parquet")):
        meta = pq.ParquetFile(path).metadata
        key = meta.schema.names.index(KEY)
        for i in range(meta.num_row_groups):
            rg = meta.row_group(i)
            st = rg.column(key).statistics
            if st is not None and st.has_min_max and (st.max < lo or st.min >= hi):
                continue
            for j in range(rg.num_columns):
                col = rg.column(j)
                start = col.dictionary_page_offset if col.has_dictionary_page else col.data_page_offset
                out.append((path, int(start), int(col.total_compressed_size)))
    return out


def read_bytes(ranges: list[tuple[Path, int, int]]) -> int:
    total = 0
    for path, start, size in ranges:
        with path.open("rb") as f:
            f.seek(start)
            total += len(f.read(size))
    return total


# ----------------------------------------------------------------------------- medida
def raw_copy(root: Path) -> Path:
    """Os mesmos arquivos, sem compressão (mesmos row groups), em ``<pasta>_raw``."""
    _, pq = _arrow()
    raw = root.with_name(root.name + "_raw")
    if (raw / "done").exists():
        return raw
    if raw.exists():
        shutil.rmtree(raw)
    for side in SIDES:
        (raw / side).mkdir(parents=True)
        for path in sorted((root / side).glob("*.parquet")):
            rg = pq.ParquetFile(path).metadata.row_group(0).num_rows
            pl.read_parquet(path).write_parquet(raw / side / path.name, compression="uncompressed", row_group_size=rg)
    (raw / "done").write_text("ok", encoding="utf-8")
    return raw


def timed(fn: Callable[[], object], runs: int) -> Stats:
    fn()  # aquecimento: cache de disco do sistema e do Polars
    samples = []
    for _ in range(runs):
        gc.collect()
        t = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t)
    return summarize(samples)


def measure(rows: int, cols: int, runs: int) -> dict[str, Any]:
    spec = Spec(rows, cols)
    root = generate(spec, DATA)
    raw = raw_copy(root)
    plan = windows(rows, cols, spec.row_group_size)
    ranges = [byte_ranges(root / side, lo, hi) for lo, hi in plan for side in SIDES]

    def every_window(read: Callable[[Path, int, int], int], base: Path) -> Callable[[], int]:
        return lambda: sum(read(base / side, lo, hi) for lo, hi in plan for side in SIDES)

    fns: dict[str, Callable[[], object]] = {
        "polars": every_window(read_polars, root),
        "arrow_cpp": every_window(read_arrow, root),
        "polars_raw": every_window(read_polars, raw),
        "bytes_floor": lambda: sum(read_bytes(r) for r in ranges),
    }
    out: dict[str, Any] = {"rows": rows, "cols": cols, "windows": len(plan)}
    for name in METHODS:
        out[name] = timed(fns[name], runs).as_dict()
        print(f"  {rows:>10,} x {cols:<3} {name:<12} {out[name]['mean']:8.3f} s", flush=True)
    out["bytes"] = sum(size for r in ranges for _, _, size in r)
    return out


# ----------------------------------------------------------------------------- relatório
def _compare_seconds(rows: int, cols: int) -> float | None:
    if not BASELINE.exists():
        return None
    case = json.loads(BASELINE.read_text(encoding="utf-8")).get(f"hash/{rows}/{cols}")
    return case["seconds"]["mean"] if case else None


def _cell(d: dict[str, float]) -> str:
    s = Stats(d["mean"], d["sd"], int(d["n"]))
    return f"{s.mean:.3f} ± {s.half_width:.3f}"


def report(results: list[dict[str, Any]]) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "leitura.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Fase 0c: leitura das janelas, Polars x leitor C++",
        "",
        "Segundos para ler todas as janelas dos dois lados (média ± meia largura do IC de 95%). `comparação` = tempo",
        "total do datacompolars 0.4.4 no caminho hash (`baseline.json`); `% leitura` = polars / comparação.",
        "",
        "| linhas | colunas | janelas | polars | arrow_cpp | polars_raw | bytes_floor | polars / arrow_cpp "
        "| comparação | % leitura |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        ratio = r["polars"]["mean"] / r["arrow_cpp"]["mean"]
        total = _compare_seconds(r["rows"], r["cols"])
        share = f"{100 * r['polars']['mean'] / total:.0f}%" if total else "—"
        lines.append(
            f"| {r['rows']:,} | {r['cols']} | {r['windows']} | {_cell(r['polars'])} | {_cell(r['arrow_cpp'])} "
            f"| {_cell(r['polars_raw'])} | {_cell(r['bytes_floor'])} | {ratio:.2f}x "
            f"| {f'{total:.3f}' if total else '—'} | {share} |"
        )
    (DOCS / "leitura.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"relatório: {(DOCS / 'leitura.md').relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+", default=list(CASES), help="linhas x colunas, ex. 1000000x50")
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()
    results = []
    for case in args.cases:
        rows, cols = (int(x.replace("_", "")) for x in case.lower().split("x"))
        print(f"{rows:,} x {cols}", flush=True)
        results.append(measure(rows, cols, args.runs))
    report(results)


if __name__ == "__main__":
    main()
