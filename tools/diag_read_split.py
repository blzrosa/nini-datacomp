"""diag_read_split.py -- onde vai o tempo de leitura de uma janela: por lado, frio x quente, colunas, codec e prefetch.

    uv run python tools/diag_read_split.py --case 2000000x84w --window-rows 220000 --windows 3
    uv run python tools/diag_read_split.py --meta CAMINHO     # só codec, row groups e dtypes (só no terminal)

Por lado (esquerdo, direito) e por janela, o melhor de ``--repeat`` leituras de como o pacote lê hoje (os arquivos da
janela com o filtro da faixa de chaves):

- ``frio``: a primeira leitura do processo (o cache do sistema pode já ter os arquivos: é o teto do "disco");
  ``quente``: a melhor das seguintes (decodificação, sem disco);
- ``chave``: só as colunas-chave; ``todas``: todas as colunas (o hash exato lê todas);
- ``raw``: os mesmos arquivos regravados sem compressão (zstd x ``_raw``: o custo de descomprimir);
- ``bytes_floor``: só ler do disco os bytes dos arquivos da janela (o piso de qualquer leitor).

Depois, o ``compare_rows`` inteiro com ``prefetch`` ligado e desligado: ``read_wait`` é a leitura que a thread
principal esperou (não escondida atrás do hash). Limitar as threads da leitura já foi testado e
piorou: não se repete.

``--meta`` imprime só no terminal (nunca grava): codec, row groups e quantas colunas há de cada tipo, sem nomes.
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

MB = 1024 * 1024


def best(fn: Callable[[], Any], repeat: int) -> float:
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return min(times)


def kind_of(dtype: pl.DataType) -> str:
    if dtype.is_integer():
        return "inteiro"
    if dtype.is_float():
        return "real"
    names = {pl.Boolean: "booleano", pl.Date: "data", pl.Datetime: "data e hora", pl.String: "texto"}
    return next((n for t, n in names.items() if dtype == t), "outro")


def parquet_files(path: Path) -> list[Path]:
    return [path] if path.is_file() else sorted(path.rglob("*.parquet"))


def meta(path: Path) -> None:
    """Codec, row groups e contagem de dtypes de uma fonte; só no terminal, sem nomes de colunas."""
    from nini_datacomp.io import row_groups  # noqa: PLC0415

    files = parquet_files(path)
    if not files:
        print("nenhum arquivo Parquet")
        return
    schema = pl.read_parquet_schema(files[0])
    counts = Counter(kind_of(d) for d in schema.values())
    groups = [n for f in files for n in row_groups(f)]
    print(f"arquivos: {len(files)}; colunas: {len(schema)}; dtypes: {dict(counts)}")
    if groups:
        print(f"row groups: {len(groups)}; linhas por row group: mín {min(groups):,}, máx {max(groups):,}")
    try:
        pq = importlib.import_module("pyarrow.parquet")
        md = pq.ParquetFile(files[0]).metadata
        codecs = {md.row_group(0).column(j).compression for j in range(md.num_columns)}
        print(f"codec: {sorted(codecs)}")
    except ImportError:
        print("codec: indisponível (sem pyarrow neste ambiente)")


def raw_copy(files: list[Path], dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for f in files:
        target = dest / f.name
        pl.read_parquet(f).write_parquet(target, compression="uncompressed", statistics=True)
        out.append(target)
    return out


def read_bytes(paths: list[Path]) -> int:
    total = 0
    for p in paths:
        with p.open("rb") as fh:
            while chunk := fh.read(8 * MB):
                total += len(chunk)
    return total


def measure_window(
    src: Any, paths: list[Path], pred: Any, repeat: int, raw_dir: Path
) -> tuple[dict[str, float], float]:
    """Tempos (s) de ler a janela de vários jeitos, e o tamanho em disco dos arquivos (MB)."""

    def scan(ps: list[Path]) -> pl.LazyFrame:
        lf = src.scan(ps)
        return lf if pred is None else lf.filter(pred)

    t = time.perf_counter()
    scan(paths).collect()
    row: dict[str, float] = {"frio": time.perf_counter() - t}
    row["quente"] = best(lambda: scan(paths).collect(), repeat)
    row["chave"] = best(lambda: scan(paths).select("id").collect(), repeat)
    raws = raw_copy(paths, raw_dir)
    row["raw"] = best(lambda: scan(raws).collect(), repeat)
    row["bytes_floor"] = best(lambda: read_bytes(paths), repeat)
    shutil.rmtree(raw_dir, ignore_errors=True)
    return row, sum(p.stat().st_size for p in paths) / MB


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="2000000x84w")
    ap.add_argument("--window-rows", type=int, default=220_000)
    ap.add_argument("--windows", type=int, default=3, help="quantas janelas medir por lado")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--memory-limit", default="2GB")
    ap.add_argument("--meta", type=Path, default=None, help="só imprime codec, row groups e dtypes desta fonte")
    args = ap.parse_args()
    if args.meta:
        meta(args.meta)
        return

    from bench_hash import case_dataset  # noqa: PLC0415
    from nini_datacomp.io import open_source  # noqa: PLC0415
    from nini_datacomp.pipeline import compare_rows, prepare_keys  # noqa: PLC0415
    from nini_datacomp.windows import plan_windows  # noqa: PLC0415

    root = case_dataset(args.case)
    ls, rs, dtypes = prepare_keys(open_source(root / "left"), open_source(root / "right"), ["id"])
    plan = plan_windows(ls, rs, ["id"], args.memory_limit, window_rows=args.window_rows)
    print(
        f"{args.case}: {len(plan.windows)} janelas de ~{plan.rows_per_window:,} linhas; amplificação "
        f"{plan.read_amplification[0]:.2f} / {plan.read_amplification[1]:.2f}; {plan.column_groups} grupo(s) de colunas"
    )
    tmp = Path(tempfile.mkdtemp(prefix="diag_read_split_"))
    try:
        for side, (name, src) in enumerate((("esquerda", ls), ("direita", rs))):
            files_by_window = plan.files[side]
            totals: dict[str, float] = {}
            for w in plan.windows[: args.windows]:
                idx = files_by_window[w.index] if files_by_window is not None else tuple(range(len(src.files)))
                paths = [src.files[i] for i in idx]
                if not paths:
                    continue
                row, size = measure_window(
                    src, paths, w.predicate(["id"], dtypes), args.repeat, tmp / name / f"w{w.index}"
                )
                print(
                    f"{name} janela {w.index:>3} ({len(paths)} arq., {size:,.0f} MB): "
                    + "  ".join(f"{k} {v * 1000:6.0f} ms" for k, v in row.items()),
                    flush=True,
                )
                for k, v in row.items():
                    totals[k] = totals.get(k, 0.0) + v
            if totals:
                print(f"{name}, soma: " + "  ".join(f"{k} {v:.2f} s" for k, v in totals.items()))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\ncompare_rows inteiro, prefetch ligado x desligado:")
    for prefetch in (True, False):
        t = time.perf_counter()
        res = compare_rows(
            str(root / "left"), str(root / "right"), ["id"], memory_limit=args.memory_limit,
            window_rows=args.window_rows, prefetch=prefetch,
        )  # fmt: skip
        secs = time.perf_counter() - t
        tm = res.timings
        print(
            f"  prefetch={prefetch!s:<5} {secs:7.2f} s  read_wait {tm.get('read_wait', 0):.2f}  "
            f"read_background {tm.get('read_background', 0):.2f}  hash {tm.get('hash', 0):.2f}  "
            f"repartition {tm.get('repartition', 0):.2f}  amplificação {res.read_amplification}",
            flush=True,
        )


if __name__ == "__main__":
    main()
