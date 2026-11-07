"""relayout.py -- regrava um par de tabelas sintéticas com outra estrutura de arquivos, sem mudar os dados.

    uv run python tools/relayout.py data/bench/2000000x84_wide84 data/bench/2000000x84_wide84_ly \
        --left-row-group-size 110000 --right-rows-per-file 123000

Esquerda: um arquivo só, com row groups de ``--left-row-group-size`` linhas. Direita: arquivos de
``--right-rows-per-file`` linhas, um row group cada. Os números vêm sempre da linha de comando (nada de perfil
fixo no código). ZSTD como o make_data. Com ``--shard-right N``, a direita vira N arquivos por ``id % N`` (row groups de
``--right-rows-per-file``) em vez de faixas contíguas: cada arquivo cobre a faixa inteira de
chaves (a poda por faixa não descarta nada).
O ``expected.json`` é copiado: as contagens esperadas não mudam.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import polars as pl


def _files(folder: Path) -> list[Path]:
    """Os arquivos na ordem das chaves: ``part-*`` e depois ``extra-*`` (ids acima do maior da esquerda)."""
    return sorted(folder.glob("*.parquet"), key=lambda p: (p.name.startswith("extra"), p.name))


def relayout(src: Path, out: Path, *, left_row_group_size: int, right_rows_per_file: int, shard_right: int = 0) -> Path:
    shutil.rmtree(out, ignore_errors=True)
    (out / "left").mkdir(parents=True)
    (out / "right").mkdir(parents=True)
    pl.scan_parquet(_files(src / "left")).sink_parquet(
        out / "left" / "part-00000.parquet", compression="zstd", row_group_size=left_row_group_size, statistics=True
    )
    right = pl.scan_parquet(_files(src / "right"))
    total = int(right.select(pl.len()).collect().item())
    files = shard_right or max(1, -(-total // right_rows_per_file))
    for k in range(files):
        part = (
            right.filter(pl.col("id") % shard_right == k % shard_right)
            if shard_right
            else right.slice(k * right_rows_per_file, right_rows_per_file)
        )
        part.sink_parquet(
            out / "right" / f"shard-{k:05d}.parquet", compression="zstd", row_group_size=right_rows_per_file,
            statistics=True,
        )  # fmt: skip
    shutil.copy(src / "expected.json", out / "expected.json")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--left-row-group-size", type=int, required=True)
    ap.add_argument("--right-rows-per-file", type=int, required=True)
    ap.add_argument("--shard-right", type=int, default=0, help="N > 0: divide a direita por id %% N (faixas cheias)")
    a = ap.parse_args()
    print(relayout(a.src, a.out, left_row_group_size=a.left_row_group_size,
                   right_rows_per_file=a.right_rows_per_file, shard_right=a.shard_right))  # fmt: skip


if __name__ == "__main__":
    main()
