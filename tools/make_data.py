"""make_data.py -- gera pares de tabelas Parquet para os testes e o benchmark, com o resultado esperado conhecido.

    uv run python tools/make_data.py --rows 1000000 --cols 50
    uv run python tools/make_data.py --rows 10000 --cols 10 --out data/teste --divergence 0.05

Os valores vêm do nini-datagen (semente fixa: mesmos arquivos em qualquer máquina, então os dados não vão para o
repositório). O lado direito é o esquerdo com mudanças controladas, e o resultado esperado sai da própria construção
(``expected.json``), sem depender de nenhum comparador: uma terceira conferência, ao lado dos gabaritos do
datacompolars.

Layout de ``<out>/<linhas>x<colunas>[_<tag>]/``:

    left/part-00000.parquet ...    ids contíguos e crescentes por arquivo (podam row groups nas janelas por chave)
    right/part-00000.parquet ...   o mesmo trecho de ids, sem as linhas removidas e com as mudanças
    right/extra-00000.parquet ...  linhas só da direita (ids acima do maior id da esquerda)
    expected.json                  especificação + resultado esperado

Mudanças: em ``divergence`` das linhas em comum, uma coluna sorteada muda sempre e cada uma das outras muda com chance
``cell_rate``. Mudar = inteiro + 1, real + 1,0, texto + "~", booleano negado, data + 1 dia, data e hora + 1 s; uma
célula nula continua nula (nulo dos dois lados = igual), então não conta como divergência.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from string import ascii_lowercase
from typing import Any

import numpy as np
import polars as pl

KEY = "id"
CELLS_PER_FILE = 100_000_000  # células (linhas x colunas) por arquivo: limita a memória da geração

# tipos das colunas, em ciclo: 20% inteiros, 20% reais, 30% texto, 10% booleano, 10% data, 10% data e hora
KINDS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("integer", {"low": 0, "high": 1_000_000}),
    ("uniform", {}),
    ("first_name", {}),
    ("date", {}),
    ("integer", {"low": 0, "high": 1_000_000}),
    ("uniform", {}),
    ("last_name", {}),
    ("boolean", {}),
    ("cpf", {"formatted": True}),
    ("datetime", {}),
)

# perfil ``wide84``: 84 colunas, 48 exatas (12 inteiros, 12 de texto, 8 booleanos, 8 datas, 8 datas e hora) e 36 reais,
# intercalados (a cada 7 colunas: 4 exatas e 3 reais)
WIDE84 = "wide84"
_WIDE_EXACT: tuple[tuple[str, dict[str, Any]], ...] = (
    ("integer", {"low": 0, "high": 1_000_000}),
    ("first_name", {}),
    ("integer", {"low": 0, "high": 1_000_000}),
    ("date", {}),
    ("last_name", {}),
    ("boolean", {}),
    ("datetime", {}),
    ("integer", {"low": 0, "high": 1_000_000}),
    ("cpf", {"formatted": True}),
    ("date", {}),
    ("boolean", {}),
    ("datetime", {}),
)


def _wide84_kinds() -> list[tuple[str, dict[str, Any]]]:
    kinds: list[tuple[str, dict[str, Any]]] = []
    exact = 0
    for j in range(84):
        if j % 7 in (1, 3, 5):
            kinds.append(("uniform", {}))
        else:
            kinds.append(_WIDE_EXACT[exact % len(_WIDE_EXACT)])
            exact += 1
    return kinds


def column_names(n: int) -> list[str]:
    """``aa``, ``ab``, ... (``id`` fica de fora: é a chave)."""
    names = [a + b for a, b in product(ascii_lowercase, repeat=2) if a + b != KEY]
    if not 1 <= n <= len(names):
        raise ValueError(f"cols deve estar entre 1 e {len(names)}")
    return names[:n]


@dataclass(frozen=True)
class Spec:
    rows: int
    cols: int
    divergence: float = 0.01  # fração das linhas em comum com alguma célula mudada
    cell_rate: float = 0.10  # numa linha divergente, chance de cada outra coluna também mudar
    null_rate: float = 0.02  # células nulas (iguais nos dois lados)
    only_left: float = 0.005  # fração das linhas da esquerda que não vão para a direita
    only_right: float = 0.005  # linhas extras só na direita (fração das linhas da esquerda)
    seed: int = 42
    row_group_size: int | None = None  # None: o padrão do Polars (o que um arquivo de verdade costuma ter)
    tag: str = ""
    profile: str = "default"  # "wide84": 84 colunas, 48 exatas e 36 reais (ver WIDE84)

    def __post_init__(self) -> None:
        if self.profile not in ("default", WIDE84):
            raise ValueError(f"profile deve ser 'default' ou '{WIDE84}'")
        if self.profile == WIDE84 and self.cols != 84:
            raise ValueError("o perfil wide84 tem 84 colunas")
        if self.rows < 1:
            raise ValueError("rows deve ser >= 1")
        column_names(self.cols)
        for name in ("divergence", "cell_rate", "null_rate", "only_left", "only_right"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} deve estar em [0, 1]")

    @property
    def label(self) -> str:
        rg = f"_rg{self.row_group_size}" if self.row_group_size else ""
        return f"{self.rows}x{self.cols}{rg}" + (f"_{self.tag}" if self.tag else "")

    @property
    def rows_per_file(self) -> int:
        """~CELLS_PER_FILE células por arquivo; com row group fixo, um múltiplo dele (pelo menos um row group)."""
        rows = max(1, CELLS_PER_FILE // (self.cols + 1))
        rg = self.row_group_size
        return rows if rg is None else max(rg, rows // rg * rg)


@dataclass
class Expected:
    left_rows: int = 0
    right_rows: int = 0
    common: int = 0
    left_only: int = 0
    right_only: int = 0
    mismatched: int = 0
    column_mismatches: dict[str, int] = field(default_factory=dict)

    def add(self, other: Expected) -> None:
        for k in ("left_rows", "right_rows", "common", "left_only", "right_only", "mismatched"):
            setattr(self, k, getattr(self, k) + getattr(other, k))
        for c, n in other.column_mismatches.items():
            self.column_mismatches[c] = self.column_mismatches.get(c, 0) + n


def schema(spec: Spec) -> dict[str, tuple[str, dict[str, Any]]]:
    names = column_names(spec.cols)
    out: dict[str, tuple[str, dict[str, Any]]] = {}
    kinds = _wide84_kinds() if spec.profile == WIDE84 else None
    for j, name in enumerate(names):
        kind, opts = kinds[j] if kinds else KINDS[j % len(KINDS)]
        out[name] = (kind, {**opts, "null_fraction": spec.null_rate} if spec.null_rate else dict(opts))
    return out


def mutated(name: str, dtype: pl.DataType) -> pl.Expr:
    """Um valor sempre diferente do original (nulo continua nulo)."""
    c = pl.col(name)
    if dtype == pl.Boolean:
        return c.not_()
    if dtype == pl.String:
        return c + "~"
    if dtype == pl.Date:
        return c + pl.duration(days=1)
    if dtype == pl.Datetime:
        return c + pl.duration(seconds=1)
    if dtype.is_float():
        return c + 1.0
    if dtype.is_integer():
        return c + 1
    raise TypeError(f"coluna {name}: sem mudança definida para {dtype}")


def _table(spec: Spec, n: int, seed: int) -> pl.DataFrame:
    import nini_datagen as dg  # noqa: PLC0415 - só quem gera dados precisa dele

    columns: Any = schema(spec)  # o formato de esquema do nini-datagen: {nome: (gerador, opções)}
    return dg.DataGen(seed=seed).table(n, columns)


def _with_ids(df: pl.DataFrame, start: int) -> pl.DataFrame:
    return df.insert_column(0, pl.Series(KEY, np.arange(start, start + df.height, dtype=np.int64)))


def make_part(
    spec: Spec, part: int, start: int, n: int, extra_start: int
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, Expected]:
    """(esquerda, direita, extras da direita, esperado) do arquivo ``part``: ids ``start .. start + n - 1``."""
    left = _with_ids(_table(spec, n, spec.seed * 1_000_003 + 2 * part), start)
    rng = np.random.default_rng([spec.seed, part])
    divergent = rng.random(n) < spec.divergence
    forced = rng.integers(0, spec.cols, n)
    keep = rng.random(n) >= spec.only_left
    exp = Expected(left_rows=n, common=int(keep.sum()), left_only=int((~keep).sum()))

    row_mismatch = np.zeros(n, dtype=bool)
    changes = []
    for j, name in enumerate(column_names(spec.cols)):
        mask = divergent & ((forced == j) | (rng.random(n) < spec.cell_rate))
        effective = mask & keep & left[name].is_not_null().to_numpy()
        row_mismatch |= effective
        exp.column_mismatches[name] = int(effective.sum())
        changes.append(pl.when(pl.lit(pl.Series(mask))).then(mutated(name, left[name].dtype)).otherwise(pl.col(name)))
    exp.mismatched = int(row_mismatch.sum())
    right = left.with_columns(changes).filter(pl.lit(pl.Series(keep)))

    r = round(n * spec.only_right)
    extra = _with_ids(_table(spec, r, spec.seed * 1_000_003 + 2 * part + 1), extra_start) if r else left.clear()
    exp.right_only = r
    exp.right_rows = exp.common + r
    return left, right, extra, exp


def _write(df: pl.DataFrame, path: Path, spec: Spec) -> None:
    df.write_parquet(path, compression="zstd", row_group_size=spec.row_group_size, statistics=True)


def generate(spec: Spec, out: Path, *, overwrite: bool = False, log: Callable[[str], None] = print) -> Path:
    """Gera o par de tabelas em ``out/<label>`` (se ainda não existe ou com ``overwrite``) e devolve a pasta."""
    root = out / spec.label
    done = root / "expected.json"
    if done.exists() and not overwrite:
        return root
    if root.exists():
        shutil.rmtree(root)
    (root / "left").mkdir(parents=True)
    (root / "right").mkdir(parents=True)
    total = Expected()
    extra_start = spec.rows  # ids da esquerda: 0 .. rows - 1
    step = spec.rows_per_file
    for part, start in enumerate(range(0, spec.rows, step)):
        n = min(step, spec.rows - start)
        left, right, extra, exp = make_part(spec, part, start, n, extra_start)
        _write(left, root / "left" / f"part-{part:05d}.parquet", spec)
        _write(right, root / "right" / f"part-{part:05d}.parquet", spec)
        if extra.height:
            _write(extra, root / "right" / f"extra-{part:05d}.parquet", spec)
        extra_start += extra.height
        total.add(exp)
        log(f"  {spec.label}: {start + n:,}/{spec.rows:,} linhas")
    done.write_text(json.dumps({"spec": asdict(spec), "expected": asdict(total)}, indent=2), encoding="utf-8")
    return root


def load_expected(root: Path) -> Expected:
    return Expected(**json.loads((root / "expected.json").read_text(encoding="utf-8"))["expected"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=lambda s: int(s.replace("_", "")), nargs="+", required=True)
    ap.add_argument("--cols", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("data/bench"))
    ap.add_argument("--divergence", type=float, default=Spec.divergence)
    ap.add_argument("--seed", type=int, default=Spec.seed)
    ap.add_argument("--tag", default="")
    ap.add_argument("--profile", choices=("default", WIDE84), default="default")
    ap.add_argument("--only-left", type=float, default=Spec.only_left)
    ap.add_argument("--only-right", type=float, default=Spec.only_right)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    for rows in args.rows:
        for cols in args.cols:
            spec = Spec(
                rows, cols, divergence=args.divergence, seed=args.seed, tag=args.tag, profile=args.profile,
                only_left=args.only_left, only_right=args.only_right,
            )  # fmt: skip
            root = generate(spec, args.out, overwrite=args.overwrite)
            print(f"{root}: {load_expected(root)}")


if __name__ == "__main__":
    main()
