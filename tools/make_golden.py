"""make_golden.py -- grava os gabaritos dos casos de borda (`cases.py`) com o datacompolars 0.4.4.

    uv run python tools/make_golden.py            # regrava tests/golden/*.json
    uv run python tools/make_golden.py --check    # só confere se estão atualizados (o teste faz o mesmo)

O datacompolars roda sempre pelo caminho columnwise (coluna a coluna, sem hash de linha): o gabarito não depende de
colisão de hash nem da versão do Polars. Para forçar esse caminho sem mudar o resultado, os casos sem tolerância
recebem uma tolerância numa coluna que não existe (``FORCE_COLUMNWISE``).
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cases import CASES, Case

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden"
FORCE_COLUMNWISE = {"__columnwise__": 1.0}
DATACOMPOLARS = "0.4.4"


def columnwise(settings: dict[str, Any]) -> dict[str, Any]:
    """As mesmas configurações, garantindo o caminho columnwise do datacompolars."""
    out = dict(settings)
    tol = out.get("abs_tol", 0.0)
    if isinstance(tol, dict):
        out["abs_tol"] = {**tol, **FORCE_COLUMNWISE}
    elif tol == 0.0:
        out["abs_tol"] = dict(FORCE_COLUMNWISE)
    return out  # tolerância global > 0: já é columnwise


def _type_name(dtype: str) -> str:
    """Só o nome do tipo: o texto dos parâmetros muda entre versões do Polars ("Categorical" no 1.30,
    "Categorical(ordering='physical')" ou com as categorias em outras), e o gabarito não pode depender disso."""
    return dtype.split("(", 1)[0]


def run(case: Case) -> dict[str, Any]:
    """O resultado do datacompolars para o caso, num dicionário pronto para JSON."""
    from datacompolars import CompareSettings, compare  # noqa: PLC0415 - só ferramentas e testes de gabarito

    installed = metadata.version("datacompolars")
    if installed != DATACOMPOLARS:
        raise RuntimeError(f"os gabaritos são do datacompolars {DATACOMPOLARS}, mas o instalado é o {installed}")
    left, right = case.build()
    res = compare(left, right, CompareSettings(**columnwise(case.settings)))
    return {
        "case": case.name,
        "description": case.description,
        "settings": case.settings,
        "source": f"datacompolars {DATACOMPOLARS} (columnwise)",
        "aborted": res.aborted,
        "fatal_errors": len(res.fatal_errors),
        "path": res.execution.path,
        "rows": {
            k: getattr(res.rows, k)
            for k in ("left_total", "right_total", "common", "left_only", "right_only", "mismatched")
        },
        "columns": {
            "compared": res.columns.compared,
            "left_only": res.columns.left_only,
            "right_only": res.columns.right_only,
            "type_mismatches": [
                [t.column, _type_name(t.left), _type_name(t.right)] for t in res.columns.type_mismatches
            ],
        },
        "column_mismatches": {st.column: st.mismatches for st in res.column_stats},
        "samples": {t.column: t.rows for t in res.samples},
    }


def dumps(data: dict[str, Any]) -> str:
    # default=str: datas, decimais e afins viram texto; NaN e infinitos ficam como no JSON do Python
    return json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n"


def stale() -> list[str]:
    """Casos cujo gabarito não existe ou não bate com o datacompolars."""

    def current(case: Case) -> bool:
        path = GOLDEN / f"{case.name}.json"
        return path.exists() and path.read_text(encoding="utf-8") == dumps(run(case))

    return [case.name for case in CASES if not current(case)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        bad = stale()
        print(f"desatualizados: {bad}" if bad else f"ok: {len(CASES)} casos")
        return 1 if bad else 0
    GOLDEN.mkdir(parents=True, exist_ok=True)
    names = {c.name for c in CASES}
    for old in GOLDEN.glob("*.json"):
        if old.stem not in names:
            old.unlink()
    for case in CASES:
        (GOLDEN / f"{case.name}.json").write_text(dumps(run(case)), encoding="utf-8", newline="\n")
    print(f"{len(CASES)} gabaritos em {GOLDEN.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
