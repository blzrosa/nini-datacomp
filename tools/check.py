"""check.py -- tudo o que o CI confere, em sequência, parando no primeiro que falhar.

    uv run python tools/check.py                    # formata, lint, tipos, testes
    uv run python tools/check.py --fix              # o lint também corrige o que der (ruff check --fix)
    uv run python tools/check.py --then bench_hash  # se tudo passar, roda tools/bench_hash.py
    uv run python tools/check.py --then bench_baseline bench_rows bench_hash -- --runs 3
    uv run python tools/check.py --then diag_profile "bench_hash --cases 1000000x50"   # argumentos de cada uma

Etapas: ``ruff format .``, ``ruff check .``, ``pyright``, ``pytest`` e, com ``--then``, as ferramentas pedidas (os
argumentos depois de ``--`` vão para elas). Cada etapa roda com o Python do ambiente (``python -m ...``), então
``uv run`` recompila o C++ uma vez só, antes de tudo, se algum .cpp mudou. Uma ferramenta entre aspas leva os
próprios argumentos (``"bench_hash --cases 1000000x50"``) no lugar dos de ``--``. No fim, um resumo com o tempo de
cada etapa.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _steps(fix: bool, then: list[str], extra: list[str]) -> list[tuple[str, list[str]]]:
    py = [sys.executable, "-m"]
    steps = [
        ("ruff format", [*py, "ruff", "format", "."]),
        ("ruff check", [*py, "ruff", "check", ".", *(["--fix"] if fix else [])]),
        ("pyright", [*py, "pyright"]),
        ("pytest", [*py, "pytest"]),
    ]
    for tool in then:
        name, *own = shlex.split(tool, posix=True)
        name = Path(name).name.removesuffix(".py")  # aceita "bench_hash", "bench_hash.py" e "tools/bench_hash.py"
        script = ROOT / "tools" / f"{name}.py"
        if not script.exists():
            raise SystemExit(f"ferramenta não encontrada: tools/{name}.py")
        steps.append((name, [sys.executable, str(script), *(own or extra)]))
    return steps


def main() -> int:
    argv = sys.argv[1:]
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fix", action="store_true", help="ruff check --fix")
    ap.add_argument("--then", nargs="+", default=[], metavar="FERRAMENTA", help="tools/<nome>.py para rodar no fim")
    args = ap.parse_args(argv)

    done: list[tuple[str, float]] = []
    for name, cmd in _steps(args.fix, args.then, extra):
        print(f"\n=== {name} ===", flush=True)
        t = time.perf_counter()
        code = subprocess.call(cmd, cwd=ROOT)
        done.append((name, time.perf_counter() - t))
        if code != 0:
            print(f"\n✗ {name} falhou (código {code}); as etapas seguintes não rodaram.")
            return code
    print("\n✓ tudo certo: " + ", ".join(f"{n} ({s:.0f} s)" for n, s in done))
    return 0


if __name__ == "__main__":
    sys.exit(main())
