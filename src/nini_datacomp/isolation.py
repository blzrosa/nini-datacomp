"""Onde a comparação roda: no próprio processo ou num processo à parte (``_worker.py``).

O alocador do Polars (mimalloc) lê ``MIMALLOC_PURGE_DELAY`` uma vez, quando carrega, e o ``nini_datacomp`` não mexe no
ambiente no import (``core/allocator.py``). Com o alocador padrão, uma tabela cujo row group já não cabe no
``memory_limit`` (ex.: 300 colunas) só cabe com o alocador ajustado: então a comparação roda num processo novo, iniciado
com ``MIMALLOC_PURGE_DELAY=10``. Um processo por comparação (sem worker persistente): custa ~0,5-1 s de ``spawn``.

``isolation="auto"``: no processo à parte só se o alocador é o padrão e ``Plan.fits_inline`` é falso. Sempre no próprio
processo (``inline_reason``): fonte que não é arquivo, comparador que não serializa, programa congelado, e o próprio
worker. A mesma exceção volta ao chamador; os avisos do worker são reemitidos aqui.
"""

from __future__ import annotations

import dataclasses
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .core.allocator import SETTINGS, Mode

if TYPE_CHECKING:
    from .options import CompareOptions
    from .results import ComparisonResult

__all__ = [
    "WORKER_ENV",
    "NeedsIsolation",
    "WorkerUnavailable",
    "inline_reason",
    "run_in_worker",
    "use_process",
]

WORKER_ENV = "NINI_DATACOMP_WORKER"
ISOLATED_NOTE = "executado em processo isolado (alocador do Polars ajustado)"


class NeedsIsolation(Exception):  # noqa: N818 - sinal interno, não um erro
    """O plano diz que, neste processo, nem um row group cabe no memory_limit (``Plan.fits_inline`` falso)."""


class WorkerUnavailable(Exception):  # noqa: N818 - o worker não iniciou ou não respondeu: a comparação segue inline
    pass


def inline_reason(
    left: object,
    right: object,
    opts: CompareOptions,
    *,
    environ: Mapping[str, str] | None = None,
    frozen: bool | None = None,
) -> str | None:
    """Por que esta comparação tem de rodar no próprio processo (``None``: pode ir para o worker)."""
    environ = os.environ if environ is None else environ
    frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    if environ.get(WORKER_ENV) == "1":
        return "já é o processo isolado"
    if frozen:
        return "programa congelado"
    for side, src in (("esquerda", left), ("direita", right)):
        if not isinstance(src, (str, os.PathLike)):
            return f"a fonte da {side} está na memória"
    try:
        pickle.dumps(opts)
    except Exception:
        return "as opções não serializam (comparador que não é uma função de módulo)"
    return None


def use_process(isolation: str, allocator: Mode, fits_inline: bool, reason: str | None) -> bool:
    """A tabela de decisão: ``True`` = processo à parte."""
    if isolation == "inline" or reason is not None:
        return False
    if isolation == "process":
        return True
    return allocator == "default" and not fits_inline


def with_note(result: ComparisonResult, note: str) -> ComparisonResult:
    execution = dataclasses.replace(result.execution, notes=(*result.execution.notes, note))
    return dataclasses.replace(result, execution=execution)


def _tail(text: str, lines: int = 5) -> str:
    return " | ".join(text.strip().splitlines()[-lines:])


def run_in_worker(left: object, right: object, on: object, opts: CompareOptions, *, t0: float) -> ComparisonResult:
    """Roda ``compare`` num processo novo (alocador ajustado) e devolve o resultado, com ``timings["spawn"]`` e a
    nota. A pasta temporária do pedido e da resposta é apagada sempre; se o pai for interrompido, o filho é morto
    (``subprocess.run``). ``WorkerUnavailable`` se o filho não inicia ou não responde."""
    scratch = Path(tempfile.mkdtemp(prefix="nini_datacomp_worker_"))
    try:
        request, response = scratch / "request.pkl", scratch / "response.pkl"
        try:
            request.write_bytes(pickle.dumps((left, right, on, opts)))
        except Exception as exc:
            raise WorkerUnavailable(f"o pedido não serializa: {exc}") from exc
        env = {**os.environ, **SETTINGS, WORKER_ENV: "1"}
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "nini_datacomp._worker", str(request), str(response)],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise WorkerUnavailable(f"o processo não iniciou: {exc}") from exc
        elapsed = time.perf_counter() - started
        if not response.exists():
            raise WorkerUnavailable(
                f"o processo terminou (código {proc.returncode}) sem resposta: {_tail(proc.stderr)}"
            )
        payload: dict[str, Any] = pickle.loads(response.read_bytes())
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    for category, message in payload["warnings"]:
        warnings.warn(message, category, stacklevel=3)
    if payload["status"] == "error":
        exc = payload["exception"]
        if exc is None:
            exc = RuntimeError(f"{payload['type']}: {payload['message']}")
        raise exc from RuntimeError(f"no processo isolado:\n{payload['traceback']}")
    result: ComparisonResult = payload["result"]
    execution = result.execution
    spawn = max(0.0, elapsed - execution.total_seconds)
    execution = dataclasses.replace(
        execution,
        total_seconds=round(time.perf_counter() - t0, 6),
        timings={**execution.timings, "spawn": round(spawn, 6)},
        notes=(*execution.notes, ISOLATED_NOTE),
    )
    return dataclasses.replace(result, execution=execution)
