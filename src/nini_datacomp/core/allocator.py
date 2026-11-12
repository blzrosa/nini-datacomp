"""O alocador do Polars: no Windows ele é o mimalloc, que por padrão guarda a memória liberada (1000 ms) em vez de
devolvê-la ao sistema.

O mimalloc lê ``MIMALLOC_PURGE_DELAY`` uma vez, quando a biblioteca do Polars carrega, e o ``.pyd`` não exporta nada
para mudar isso depois. Por isso o ``nini_datacomp`` NÃO mexe no ambiente no import (a ordem de import é
irrelevante, sem aviso): ele só olha em que modo o processo está.

- ``"default"``: o alocador padrão. O planejamento usa janelas menores (``windows.DEFAULT_*``), que cabem no
  ``memory_limit`` com o que o mimalloc guarda. Só o que não cabe assim (row groups largos demais) vai para um processo
  à parte, com o alocador ajustado (``isolation``, ``_worker.py``).
- ``"tuned"``: o processo já foi iniciado com ``MIMALLOC_PURGE_DELAY`` <= 50 (ms): o planejamento usa as constantes
  calibradas para essa devolução rápida, e tudo roda no próprio processo.

Medido (2026-11-11, 10M x 50, 1 GB): purgar em 10 ms custa tempo (13,3 s e 740 MiB, contra 10,5 s e 1035 MiB no
padrão); com janelas menores o padrão cabe e ainda ganha (12,3 s, 761 MiB).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Literal

__all__ = ["PURGE_DELAY_MS", "SETTINGS", "TUNED_MAX_DELAY_MS", "Mode", "mode"]

Mode = Literal["default", "tuned"]

PURGE_DELAY_MS = "10"
# RESET_DELAY: o nome antigo, mimalloc < 2.1
SETTINGS = {"MIMALLOC_PURGE_DELAY": PURGE_DELAY_MS, "MIMALLOC_RESET_DELAY": PURGE_DELAY_MS}
TUNED_MAX_DELAY_MS = 50  # a partir daqui o mimalloc guarda memória demais para as constantes "tuned"


def mode(environ: Mapping[str, str] | None = None) -> Mode:
    """``"tuned"`` se ``MIMALLOC_PURGE_DELAY`` já existe no ambiente e é <= 50 ms; senão ``"default"``."""
    environ = os.environ if environ is None else environ
    raw = environ.get("MIMALLOC_PURGE_DELAY")
    if raw is None:
        return "default"
    try:
        delay = int(raw.strip())
    except ValueError:
        return "default"
    return "tuned" if 0 <= delay <= TUNED_MAX_DELAY_MS else "default"  # negativo = nunca devolver
