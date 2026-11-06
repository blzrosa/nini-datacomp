"""Importa o nini_datacomp antes de tudo (antes do Polars), como o pacote recomenda: o alocador do Polars sai
configurado (`core/allocator.py`) e o aviso de import fora de ordem não aparece nos testes."""

import json
import os
import sys

import psutil
import pytest

import nini_datacomp  # noqa: F401
from nini_datacomp import windows

_MEMORY_LIMIT_ENV = "NINI_TEST_MEMORY_LIMIT"
_PEAK_FILE_ENV = "NINI_TEST_PEAK_FILE"


@pytest.fixture(autouse=True)
def _test_memory_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Com ``NINI_TEST_MEMORY_LIMIT`` (ex. ``1GB``), o ``memory_limit="auto"`` vira esse limite em todos os testes."""
    raw = os.environ.get(_MEMORY_LIMIT_ENV)
    if not raw:
        return
    limit = windows.parse_memory_limit(raw)
    monkeypatch.setattr(windows, "AUTO_CAP", limit)
    monkeypatch.setattr(windows, "_available_memory", lambda: int(limit / windows.AUTO_FRACTION))


def _self_peak_bytes() -> int:
    """Pico do processo (``peak_wset`` no Windows, ``ru_maxrss`` fora dele), como ``tools/bench_baseline.py``."""
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)
    if peak:
        return int(peak)
    import resource  # noqa: PLC0415 - não existe no Windows

    getrusage = getattr(resource, "getrusage", None)
    if getrusage is None:
        return int(info.rss)
    raw = int(getrusage(getattr(resource, "RUSAGE_SELF", 0)).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    raw = os.environ.get(_MEMORY_LIMIT_ENV)
    path = os.environ.get(_PEAK_FILE_ENV)
    if not raw or not path:
        return
    record = {
        "limite": windows.parse_memory_limit(raw),
        "worker": os.environ.get("PYTEST_XDIST_WORKER", "main"),
        "pico": _self_peak_bytes(),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
