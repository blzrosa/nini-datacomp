from __future__ import annotations

import os
import subprocess
import sys

import pytest

from nini_datacomp import AllocatorWarning
from nini_datacomp.core.allocator import ALLOCATOR_ENV, PURGE_DELAY_MS, SETTINGS, configure


def test_sets_when_polars_not_loaded() -> None:
    env: dict[str, str] = {}
    assert configure(env, modules={"numpy"}, platform="win32") == "set"
    assert env == SETTINGS


def test_keeps_user_value() -> None:
    env = {"MIMALLOC_PURGE_DELAY": "10"}
    assert configure(env, modules=set(), platform="win32") == "user"
    assert env == {"MIMALLOC_PURGE_DELAY": "10"}


def test_disabled() -> None:
    env = {ALLOCATOR_ENV: "0"}
    assert configure(env, modules={"polars"}, platform="win32") == "disabled"
    assert env == {ALLOCATOR_ENV: "0"}


def test_late_warns_on_windows() -> None:
    env: dict[str, str] = {}
    with pytest.warns(AllocatorWarning, match="MIMALLOC_PURGE_DELAY=10"):
        assert configure(env, modules={"polars"}, platform="win32") == "late"
    assert env == {}


@pytest.mark.filterwarnings("error")
def test_late_is_silent_elsewhere() -> None:
    assert configure({}, modules={"polars"}, platform="linux") == "late"


def _child(code: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in SETTINGS and k != ALLOCATOR_ENV}
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)


def test_import_order_in_a_fresh_process() -> None:
    code = "import os, nini_datacomp, polars; print(nini_datacomp.ALLOCATOR_STATUS, os.environ['MIMALLOC_PURGE_DELAY'])"
    first = _child(code)
    assert first.stdout.split() == ["set", PURGE_DELAY_MS]
    late = _child("import polars, nini_datacomp; print(nini_datacomp.ALLOCATOR_STATUS)")
    assert late.stdout.split() == ["late"]
    assert ("AllocatorWarning" in late.stderr) == (sys.platform == "win32")
