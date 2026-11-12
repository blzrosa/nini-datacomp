"""``core.allocator.mode`` e o import sem efeito colateral (a ordem de import é irrelevante)."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from nini_datacomp.core.allocator import mode


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, "default"),
        ({"MIMALLOC_PURGE_DELAY": "10"}, "tuned"),
        ({"MIMALLOC_PURGE_DELAY": "0"}, "tuned"),
        ({"MIMALLOC_PURGE_DELAY": "50"}, "tuned"),
        ({"MIMALLOC_PURGE_DELAY": "51"}, "default"),
        ({"MIMALLOC_PURGE_DELAY": "1000"}, "default"),
        ({"MIMALLOC_PURGE_DELAY": "-1"}, "default"),
        ({"MIMALLOC_PURGE_DELAY": "abc"}, "default"),
        ({"MIMALLOC_RESET_DELAY": "10"}, "default"),
        ({"NINI_DATACOMP_ALLOCATOR": "0"}, "default"),  # a variável antiga é ignorada
    ],
)
def test_mode_default_and_tuned(env: dict[str, str], expected: str) -> None:
    assert mode(env) == expected


def _child(code: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("MIMALLOC_") and k != "NINI_DATACOMP_ALLOCATOR"}
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)


@pytest.mark.parametrize("order", ["import nini_datacomp, polars", "import polars, nini_datacomp"])
def test_import_has_no_side_effects(order: str) -> None:
    code = (
        f"{order}\nimport os, nini_datacomp\n"
        "print(nini_datacomp.ALLOCATOR_STATUS, sorted(k for k in os.environ if k.startswith('MIMALLOC_')))"
    )
    out = _child(code)
    assert out.stderr == ""
    assert out.stdout.split(" ", 1) == ["default", "[]\n"]


def test_allocator_env_is_ignored() -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("MIMALLOC_")} | {"NINI_DATACOMP_ALLOCATOR": "0"}
    out = subprocess.run(
        [sys.executable, "-c", "import polars, nini_datacomp, os; print(nini_datacomp.ALLOCATOR_STATUS)"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert out.stdout.split() == ["default"]
    assert out.stderr == ""


def test_tuned_status_from_environment() -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("MIMALLOC_")} | {"MIMALLOC_PURGE_DELAY": "10"}
    out = subprocess.run(
        [sys.executable, "-c", "import nini_datacomp; print(nini_datacomp.ALLOCATOR_STATUS)"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert out.stdout.split() == ["tuned"]
