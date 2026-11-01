"""Carregamento da biblioteca C++, ABI e assinaturas geradas."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import nini_datacomp as dc
from nini_datacomp import _native
from nini_datacomp._signatures import ABI_VERSION, SIGNATURES

ROOT = Path(__file__).resolve().parents[1]


def test_abi_version_matches() -> None:
    assert _native.lib().dc_abi_version() == ABI_VERSION


def test_every_signature_is_exported() -> None:
    lib = _native.lib()
    for name in SIGNATURES:
        assert hasattr(lib, name), name


def test_library_path_exists() -> None:
    path = _native.library_path()
    assert path.is_file()
    assert path.name in _native.LIB_NAMES


@pytest.mark.skipif(not (ROOT / "tools" / "gen_bindings.py").exists(), reason="fora do repositório (wheel)")
def test_bindings_are_up_to_date() -> None:
    cmd = [sys.executable, str(ROOT / "tools" / "gen_bindings.py"), "--check"]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout + r.stderr


def test_env_var_pointing_nowhere(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with monkeypatch.context() as mp:  # ao sair, a variável volta ao valor de antes (ou some, se não existia)
        mp.setenv(_native.ENV_VAR, str(tmp_path / "nada.so"))
        _native._reset()
        try:
            with pytest.raises(dc.NativeLibraryError, match="não encontrada"):
                _native.lib()
        finally:
            _native._reset()
    assert _native.lib().dc_abi_version() == ABI_VERSION


@pytest.mark.parametrize("offset", [0, 1, 5, 8, 63, 64, 100])
@pytest.mark.parametrize("m", [0, 1, 7, 64, 65, 200, 1000])
def test_count_unset(offset: int, m: int) -> None:
    rng = np.random.default_rng(offset * 1000 + m)
    bools = rng.random(offset + m + 9) < 0.7
    bits = np.packbits(bools.astype(np.uint8), bitorder="little")
    got = _native.lib().dc_count_unset(_native.ptr(bits), offset, m)
    assert got == int((~bools[offset : offset + m]).sum())


def test_version_and_public_api() -> None:
    assert dc.__version__
    assert callable(dc.release_memory)
    assert dc.pool_limit() > 0
