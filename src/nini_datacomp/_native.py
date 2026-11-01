"""Carrega a biblioteca C++ (libdatacomp) via ctypes e aplica as assinaturas de `_signatures.py`.

Ordem de busca:
1. variável de ambiente ``NINI_DATACOMP_LIB`` (caminho do arquivo; útil para testar um build manual);
2. a pasta do pacote (instalação normal por wheel);
3. os arquivos registrados na instalação (instalação editável do ``uv sync``, em que o pacote aponta para ``src/`` e a
   biblioteca compilada fica no site-packages).

O ctypes solta o GIL durante cada chamada, então os blocos rodam em paralelo de verdade numa ThreadPool.
Copiado do nini-datagen 0.3.0, sem a inicialização de tabelas (aqui não há tabelas a montar no carregamento).
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from importlib import metadata
from pathlib import Path

import numpy as np

from ._signatures import ABI_VERSION, SIGNATURES
from .errors import NativeLibraryError

__all__ = ["ENV_VAR", "LIB_NAMES", "lib", "library_path", "ptr"]

ENV_VAR = "NINI_DATACOMP_LIB"
if sys.platform == "win32":
    LIB_NAMES: tuple[str, ...] = ("datacomp.dll",)
elif sys.platform == "darwin":
    LIB_NAMES = ("libdatacomp.dylib", "libdatacomp.so")
else:
    LIB_NAMES = ("libdatacomp.so",)

_lock = threading.Lock()
_lib: ctypes.CDLL | None = None
_path: Path | None = None


def _candidates() -> list[Path]:
    env = os.environ.get(ENV_VAR)
    if env:
        return [Path(env)]  # pedido explícito: não procura em outro lugar
    here = Path(__file__).resolve().parent
    found = [here / name for name in LIB_NAMES]
    try:
        files = metadata.files("nini-datacomp") or []
    except metadata.PackageNotFoundError:
        files = []
    found += [Path(str(f.locate())) for f in files if f.name in LIB_NAMES]
    return found


def _load(path: Path) -> ctypes.CDLL:
    try:
        handle = ctypes.CDLL(str(path))
    except OSError as e:
        raise NativeLibraryError(f"não consegui carregar {path}: {e}") from e
    try:
        version = handle.dc_abi_version()
    except AttributeError as e:
        raise NativeLibraryError(f"{path} não é a biblioteca do nini-datacomp (falta dc_abi_version)") from e
    if version != ABI_VERSION:
        raise NativeLibraryError(
            f"{path} tem ABI {version}, mas o pacote Python espera {ABI_VERSION}: recompile"
            " (uv sync --reinstall-package nini-datacomp) ou reinstale o pacote"
        )
    missing = [name for name in SIGNATURES if not hasattr(handle, name)]
    if missing:
        raise NativeLibraryError(f"{path} não exporta {missing}")
    for name, (restype, argtypes) in SIGNATURES.items():
        fn = getattr(handle, name)
        fn.restype = restype
        fn.argtypes = argtypes
    return handle


def lib() -> ctypes.CDLL:
    """A biblioteca carregada (carrega na primeira chamada; seguro entre threads)."""
    global _lib, _path
    if _lib is not None:
        return _lib
    with _lock:
        if _lib is None:
            tried = []
            for path in _candidates():
                if path.is_file():
                    _lib, _path = _load(path), path
                    break
                tried.append(str(path))
            else:
                raise NativeLibraryError(
                    "biblioteca C++ do nini-datacomp não encontrada (procurei em: "
                    + ", ".join(tried)
                    + "). Numa cópia do repositório, rode `uv sync`; para um build manual, aponte "
                    + f"{ENV_VAR} para o arquivo."
                )
    return _lib


def library_path() -> Path:
    """Caminho do arquivo da biblioteca carregada."""
    lib()
    assert _path is not None
    return _path


def _reset() -> None:
    """Só para testes: esquece a biblioteca carregada (a próxima chamada a `lib()` procura de novo)."""
    global _lib, _path
    with _lock:
        _lib, _path = None, None


def ptr(a: np.ndarray) -> int:
    """Endereço do primeiro elemento de um array contíguo (o C++ escreve/lê nele como um vetor plano)."""
    if not a.flags.c_contiguous:
        raise ValueError("o array precisa ser contíguo (C order) para ir ao C++")
    return a.ctypes.data
