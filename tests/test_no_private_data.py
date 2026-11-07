"""Sigilo: nenhum arquivo versionado cita os termos de casos externos (a lista fica fora do repositório)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TERMS_FILE = Path(os.environ.get("NINI_PRIVATE_TERMS", Path.home() / ".nini-private-terms.txt"))


def test_no_external_case_in_repo() -> None:
    if not TERMS_FILE.exists():
        pytest.skip(f"lista de termos privados ausente ({TERMS_FILE}; ou defina NINI_PRIVATE_TERMS)")
    terms = [t.strip().lower() for t in TERMS_FILE.read_text(encoding="utf-8").splitlines() if t.strip()]
    assert terms, "a lista de termos privados está vazia"
    files = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split("\0")
    hits: list[str] = []
    for name in filter(None, files):
        path = ROOT / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8").lower()
        except (UnicodeDecodeError, OSError):
            continue
        hits += [f"{name}" for t in terms if t in text or t in name.lower()]
    assert not hits, f"arquivos com termos privados: {sorted(set(hits))}"  # nunca imprime o termo
