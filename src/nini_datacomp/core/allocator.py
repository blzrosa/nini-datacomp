"""O alocador do Polars: no Windows ele é o mimalloc, que por padrão guarda a memória liberada em vez de devolvê-la.

Medido na fase 1c (``tools/diag_memory.py``, 10M linhas, ``memory_limit = 1GB``): com as janelas de ~200 MB, o
processo ia acumulando memória janela após janela e terminava com ~1,1 GB, quase toda já liberada pelo Polars. Com
``MIMALLOC_PURGE_DELAY=0`` o mimalloc devolve a memória na hora: pico de ~0,6–0,7 GB (−40 a −48%).

Medido na fase 5 (``tools/bench_hash.py``): devolver na hora tem um custo, porque a janela seguinte pede a mesma
memória de volta ao sistema (e o Windows a entrega zerada). Com um atraso, a memória liberada ainda está lá quando é
pedida de novo. Mas o que o mimalloc guarda acompanha a velocidade com que a memória passa por ele (bytes por segundo
vezes o atraso), não o tamanho da janela: com 50 ms, as tabelas de 300 colunas (as que decodificam mais rápido)
passavam do ``memory_limit`` (até 1.122 MB com 1 GB), mesmo com janelas e grupos de colunas menores. Com 10 ms, a grade
inteira coube (pico até 767 MB com 1 GB), ainda mais rápida que com 0.

O mimalloc lê a variável uma vez, quando a biblioteca do Polars carrega; depois não há como mudar. Por isso o
``nini_datacomp`` a define no próprio import, antes de importar o Polars, e só se o Polars ainda não estiver carregado
(e sem sobrescrever um valor que já exista). Se o Polars veio antes, avisa com `AllocatorWarning`.

``NINI_DATACOMP_ALLOCATOR=0`` desliga tudo isso (nem define a variável, nem avisa).
"""

from __future__ import annotations

import os
import sys
import warnings
from collections.abc import Container, MutableMapping
from typing import Literal

from ..errors import AllocatorWarning

__all__ = ["ALLOCATOR_ENV", "PURGE_DELAY_MS", "SETTINGS", "Status", "configure"]

ALLOCATOR_ENV = "NINI_DATACOMP_ALLOCATOR"
PURGE_DELAY_MS = "10"
# RESET_DELAY: o nome antigo, mimalloc < 2.1
SETTINGS = {"MIMALLOC_PURGE_DELAY": PURGE_DELAY_MS, "MIMALLOC_RESET_DELAY": PURGE_DELAY_MS}

# "set": definida agora; "user": já estava definida; "late": o Polars já tinha carregado; "disabled": desligado
Status = Literal["set", "user", "late", "disabled"]

_LATE = (
    "o Polars foi importado antes do nini_datacomp, então o alocador dele (mimalloc) vai guardar a memória que libera "
    "e o pico de uma comparação grande pode ficar bem maior (medido: até ~1,9x). Para evitar, importe o nini_datacomp "
    "antes do Polars ou defina MIMALLOC_PURGE_DELAY=10 antes de iniciar o Python. NINI_DATACOMP_ALLOCATOR=0 silencia "
    "este aviso."
)


def configure(
    environ: MutableMapping[str, str] | None = None,
    modules: Container[str] | None = None,
    platform: str | None = None,
) -> Status:
    """Define as variáveis do mimalloc se ainda der tempo; avisa (só no Windows, onde foi medido) se não der."""
    environ = os.environ if environ is None else environ
    modules = sys.modules if modules is None else modules
    platform = sys.platform if platform is None else platform
    if environ.get(ALLOCATOR_ENV, "1") == "0":
        return "disabled"
    if "MIMALLOC_PURGE_DELAY" in environ:
        return "user"
    if "polars" in modules:
        if platform == "win32":
            warnings.warn(_LATE, AllocatorWarning, stacklevel=2)
        return "late"
    for name, value in SETTINGS.items():
        environ.setdefault(name, value)
    return "set"
