"""O processo à parte de ``isolation="process"``: ``python -m nini_datacomp._worker <pedido.pkl> <resposta.pkl>``.

Uso único: lê ``(left, right, on, opts)`` do pedido, roda ``compare`` (sempre no próprio processo, que já nasceu com o
alocador do Polars ajustado) e grava a resposta: o resultado, ou a exceção (o mesmo tipo, se serializa), e os avisos
emitidos, que o pai reemite. Ver ``isolation.py``.
"""

from __future__ import annotations

import pickle
import sys
import traceback
import warnings
from typing import Any


def _portable(obj: object) -> bool:
    try:
        pickle.loads(pickle.dumps(obj))
    except Exception:
        return False
    return True


def main(argv: list[str]) -> int:
    request, response = argv
    with open(request, "rb") as f:
        left, right, on, opts = pickle.load(f)
    from .api import compare  # noqa: PLC0415 - depois de ler o pedido: um import que falha vira resposta de erro

    payload: dict[str, Any]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            payload = {
                "status": "ok",
                "result": compare(left, right, on, options=opts.model_copy(update={"isolation": "inline"})),
            }
        except BaseException as exc:
            payload = {
                "status": "error",
                "exception": exc if _portable(exc) else None,
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(exc)),
            }
    payload["warnings"] = [(w.category if _portable(w.category) else UserWarning, str(w.message)) for w in caught]
    with open(response, "wb") as f:
        pickle.dump(payload, f)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
