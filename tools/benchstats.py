"""benchstats.py -- estatística dos benchmarks: média, desvio padrão e intervalo de confiança de 95%.

Copiado do nini-datagen 0.3.0. Só usa a biblioteca padrão. As medidas podem ser de qualquer unidade (aqui, segundos e
MB por execução).

Comparação com a linha de base (regra do projeto: nenhum caso mais de 10% mais lento que o datacompolars 0.4.4): a
razão ``novo / base`` tem um intervalo de 95% (Welch, que não supõe variâncias iguais). Com o limite superior até 1,10,
``ok``; com o inferior acima de 1,10, ``mais lento``; no meio, ``inconclusivo`` (o ruído não deixa decidir: rode de
novo ou com mais repetições).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["Stats", "compare", "summarize", "t975"]

# quantil 97,5% da t de Student por graus de liberdade (os intermediários por interpolação em 1/gl)
_T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 99: 1.984, 120: 1.980,
}  # fmt: skip


def t975(df: float) -> float:
    """Quantil 97,5% da t de Student com ``df`` graus de liberdade (erro < 0,005 entre os pontos da tabela)."""
    if df <= 1:
        return _T975[1]
    keys = sorted(_T975)
    if df >= keys[-1]:
        x = 1 / df  # entre 120 e infinito (1,96)
        return 1.96 + (_T975[120] - 1.96) * x * 120
    lo = max(k for k in keys if k <= df)
    hi = min(k for k in keys if k >= df)
    if lo == hi:
        return _T975[lo]
    w = (1 / lo - 1 / df) / (1 / lo - 1 / hi)
    return _T975[lo] + (_T975[hi] - _T975[lo]) * w


@dataclass(frozen=True)
class Stats:
    """Medidas de um caso (média, desvio padrão e número de repetições)."""

    mean: float
    sd: float
    n: int

    @property
    def half_width(self) -> float:
        """Meia largura do intervalo de 95% da média."""
        return t975(self.n - 1) * self.sd / math.sqrt(self.n) if self.n > 1 else float("inf")

    def __str__(self) -> str:
        return f"{self.mean:.2f} ± {self.half_width:.2f}" if self.n > 1 else f"{self.mean:.2f}"

    def as_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "sd": self.sd, "n": self.n}


def summarize(samples: Sequence[float]) -> Stats:
    xs = list(samples)
    return Stats(statistics.fmean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0, len(xs))


def compare(new: Stats, ref: Stats, tolerance: float = 1.10) -> tuple[float, float, float, str]:
    """``(razão, limite inferior, limite superior, situação)`` de ``new / ref`` com 95% (Welch, método delta)."""
    ratio = new.mean / ref.mean
    # variância da razão pelo método delta: (r^2) * (cv_novo^2 / n_novo + cv_ref^2 / n_ref)
    va = (new.sd / new.mean) ** 2 / new.n
    vb = (ref.sd / ref.mean) ** 2 / ref.n
    se = ratio * math.sqrt(va + vb)
    # graus de liberdade de Welch-Satterthwaite (sobre as variâncias relativas)
    den = (va**2 / (new.n - 1) if new.n > 1 else 0.0) + (vb**2 / (ref.n - 1) if ref.n > 1 else 0.0)
    df = (va + vb) ** 2 / den if den > 0 else float("inf")
    h = t975(df) * se
    lo, hi = ratio - h, ratio + h
    if hi <= tolerance:
        status = "ok"
    elif lo > tolerance:
        status = "mais lento"
    else:
        status = "inconclusivo"
    return ratio, lo, hi, status
