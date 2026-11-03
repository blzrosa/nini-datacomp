"""`CompareOptions`: tudo o que a comparação aceita além das duas fontes e da chave, validado (pydantic).

``extra="forbid"``: uma opção com nome errado é erro, em vez de ser ignorada. Os nomes seguem o datacompolars 0.4.4
com os ajustes da tabela de migração do README (``names`` no lugar de ``left_name``/``right_name``,
``case_sensitive_columns`` no lugar de ``lowercase_columns``, ``cast`` no lugar de ``casting`` e assim por diante).
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .windows import parse_memory_limit

__all__ = ["CompareOptions"]

Tolerance = float | dict[str, float]


class CompareOptions(BaseModel):
    """Como comparar. Todas têm um padrão; ``dc.compare(..., abs_tol=0.01)`` é o mesmo que
    ``dc.compare(..., options=dc.CompareOptions(abs_tol=0.01))``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    names: tuple[str, str] = Field(
        default=("left", "right"), description="Os rótulos dos dois lados (amostras e relatório)."
    )
    case_sensitive_columns: bool = Field(
        default=False,
        description=(
            "False: os nomes das colunas (e os de `on`, de `abs_tol`/`rel_tol` por coluna) são comparados em "
            "minúsculas, como o lowercase_columns=True do datacompolars; dois nomes que viram o mesmo são erro."
        ),
    )
    cast: Literal["none", "left", "right"] = Field(
        default="none",
        description=(
            "Colunas de tipos diferentes: 'left' converte a direita para o tipo da esquerda, 'right' o contrário, "
            "'none' só as lista em columns.type_mismatches (ficam fora da comparação; na chave, é erro)."
        ),
    )
    abs_tol: Tolerance = Field(
        default=0.0, description="Tolerância absoluta das colunas numéricas: um número, ou um por coluna (dict)."
    )
    rel_tol: Tolerance = Field(
        default=0.0, description="Tolerância relativa: igual se |l - r| <= abs_tol + rel_tol * |r|."
    )
    ignore_whitespace: bool = Field(default=False, description="Texto comparado sem os espaços em branco (\\s+).")
    ignore_case: bool = Field(default=False, description="Texto comparado em minúsculas.")
    details: bool = Field(
        default=True, description="Divergências por coluna e amostras. False: só as contagens de linhas."
    )
    samples: int = Field(default=5, ge=0, description="Linhas divergentes guardadas por coluna (0: nenhuma).")
    memory_limit: int | str = Field(
        default="auto",
        description="Orçamento de memória: bytes ou texto ('2GB', '512MiB'); 'auto' = 1/4 da RAM livre, até 2 GB.",
    )
    window_rows: int | None = Field(
        default=None, ge=1, description="Avançado: linhas por janela, no lugar do cálculo pelo memory_limit."
    )
    threads: int | None = Field(default=None, ge=1, description="Threads dos kernels C++ (None: todos os núcleos).")
    verify_hash: bool = Field(
        default=False,
        description="Confere as colunas exatas célula a célula em todos os pares (o hash de linha nunca decide só).",
    )

    @field_validator("names")
    @classmethod
    def _valid_names(cls, v: tuple[str, str]) -> tuple[str, str]:
        if not all(n.strip() for n in v):
            raise ValueError("names não pode ter rótulo vazio")
        if v[0] == v[1]:
            raise ValueError("names precisa de dois rótulos diferentes")
        return v

    @field_validator("abs_tol", "rel_tol")
    @classmethod
    def _valid_tolerance(cls, v: Tolerance) -> Tolerance:
        values = list(v.values()) if isinstance(v, dict) else [v]
        if any(not math.isfinite(x) or x < 0 for x in values):
            raise ValueError("tolerâncias devem ser números finitos e >= 0")
        return v

    @field_validator("memory_limit", mode="before")
    @classmethod
    def _valid_memory_limit(cls, v: object) -> object:
        if isinstance(v, bool) or not isinstance(v, (int, str)):
            raise ValueError("memory_limit deve ser bytes (int) ou um texto como '2GB'")
        if not (isinstance(v, str) and v.strip().lower() == "auto"):
            parse_memory_limit(v)  # o mesmo erro do planejamento, já na validação
        return v

    @field_validator("window_rows", "threads", mode="before")
    @classmethod
    def _not_bool(cls, v: object) -> object:
        if isinstance(v, bool):
            raise ValueError("deve ser um inteiro positivo")
        return v
