"""Colunas do Polars vistas como buffers do Arrow, sem cópia, para os kernels C++ lerem direto.

`import_column` pega uma ``pl.Series`` pelo ``__arrow_c_stream__`` (via `arrow_c`, sem o pyarrow) e devolve um
`Column`: endereços da validade, dos valores (ou das views, no texto) e dos buffers de texto, mais o objeto dono da
memória. Enquanto o `Column` existir, o Polars não libera os buffers.

Convenções (as mesmas que os kernels vão receber):

* ``data`` já aponta para o primeiro elemento da coluna (o deslocamento do Arrow foi aplicado), exceto em booleanos,
  que são bits: aí ``data`` aponta para o byte 0 e ``bit_offset`` diz onde começa;
* ``validity`` sempre aponta para o byte 0 do bitmap e também usa ``bit_offset`` (``None`` = sem nulos);
* texto (``string_view``, o String do Polars): ``data`` = views de 16 bytes; ``texts`` / ``text_sizes`` = endereço e
  tamanho de cada buffer de texto, na ordem do índice de buffer das views.

Os tipos aceitos são os da 0.1 (inteiros, reais, booleano, texto, Date, Datetime, Duration, Time, Decimal). Outro tipo
levanta `UnsupportedTypeError`: não há caminho lento escondido.
"""

from __future__ import annotations

import ctypes
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from .._native import lib
from ..errors import UnsupportedTypeError
from .arrow_c import Imported, import_series

__all__ = ["Column", "import_column", "value_width"]

# largura em bytes dos tipos de largura fixa, pelo formato da interface C do Arrow
_FIXED = {
    b"c": 1, b"C": 1, b"s": 2, b"S": 2, b"i": 4, b"I": 4, b"l": 8, b"L": 8,  # inteiros
    b"e": 2, b"f": 4, b"g": 8,  # reais
    b"tdD": 4, b"tdm": 8,  # Date (dias), Date (ms)
    b"tts": 4, b"ttm": 4, b"ttu": 8, b"ttn": 8,  # Time
}  # fmt: skip
_PREFIXED = {b"ts": 8, b"tD": 8}  # Datetime ("tsu:UTC"...) e Duration ("tDu"...): int64
_BOOL = b"b"
_VIEWS = (b"vu", b"vz")  # string_view e binary_view


def value_width(fmt: bytes) -> int | None:
    """Bytes por valor de um formato de largura fixa (Decimal: 16); ``None`` para bits, views ou tipo desconhecido."""
    if fmt in _FIXED:
        return _FIXED[fmt]
    for prefix, width in _PREFIXED.items():
        if fmt.startswith(prefix):
            return width
    if fmt.startswith(b"d:"):  # decimal128 "d:precisão,escala"; decimal256 traz ",256"
        return 32 if fmt.count(b",") >= 2 and fmt.rsplit(b",", 1)[1] == b"256" else 16
    return None


@dataclass(frozen=True)
class Column:
    """Os buffers de uma coluna do Polars, sem cópia (veja as convenções no topo do módulo)."""

    name: str
    dtype: pl.DataType  # o tipo do Polars (o lógico, mesmo quando os buffers vieram da representação física)
    format: bytes  # formato do Arrow dos buffers recebidos
    length: int
    null_count: int
    bit_offset: int
    validity: int | None
    data: int | None
    width: int  # bytes por valor; 16 para views; 0 para booleanos (bits)
    texts: tuple[int, ...]
    text_sizes: tuple[int, ...]
    owner: Imported

    @property
    def is_text(self) -> bool:
        return self.format in _VIEWS

    @property
    def is_bool(self) -> bool:
        return self.format == _BOOL

    def values(self, dtype: npt.DTypeLike) -> np.ndarray:
        """Os valores como array NumPy somente leitura sobre a memória do Polars (para testes e depuração)."""
        dt = np.dtype(dtype)
        if self.is_bool or self.data is None:
            raise ValueError("coluna sem buffer de valores de largura fixa")
        if self.length == 0:
            return np.empty(0, dt)
        n = self.length * self.width
        raw = (ctypes.c_uint8 * n).from_address(self.data)
        raw.owner = self.owner  # type: ignore[attr-defined]  # liga a vida da memória ao array
        out = np.frombuffer(memoryview(raw).cast("B"), dt)
        out.flags.writeable = False
        return out


def _export_error(s: pl.Series) -> str:
    """O erro do Polars ao exportar ``s`` (para a mensagem), ou vazio."""
    try:
        s.__arrow_c_stream__()
    except Exception as e:  # só para descrever o problema na mensagem
        return f" ({type(e).__name__}: {e})"
    return ""


def _single_chunk(s: pl.Series) -> Imported:
    # mais de um pedaço: junta e tenta de novo. Polars antigos (1.30) não exportam os tipos lógicos (Date, Datetime,
    # Duration, Time, Decimal) pelo stream; aí vai a representação física (dias, microssegundos...), que é a mesma
    # memória e é o que os kernels leem de qualquer jeito.
    def candidates() -> Iterator[pl.Series]:  # preguiçoso: rechunk e to_physical só quando precisa
        yield s
        yield s.rechunk()
        if not (s.dtype.is_temporal() or s.dtype == pl.Decimal):  # nunca os códigos de um Categorical/Enum
            return
        physical = s.to_physical()
        if physical.dtype != s.dtype:
            yield physical
            yield physical.rechunk()

    for candidate in candidates():
        imp = import_series(candidate)
        if imp is not None:
            return imp
    raise UnsupportedTypeError(
        f"coluna {s.name!r} ({s.dtype}): o Polars {pl.__version__} não a exportou pela interface C do Arrow"
        + _export_error(s)
    )


def import_column(s: pl.Series) -> Column:
    """Os buffers de ``s``, sem cópia. Levanta `UnsupportedTypeError` para tipos fora da lista da 0.1."""
    # codificada por dicionário: os buffers seriam códigos, e o mesmo texto tem códigos diferentes em cada tabela.
    # Recusa pelo tipo antes de exportar (o Polars 1.30 nem exporta Categorical) e de novo pelo formato exportado.
    dictionary = f"coluna {s.name!r} ({s.dtype}) é codificada por dicionário: converta para String (cast(pl.String))"
    if isinstance(s.dtype, (pl.Categorical, pl.Enum)):
        raise UnsupportedTypeError(dictionary)
    imp = _single_chunk(s)
    if imp.dictionary:
        raise UnsupportedTypeError(dictionary)
    fmt, arr = imp.format, imp.array
    off, m, nb = int(arr.offset), int(arr.length), int(arr.n_buffers)
    validity = imp.buffer(0) if nb else None
    null_count = int(arr.null_count)
    if null_count < 0:  # -1 = "não calculado" no Arrow: conta no C++
        null_count = 0 if validity is None else int(lib().dc_count_unset(validity, off, m))

    if fmt == _BOOL:
        return Column(s.name, s.dtype, fmt, m, null_count, off, validity, imp.buffer(1), 0, (), (), imp)
    if fmt in _VIEWS:
        k = nb - 3  # validade, views, k buffers de texto e, por último, o buffer com o tamanho de cada um
        views = imp.buffer(1)
        sizes: tuple[int, ...] = ()
        if k > 0:
            sizes_addr = imp.buffer(nb - 1)
            if sizes_addr is None:
                raise UnsupportedTypeError(f"coluna {s.name!r}: string_view sem o buffer de tamanhos")
            sizes = tuple(int(x) for x in (ctypes.c_int64 * k).from_address(sizes_addr))
        texts = tuple(imp.buffer(2 + j) or 0 for j in range(k))
        data = None if views is None else views + 16 * off
        return Column(s.name, s.dtype, fmt, m, null_count, off, validity, data, 16, texts, sizes, imp)
    width = value_width(fmt)
    if width is None:
        raise UnsupportedTypeError(
            f"coluna {s.name!r} ({s.dtype}, formato Arrow {fmt.decode(errors='replace')!r}) não é suportada"
        )
    values = imp.buffer(1)
    data = None if values is None else values + width * off
    return Column(s.name, s.dtype, fmt, m, null_count, off, validity, data, width, (), (), imp)
