"""Interface C do Arrow (Arrow C Data Interface) por PyCapsule, só com ctypes: sem o pyarrow.

Duas direções:

* **exportar** (`export_array`): buffers que os kernels escreveram (arrays NumPy) viram um objeto com
  ``__arrow_c_array__``, que o Polars importa sem copiar (``pl.Series(obj)``). Os buffers ficam vivos até o Polars
  chamar o ``release`` (quando a Series deixa de existir); aí as referências são soltas e a memória volta ao pool.
* **importar** (`import_series`): uma Series do Polars entregue por ``__arrow_c_stream__``, para o C++ ler os buffers
  das duas tabelas comparadas direto, sem copiar.

Copiado do nini-datagen 0.3.0 sem mudanças de comportamento.

Estruturas e regras: https://arrow.apache.org/docs/format/CDataInterface.html e .../CStreamInterface.html. Quem recebe
uma estrutura "move" (copia e zera o ``release`` da original) e chama o ``release`` da cópia quando termina; por isso o
que mantém os buffers vivos fica num registro indexado por ``private_data``, que viaja junto com a cópia.

O ``release`` é uma função Python chamada pelo Polars (o ctypes pega o GIL sozinho), como acontecia com os buffers
do NumPy entregues pelo pyarrow.
"""

from __future__ import annotations

import ctypes
import itertools
import threading
from collections.abc import Sequence
from ctypes import CFUNCTYPE, POINTER, Structure, c_char_p, c_int, c_int64, c_void_p
from typing import Any

__all__ = [
    "ArrowArray",
    "ArrowSchema",
    "Exported",
    "Imported",
    "ImportedBatch",
    "export_array",
    "import_batch",
    "import_series",
]


class ArrowSchema(Structure):
    pass


class ArrowArray(Structure):
    pass


class ArrowArrayStream(Structure):
    pass


SchemaRelease = CFUNCTYPE(None, POINTER(ArrowSchema))
ArrayRelease = CFUNCTYPE(None, POINTER(ArrowArray))
StreamGetSchema = CFUNCTYPE(c_int, POINTER(ArrowArrayStream), POINTER(ArrowSchema))
StreamGetNext = CFUNCTYPE(c_int, POINTER(ArrowArrayStream), POINTER(ArrowArray))
StreamGetLastError = CFUNCTYPE(c_char_p, POINTER(ArrowArrayStream))
StreamRelease = CFUNCTYPE(None, POINTER(ArrowArrayStream))

ArrowSchema._fields_ = [
    ("format", c_char_p),
    ("name", c_char_p),
    ("metadata", c_char_p),
    ("flags", c_int64),
    ("n_children", c_int64),
    ("children", POINTER(POINTER(ArrowSchema))),
    ("dictionary", POINTER(ArrowSchema)),
    ("release", SchemaRelease),
    ("private_data", c_void_p),
]
ArrowArray._fields_ = [
    ("length", c_int64),
    ("null_count", c_int64),
    ("offset", c_int64),
    ("n_buffers", c_int64),
    ("n_children", c_int64),
    ("buffers", POINTER(c_void_p)),
    ("children", POINTER(POINTER(ArrowArray))),
    ("dictionary", POINTER(ArrowArray)),
    ("release", ArrayRelease),
    ("private_data", c_void_p),
]
ArrowArrayStream._fields_ = [
    ("get_schema", StreamGetSchema),
    ("get_next", StreamGetNext),
    ("get_last_error", StreamGetLastError),
    ("release", StreamRelease),
    ("private_data", c_void_p),
]

_FLAG_NULLABLE = 2
_SCHEMA_NAME = b"arrow_schema"
_ARRAY_NAME = b"arrow_array"
_STREAM_NAME = b"arrow_array_stream"

# ------------------------------------------------------------------ PyCapsule pela API C do Python
_api = ctypes.pythonapi
_capsule_new = _api.PyCapsule_New
_capsule_new.restype = ctypes.py_object
_capsule_new.argtypes = [c_void_p, c_char_p, c_void_p]
_capsule_get = _api.PyCapsule_GetPointer
_capsule_get.restype = c_void_p
_capsule_get.argtypes = [ctypes.py_object, c_char_p]
# no destrutor a cápsula já está sendo destruída: recebe o ponteiro cru (sem criar outra referência a ela)
_capsule_get_raw = ctypes.PYFUNCTYPE(c_void_p, c_void_p, c_char_p)(("PyCapsule_GetPointer", _api))
CapsuleDestructor = CFUNCTYPE(None, c_void_p)

# o que mantém os buffers de cada estrutura exportada vivos, por private_data (solto no release)
_live: dict[int, tuple[Any, ...]] = {}
# a memória das próprias estruturas apontadas por cápsulas, pelo endereço (solta no destrutor da cápsula: o release
# pode vir antes, de uma cópia movida, e o destrutor ainda lê a estrutura original)
_in_capsule: dict[int, Any] = {}
_live_lock = threading.Lock()
_ids = itertools.count(1)


def _keep(obj: tuple[Any, ...]) -> int:
    key = next(_ids)
    with _live_lock:
        _live[key] = obj
    return key


def _drop(key: int | None) -> None:
    if key:
        with _live_lock:
            _live.pop(key, None)


def _release_schema(p: Any) -> None:
    s = p.contents
    key = s.private_data
    s.release = SchemaRelease()  # NULL: liberada
    _drop(key)


def _release_array(p: Any) -> None:
    a = p.contents
    key = a.private_data
    a.release = ArrayRelease()
    _drop(key)


_release_schema_c = SchemaRelease(_release_schema)
_release_array_c = ArrayRelease(_release_array)


def _destroy_schema_capsule(cap: int) -> None:
    try:
        p = _capsule_get_raw(cap, _SCHEMA_NAME)
        if p:
            s = ArrowSchema.from_address(p)
            if s.release:  # ninguém tomou posse: libera aqui
                _release_schema(ctypes.pointer(s))
            with _live_lock:
                _in_capsule.pop(p, None)
    except Exception:  # um destrutor não pode levantar exceção
        pass


def _destroy_array_capsule(cap: int) -> None:
    try:
        p = _capsule_get_raw(cap, _ARRAY_NAME)
        if p:
            a = ArrowArray.from_address(p)
            if a.release:
                _release_array(ctypes.pointer(a))
            with _live_lock:
                _in_capsule.pop(p, None)
    except Exception:  # idem
        pass


_destroy_schema_c = CapsuleDestructor(_destroy_schema_capsule)
_destroy_array_c = CapsuleDestructor(_destroy_array_capsule)
_destroy_schema_ptr = ctypes.cast(_destroy_schema_c, c_void_p)
_destroy_array_ptr = ctypes.cast(_destroy_array_c, c_void_p)


def _address(buf: Any) -> int | None:
    """Endereço de um buffer: array NumPy (contíguo), objeto do ctypes ou None (buffer ausente, ex. sem nulos)."""
    if buf is None:
        return None
    data = getattr(buf, "ctypes", None)
    if data is not None:
        if not buf.flags.c_contiguous:
            raise ValueError("buffer não contíguo")
        return int(data.data)
    return ctypes.addressof(buf)


class Exported:
    """Um array exportado: entregue ao Polars com ``pl.Series(obj)`` (protocolo ``__arrow_c_array__``)."""

    __slots__ = ("_array", "_schema", "_used")

    def __init__(self, schema: ArrowSchema, array: ArrowArray) -> None:
        self._schema = schema
        self._array = array
        self._used = False

    def __arrow_c_array__(self, requested_schema: object = None) -> tuple[object, object]:
        if self._used:
            raise RuntimeError("este array do Arrow já foi entregue")
        self._used = True
        sa, aa = ctypes.addressof(self._schema), ctypes.addressof(self._array)
        with _live_lock:
            _in_capsule[sa] = self._schema
            _in_capsule[aa] = self._array
        return _capsule_new(sa, _SCHEMA_NAME, _destroy_schema_ptr), _capsule_new(aa, _ARRAY_NAME, _destroy_array_ptr)

    def __del__(self) -> None:  # nunca entregue: solta os buffers
        if not self._used:
            if self._schema.release:
                _release_schema(ctypes.pointer(self._schema))
            if self._array.release:
                _release_array(ctypes.pointer(self._array))


def export_array(fmt: bytes, length: int, buffers: Sequence[Any], null_count: int) -> Exported:
    """Array do Arrow sem filhos, com formato ``fmt`` (ex. ``b"l"`` int64, ``b"vu"`` string_view, ``b"b"`` bool),
    ``length`` linhas e os ``buffers`` na ordem do formato (validade primeiro; ``None`` = ausente). Os buffers ficam
    vivos até o consumidor liberar o array."""
    fmt_b = bytes(fmt)
    schema = ArrowSchema()
    schema.format = fmt_b
    schema.name = b""
    schema.metadata = None
    schema.flags = _FLAG_NULLABLE
    schema.n_children = 0
    schema.release = _release_schema_c
    schema.private_data = _keep((fmt_b,))

    addrs = (c_void_p * len(buffers))(*[_address(b) for b in buffers])
    array = ArrowArray()
    array.length = length
    array.null_count = null_count
    array.offset = 0
    array.n_buffers = len(buffers)
    array.n_children = 0
    array.buffers = ctypes.cast(addrs, POINTER(c_void_p))
    array.release = _release_array_c
    array.private_data = _keep((addrs, tuple(buffers)))
    return Exported(schema, array)


# ------------------------------------------------------------------ importar (ler uma Series do Polars)
class Imported:
    """Um array importado de um stream do Arrow (um pedaço só), dono dos buffers enquanto existir."""

    __slots__ = ("array", "dictionary", "format", "stream")

    def __init__(self, fmt: bytes, array: ArrowArray, stream: ArrowArrayStream, dictionary: bool = False) -> None:
        self.format = fmt
        self.dictionary = dictionary  # codificado por dicionário (Categorical, Enum): os buffers são códigos
        self.array = array
        self.stream = stream

    def buffer(self, i: int) -> int | None:
        """Endereço do i-ésimo buffer (None se ausente)."""
        if i >= self.array.n_buffers:
            raise IndexError(i)
        return self.array.buffers[i] or None

    def __del__(self) -> None:
        if self.array.release:
            self.array.release(ctypes.byref(self.array))
        if self.stream.release:
            self.stream.release(ctypes.byref(self.stream))


def _take_stream(s: Any) -> ArrowArrayStream | None:
    """O stream do Arrow da Series, já movido para uma estrutura nossa (ou None se o Polars não exporta)."""
    get_stream = getattr(s, "__arrow_c_stream__", None)
    if get_stream is None:
        return None
    try:
        cap = get_stream()
        p = _capsule_get(cap, _STREAM_NAME)
    except Exception:  # Polars antigo ou tipo sem exportação: o chamador decide (rechunk, erro claro)
        return None
    if not p:
        return None
    stream = ArrowArrayStream()
    ctypes.memmove(ctypes.byref(stream), p, ctypes.sizeof(ArrowArrayStream))
    ArrowArrayStream.from_address(p).release = StreamRelease()  # tomou posse: a cápsula não libera mais
    return stream


def _format_of(stream: ArrowArrayStream) -> tuple[bytes, bool] | None:
    """(formato, se é codificado por dicionário) da coluna do stream."""
    schema = ArrowSchema()
    if stream.get_schema(ctypes.byref(stream), ctypes.byref(schema)) != 0:
        return None
    fmt = bytes(schema.format or b"")
    dictionary = bool(schema.dictionary)
    if schema.release:
        schema.release(ctypes.byref(schema))
    return fmt, dictionary


def import_series(s: Any) -> Imported | None:
    """A Series como um array do Arrow de um pedaço só, pelo ``__arrow_c_stream__`` do Polars (sem copiar), ou None
    (Polars sem o protocolo, ou a Series em mais de um pedaço)."""
    stream = _take_stream(s)
    if stream is None:
        return None
    info = _format_of(stream)
    first = ArrowArray()
    if info is None or stream.get_next(ctypes.byref(stream), ctypes.byref(first)) != 0 or not first.release:
        stream.release(ctypes.byref(stream))
        return None
    out = Imported(info[0], first, stream, info[1])  # daqui em diante, o __del__ dele libera
    extra = ArrowArray()
    if stream.get_next(ctypes.byref(stream), ctypes.byref(extra)) != 0:
        return None
    if extra.release:  # mais de um pedaço: o chamador junta os pedaços (rechunk) e tenta de novo
        extra.release(ctypes.byref(extra))
        return None
    return out


class ImportedBatch:
    """Um lote do Arrow (um DataFrame exportado de uma vez: um struct com uma coluna por filho), dono dos buffers."""

    __slots__ = ("array", "schema", "stream")

    def __init__(self, schema: ArrowSchema, array: ArrowArray, stream: ArrowArrayStream) -> None:
        self.schema = schema
        self.array = array
        self.stream = stream

    @property
    def schema_address(self) -> int:
        return ctypes.addressof(self.schema)

    @property
    def array_address(self) -> int:
        return ctypes.addressof(self.array)

    def __del__(self) -> None:
        if self.array.release:
            self.array.release(ctypes.byref(self.array))
        if self.schema.release:
            self.schema.release(ctypes.byref(self.schema))
        if self.stream.release:
            self.stream.release(ctypes.byref(self.stream))


def import_batch(df: Any) -> ImportedBatch | None:
    """O DataFrame como um lote só do Arrow, pelo ``__arrow_c_stream__`` (sem copiar), ou None (Polars sem o
    protocolo, algum tipo que ele não exporta, ou o DataFrame em mais de um lote)."""
    stream = _take_stream(df)
    if stream is None:
        return None
    schema = ArrowSchema()
    if stream.get_schema(ctypes.byref(stream), ctypes.byref(schema)) != 0:
        stream.release(ctypes.byref(stream))
        return None
    first = ArrowArray()
    if stream.get_next(ctypes.byref(stream), ctypes.byref(first)) != 0 or not first.release:
        if schema.release:
            schema.release(ctypes.byref(schema))
        stream.release(ctypes.byref(stream))
        return None
    out = ImportedBatch(schema, first, stream)  # daqui em diante, o __del__ dele libera
    extra = ArrowArray()
    if stream.get_next(ctypes.byref(stream), ctypes.byref(extra)) != 0:
        return None
    if extra.release:  # mais de um lote
        extra.release(ctypes.byref(extra))
        return None
    return out
