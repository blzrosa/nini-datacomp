"""Tipos dos argumentos de cada função da biblioteca C++ (GERADO por tools/gen_bindings.py; não edite)."""
from __future__ import annotations

from ctypes import c_double, c_float, c_int, c_int64, c_uint8, c_uint16, c_uint32, c_uint64, c_void_p

ABI_VERSION = 3

# nome: (tipo de retorno, [tipos dos argumentos])
SIGNATURES = {
    "dc_abi_version": (c_int, []),
    "dc_count_unset": (c_int64, [c_void_p, c_int64, c_int64]),
    "dc_align_keys": (c_int, [c_int, c_void_p, c_int64, c_void_p, c_int64, c_int, c_void_p, c_void_p, c_void_p]),
    "dc_pool_alloc": (c_int64, [c_int64]),
    "dc_pool_free": (None, [c_int64]),
    "dc_pool_limit": (c_int64, [c_int64]),
    "dc_pool_trim": (None, []),
    "dc_pool_stats": (None, [c_void_p]),
    "dc_parquet_row_groups": (c_int64, [c_void_p, c_int64, c_void_p, c_int64]),
}

__all__ = ['ABI_VERSION', 'SIGNATURES', 'c_double', 'c_float', 'c_int', 'c_int64', 'c_uint8', 'c_uint16', 'c_uint32', 'c_uint64', 'c_void_p']
