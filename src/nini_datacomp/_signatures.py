"""Tipos dos argumentos de cada função da biblioteca C++ (GERADO por tools/gen_bindings.py; não edite)."""

# ruff: noqa
# fmt: off
from __future__ import annotations

from ctypes import c_double, c_float, c_int, c_int64, c_uint8, c_uint16, c_uint32, c_uint64, c_void_p

ABI_VERSION = 15

# nome: (tipo de retorno, [tipos dos argumentos])
SIGNATURES = {
    "dc_abi_version": (c_int, []),
    "dc_count_unset": (c_int64, [c_void_p, c_int64, c_int64]),
    "dc_describe_batch": (c_int, [c_void_p, c_void_p, c_int, c_void_p]),
    "dc_compare_column": (c_int64, [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int64, c_int64, c_int, c_double, c_double, c_double, c_void_p]),
    "dc_compare_rows": (c_int, [c_int, c_void_p, c_void_p, c_int64, c_int64, c_int64, c_void_p]),
    "dc_compare_pairs": (c_int, [c_int, c_int, c_void_p, c_void_p, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int64, c_int64, c_int64, c_void_p]),
    "dc_compare_column_all": (c_int64, [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int64, c_int, c_double, c_double, c_double, c_void_p, c_int]),
    "dc_compare_pairs_all": (c_int, [c_int, c_int, c_void_p, c_void_p, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int64, c_void_p, c_int]),
    "dc_expr_differs": (c_int64, [c_void_p, c_void_p, c_void_p, c_void_p, c_int64, c_void_p, c_int64, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int]),
    "dc_align_keys": (c_int, [c_int, c_void_p, c_int64, c_void_p, c_int64, c_int, c_void_p, c_void_p, c_void_p, c_int]),
    "dc_align_pieces": (c_int, [c_int, c_void_p, c_void_p, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]),
    "dc_pool_alloc": (c_int64, [c_int64]),
    "dc_pool_free": (None, [c_int64]),
    "dc_pool_limit": (c_int64, [c_int64]),
    "dc_pool_trim": (None, []),
    "dc_pool_stats": (None, [c_void_p]),
    "dc_parquet_row_groups": (c_int64, [c_void_p, c_int64, c_void_p, c_int64]),
    "dc_parquet_key_stats": (c_int64, [c_void_p, c_int64, c_void_p, c_int64, c_void_p, c_int64]),
    "dc_row_hash": (c_int, [c_int, c_void_p, c_int64, c_int64, c_int, c_void_p]),
    "dc_diff_pairs": (c_int64, [c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int64, c_int64, c_int, c_void_p, c_void_p]),
    "dc_row_hash_pieces": (c_int, [c_int, c_int, c_void_p, c_void_p, c_int, c_void_p, c_int]),
    "dc_diff_pairs_all": (c_int64, [c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_int64, c_int, c_void_p, c_void_p, c_int]),
    "dc_stat_open": (c_int64, [c_void_p, c_int64, c_int, c_void_p, c_int64, c_int64, c_int64, c_int]),
    "dc_stat_schema": (c_int, [c_int64]),
    "dc_stat_row_count": (c_int64, [c_int64]),
    "dc_stat_column": (c_int, [c_int64, c_int, c_void_p, c_int64, c_void_p, c_int64, c_void_p]),
    "dc_stat_next": (c_int64, [c_int64]),
    "dc_stat_buffers": (c_int, [c_int64, c_int, c_void_p]),
    "dc_stat_release": (None, [c_int64]),
    "dc_stat_error": (c_int64, [c_int64, c_void_p, c_int64]),
    "dc_stat_close": (None, [c_int64]),
    "dc_stat_encoding_supported": (c_int, [c_void_p, c_int64]),
}

__all__ = ['ABI_VERSION', 'SIGNATURES', 'c_double', 'c_float', 'c_int', 'c_int64', 'c_uint8', 'c_uint16', 'c_uint32', 'c_uint64', 'c_void_p']
