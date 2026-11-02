// datacomp.h -- ABI em C do nini-datacomp, chamada pelo Python via ctypes (src/nini_datacomp/_native.py).
// GERADO por tools/gen_bindings.py a partir dos .cpp: não edite à mão. Os kernels leem os buffers do Arrow
// entregues pelo Polars (sem cópia) e escrevem em buffers alocados pelo chamador.
// Mudou a assinatura de alguma função? Aumente ABI_VERSION em tools/gen_bindings.py e regenere.
#pragma once
#include <stdint.h>

#define DC_ABI_VERSION 5

// DC_API: exportada ao compilar a biblioteca (DC_BUILDING), importada ao usá-la de outro código C/C++
#if defined(_WIN32)
#  if defined(DC_BUILDING)
#    define DC_API __declspec(dllexport)
#  else
#    define DC_API __declspec(dllimport)
#  endif
#else
#  define DC_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* abi.cpp */
DC_API int dc_abi_version();
DC_API int64_t dc_count_unset(const uint8_t* bits, int64_t offset, int64_t m);

/* batch.cpp */
DC_API int dc_describe_batch(const uint8_t* schema, const uint8_t* array, int k, int64_t* desc);

/* keys.cpp */
DC_API int dc_align_keys(int k, const int64_t* left, int64_t nl, const int64_t* right, int64_t nr, int strategy, uint32_t* li, uint32_t* ri, int64_t* out);

/* memory.cpp */
DC_API int64_t dc_pool_alloc(int64_t nbytes);
DC_API void dc_pool_free(int64_t addr);
DC_API int64_t dc_pool_limit(int64_t bytes);
DC_API void dc_pool_trim();
DC_API void dc_pool_stats(int64_t* out);

/* parquet_meta.cpp */
DC_API int64_t dc_parquet_row_groups(const uint8_t* footer, int64_t n, int64_t* rows, int64_t cap);

/* rowhash.cpp */
DC_API int dc_row_hash(int k, const int64_t* desc, int64_t start, int64_t end, int first, uint64_t* out);
DC_API int64_t dc_diff_pairs(int k, const int64_t* left, const int64_t* right, const uint64_t* hl, const uint64_t* hr, const uint32_t* li, const uint32_t* ri, int64_t start, int64_t end, int verify, uint32_t* diff, int64_t* stats);

#ifdef __cplusplus
}
#endif
