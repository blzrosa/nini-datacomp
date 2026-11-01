// datacomp.h -- ABI em C do nini-datacomp, chamada pelo Python via ctypes (src/nini_datacomp/_native.py).
// GERADO por tools/gen_bindings.py a partir dos .cpp: não edite à mão. Os kernels leem os buffers do Arrow
// entregues pelo Polars (sem cópia) e escrevem em buffers alocados pelo chamador.
// Mudou a assinatura de alguma função? Aumente ABI_VERSION em tools/gen_bindings.py e regenere.
#pragma once
#include <stdint.h>

#define DC_ABI_VERSION 1

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

/* memory.cpp */
DC_API int64_t dc_pool_alloc(int64_t nbytes);
DC_API void dc_pool_free(int64_t addr);
DC_API int64_t dc_pool_limit(int64_t bytes);
DC_API void dc_pool_trim();
DC_API void dc_pool_stats(int64_t* out);

#ifdef __cplusplus
}
#endif
