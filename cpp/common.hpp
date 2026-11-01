// common.hpp -- tipos e macros compartilhados pelos kernels do nini-datacomp.
// Toda função exportada é `DC_EXPORT` (vínculo C, atributo de exportação de datacomp.h) e escreve em buffers alocados
// pelo chamador. Assinaturas: só tipos fixos (u8..u64, i32, i64, double, ponteiros), que tools/gen_bindings.py traduz
// para o ctypes.
#pragma once
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "nini-datacomp assume CPU little-endian (x86-64, ARM64): os buffers do Arrow são lidos byte a byte"
#endif
#include <cstdint>
#include <cstring>

// DC_BUILDING: estamos compilando a biblioteca, então datacomp.h declara as funções como exportadas (dllexport no
// Windows). A declaração no header e a definição no .cpp precisam ter o mesmo atributo, senão o MSVC acusa C2375.
#define DC_BUILDING 1
#include "datacomp.h"

// Na definição: mesmo vínculo (C) e mesmo atributo de exportação da declaração em datacomp.h
#define DC_EXPORT extern "C" DC_API

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int32_t i32;
typedef int64_t i64;

// bit i de um bitmap do Arrow (LSB primeiro): 1 = valor presente
static inline bool bit_at(const u8* bits, i64 i) { return (bits[i >> 3] >> (i & 7)) & 1; }
