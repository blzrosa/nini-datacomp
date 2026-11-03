/* miniz.h -- no lugar do miniz.h completo: o nini-datacomp só usa o descompressor (tinfl), para os .zsav do SPSS
 * (third_party/shim/zlib_shim.c). Configuração conservadora: sem leituras desalinhadas, sem stdio. */
#pragma once
#define MINIZ_NO_STDIO
#define MINIZ_NO_TIME
#define MINIZ_LITTLE_ENDIAN 1 /* o nini-datacomp só roda em CPU little-endian (veja cpp/common.hpp) */
#define MINIZ_USE_UNALIGNED_LOADS_AND_STORES 0
#if defined(_WIN64) || defined(__x86_64__) || defined(__aarch64__) || defined(_M_ARM64)
#define MINIZ_HAS_64BIT_REGISTERS 1
#else
#define MINIZ_HAS_64BIT_REGISTERS 0
#endif
#include "miniz_common.h"
#include "miniz_tinfl.h"
