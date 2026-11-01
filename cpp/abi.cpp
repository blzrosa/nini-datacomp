// abi.cpp -- versão da ABI e utilidades pequenas sobre bitmaps do Arrow.
#include "common.hpp"

// Conferida pelo Python ao carregar a biblioteca (src/nini_datacomp/_native.py): um .dll/.so velho com um pacote
// Python novo (ou o contrário) falha cedo, com uma mensagem clara, em vez de corromper memória.
DC_EXPORT int dc_abi_version() { return DC_ABI_VERSION; }

// Bits 0 (nulos) entre as posições [offset, offset + m) de um bitmap de validade do Arrow. Lê 64 bits por vez no meio.
DC_EXPORT i64 dc_count_unset(const u8* bits, i64 offset, i64 m) {
  if (m <= 0) return 0;
  i64 i = offset, end = offset + m, zeros = 0;
  while (i < end && (i & 7)) zeros += !bit_at(bits, i++);   // até alinhar no byte
  while (i + 64 <= end) {
    u64 w;
    std::memcpy(&w, bits + (i >> 3), 8);
    // popcount portátil (sem depender de __builtin_popcountll nem de _mm_popcnt_u64)
    w = w - ((w >> 1) & 0x5555555555555555ULL);
    w = (w & 0x3333333333333333ULL) + ((w >> 2) & 0x3333333333333333ULL);
    w = (w + (w >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    zeros += 64 - (i64)((w * 0x0101010101010101ULL) >> 56);
    i += 64;
  }
  while (i < end) zeros += !bit_at(bits, i++);
  return zeros;
}
