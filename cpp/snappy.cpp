// snappy.cpp -- descompressor de snappy escrito a partir da descrição oficial do formato
// (https://github.com/google/snappy/blob/main/format_description.txt, licença BSD; crédito em NOTICE).
//
// Bloco = tamanho descomprimido (varint) + elementos. Cada elemento começa por um byte de marca; os 2 bits baixos
// dizem o tipo:
//   00 literal: tamanho-1 nos 6 bits altos (0..59), ou 60..63 = o tamanho-1 vem em 1..4 bytes (little-endian) a seguir
//   01 cópia, offset de 11 bits: tamanho = 4 + bits 2..4 da marca (4..11); offset = 3 bits altos da marca e o byte seguinte
//   10 cópia, offset de 16 bits: tamanho = 1 + 6 bits altos (1..64); offset = 2 bytes seguintes
//   11 cópia, offset de 32 bits: tamanho = 1 + 6 bits altos (1..64); offset = 4 bytes seguintes
// A cópia volta `offset` bytes no que já foi escrito e pode se sobrepor ao próprio destino (offset < tamanho repete
// o padrão), por isso é byte a byte nesse caso. Tudo que vem do arquivo (tamanhos, offsets) é conferido antes de usar.
#include "snappy.hpp"

namespace dc {

bool snappy_uncompressed_length(const u8* src, i64 n, i64* out) {
  u64 v = 0;
  for (int i = 0, shift = 0; i < 5 && i < n; i++, shift += 7) {
    v |= (u64)(src[i] & 0x7F) << shift;
    if (!(src[i] & 0x80)) {
      if (v > 0xFFFFFFFFull) return false;
      *out = (i64)v;
      return true;
    }
  }
  return false;
}

bool snappy_decompress(const u8* src, i64 n, u8* dst, i64 dst_len) {
  i64 declared;
  if (!snappy_uncompressed_length(src, n, &declared) || declared != dst_len) return false;
  i64 i = 0;
  while ((src[i++] & 0x80) != 0) {}   // pula o varint do cabeçalho (já validado acima)
  i64 o = 0;
  while (i < n) {
    const u8 tag = src[i++];
    const int kind = tag & 3;
    if (kind == 0) {
      u64 len = tag >> 2;
      if (len >= 60) {
        const int extra = (int)len - 59;   // 1..4 bytes
        if (n - i < extra) return false;
        len = 0;
        for (int k = 0; k < extra; k++) len |= (u64)src[i + k] << (8 * k);
        i += extra;
      }
      len += 1;
      if (len > (u64)(n - i) || len > (u64)(dst_len - o)) return false;
      std::memcpy(dst + o, src + i, (size_t)len);
      i += (i64)len;
      o += (i64)len;
      continue;
    }
    u64 len, offset;
    if (kind == 1) {
      if (i >= n) return false;
      len = 4 + ((tag >> 2) & 7);
      offset = ((u64)(tag >> 5) << 8) | src[i];
      i += 1;
    } else if (kind == 2) {
      if (n - i < 2) return false;
      len = 1 + (tag >> 2);
      offset = (u64)src[i] | ((u64)src[i + 1] << 8);
      i += 2;
    } else {
      if (n - i < 4) return false;
      len = 1 + (tag >> 2);
      offset = (u64)src[i] | ((u64)src[i + 1] << 8) | ((u64)src[i + 2] << 16) | ((u64)src[i + 3] << 24);
      i += 4;
    }
    if (offset == 0 || offset > (u64)o || len > (u64)(dst_len - o)) return false;
    u8* d = dst + o;
    const u8* s = d - offset;
    if (offset >= len) {
      std::memcpy(d, s, (size_t)len);
    } else {
      for (u64 k = 0; k < len; k++) d[k] = s[k];   // sobreposta: repete o padrão
    }
    o += (i64)len;
  }
  return o == dst_len;
}

}  // namespace dc
