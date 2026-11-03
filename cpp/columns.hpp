// columns.hpp -- leitura das colunas do Arrow pelos kernels (chaves e hash de linha): descritores, valores, hash e
// igualdade de uma célula. Tudo `inline` num namespace anônimo: cada .cpp tem a sua cópia, sem símbolo exportado.
//
// Descritor de cada coluna (DESC inteiros de 64 bits, na ordem): tipo, largura em bytes, endereço dos valores (já no
// primeiro elemento; nos booleanos, o byte 0 dos bits), endereço da validade (0 = sem nulos), deslocamento em bits
// (validade e booleanos), endereço de um vetor com o endereço de cada buffer de texto (só texto).
//
// Igualdade (a do `eq_missing` do Polars, que o datacompolars usa): nulo = nulo; nulo de um lado só = diferente;
// reais: -0.0 == 0.0 e qualquer NaN == qualquer NaN; texto: byte a byte.
#pragma once
#include "common.hpp"
#include <cmath>
#include <vector>

namespace {

enum Kind : i64 { SIGNED = 1, UNSIGNED = 2, TEXT = 3, BOOL = 4, BYTES = 5, FLOAT = 6 };
const int DESC = 6;
const u64 NULL_HASH = 0x9E3779B97F4A7C15ULL;

struct Col {
  i64 kind, width;
  const u8* data;
  const u8* valid;
  i64 bit_off;
  const u8* const* texts;

  bool is_null(i64 i) const { return valid != nullptr && !bit_at(valid, bit_off + i); }
};

struct Side {
  std::vector<Col> cols;
  i64 n;
};

inline Side read_side(const i64* desc, int k, i64 n) {
  Side s;
  s.n = n;
  for (int c = 0; c < k; c++) {
    const i64* d = desc + c * DESC;
    s.cols.push_back(Col{d[0], d[1], (const u8*)(uintptr_t)d[2], (const u8*)(uintptr_t)d[3], d[4],
                         (const u8* const*)(uintptr_t)d[5]});
  }
  return s;
}

// ------------------------------------------------------------------ valores
inline i64 load_signed(const u8* p, i64 w) {
  switch (w) {
    case 1: { int8_t v; std::memcpy(&v, p, 1); return v; }
    case 2: { int16_t v; std::memcpy(&v, p, 2); return v; }
    case 4: { i32 v; std::memcpy(&v, p, 4); return v; }
    default: { i64 v; std::memcpy(&v, p, 8); return v; }
  }
}

inline u64 load_unsigned(const u8* p, i64 w) {
  u64 v = 0;
  std::memcpy(&v, p, (size_t)w);   // little-endian (conferido em common.hpp)
  return v;
}

inline double load_float(const u8* p, i64 w) {
  if (w == 4) { float f; std::memcpy(&f, p, 4); return f; }
  double d; std::memcpy(&d, p, 8); return d;
}

// texto (view de 16 bytes do Arrow): {len, 12 bytes} se len <= 12; senão {len, prefixo, buffer, deslocamento}
struct Str { const u8* p; u32 n; };

inline Str load_text(const Col& c, i64 i) {
  const u8* v = c.data + 16 * i;
  u32 n;
  std::memcpy(&n, v, 4);
  if (n <= 12) return Str{v + 4, n};
  u32 buf, off;
  std::memcpy(&buf, v + 8, 4);
  std::memcpy(&off, v + 12, 4);
  return Str{c.texts[buf] + off, n};
}

inline bool load_bool(const Col& c, i64 i) { return bit_at(c.data, c.bit_off + i); }

// ------------------------------------------------------------------ hash
inline u64 mix(u64 x) {   // finalizador do murmur3 (fmix64): avalanche completa
  x ^= x >> 33;
  x *= 0xFF51AFD7ED558CCDULL;
  x ^= x >> 33;
  x *= 0xC4CEB9FE1A85EC53ULL;
  x ^= x >> 33;
  return x;
}

// os r (1 a 7) bytes em p como um inteiro little-endian (os bytes de cima zerados): o mesmo valor de um memcpy de r
// bytes num u64 zerado, mas com leituras de tamanho fixo (sem chamar o memcpy da libc, que um tamanho variável
// força) e sem ler além de p[r - 1]. As leituras que se sobrepõem gravam os mesmos bytes nas mesmas posições.
inline u64 load_tail(const u8* p, u64 r) {
  if (r >= 4) {
    u32 a, b;
    std::memcpy(&a, p, 4);
    std::memcpy(&b, p + r - 4, 4);
    return (u64)a | ((u64)b << (8 * (r - 4)));
  }
  return (u64)p[0] | ((u64)p[r >> 1] << (8 * (r >> 1))) | ((u64)p[r - 1] << (8 * (r - 1)));
}

inline u64 hash_bytes(const u8* p, u64 n) {
  u64 h = 0x6A09E667F3BCC909ULL ^ (n * 0x9E3779B97F4A7C15ULL);
  u64 i = 0;
  for (; i + 8 <= n; i += 8) {
    u64 w;
    std::memcpy(&w, p + i, 8);
    h = mix(h ^ w) + 0x632BE59BD9B4E019ULL;
  }
  if (i < n) h = mix(h ^ load_tail(p + i, n - i) ^ 0xA0761D6478BD642FULL);
  return mix(h);
}

inline u64 hash_cell(const Col& c, i64 i) {
  if (c.is_null(i)) return NULL_HASH;
  switch (c.kind) {
    case SIGNED: return mix((u64)load_signed(c.data + c.width * i, c.width) + 1);
    case UNSIGNED: return mix(load_unsigned(c.data + c.width * i, c.width) + 1);
    case BOOL: return mix(load_bool(c, i) ? 3 : 5);
    case FLOAT: {
      double d = load_float(c.data + c.width * i, c.width);
      if (std::isnan(d)) return mix(0x7FF8000000000000ULL);   // todo NaN é igual
      if (d == 0.0) d = 0.0;                                   // -0.0 == 0.0
      u64 bits;
      std::memcpy(&bits, &d, 8);
      return mix(bits);
    }
    case TEXT: { Str s = load_text(c, i); return hash_bytes(s.p, s.n); }
    default: return hash_bytes(c.data + c.width * i, (u64)c.width);
  }
}

// ------------------------------------------------------------------ igualdade
inline bool cell_equal(const Col& a, i64 i, const Col& b, i64 j) {
  bool na = a.is_null(i), nb = b.is_null(j);
  if (na || nb) return na && nb;
  switch (a.kind) {
    case SIGNED: return load_signed(a.data + a.width * i, a.width) == load_signed(b.data + b.width * j, b.width);
    case UNSIGNED: return load_unsigned(a.data + a.width * i, a.width) == load_unsigned(b.data + b.width * j, b.width);
    case BOOL: return load_bool(a, i) == load_bool(b, j);
    case FLOAT: {
      double x = load_float(a.data + a.width * i, a.width), y = load_float(b.data + b.width * j, b.width);
      return x == y || (std::isnan(x) && std::isnan(y));
    }
    case TEXT: {
      Str x = load_text(a, i), y = load_text(b, j);
      return x.n == y.n && std::memcmp(x.p, y.p, x.n) == 0;
    }
    default: return std::memcmp(a.data + a.width * i, b.data + b.width * j, (size_t)a.width) == 0;
  }
}

inline bool row_equal(const Side& a, i64 i, const Side& b, i64 j) {
  for (size_t c = 0; c < a.cols.size(); c++)
    if (!cell_equal(a.cols[c], i, b.cols[c], j)) return false;
  return true;
}

}  // namespace
