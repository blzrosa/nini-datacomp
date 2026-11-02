// rowhash.cpp -- hash de linha (64 bits) e linhas divergentes: o caminho exato da comparação.
//
// Cada lado da janela ganha um hash por linha, das colunas comparadas, na mesma ordem nos dois lados; um par de linhas
// com a mesma chave (de dc_align_keys) diverge quando os hashes diferem. O hash é próprio (não o do Polars): igual em
// toda plataforma e em toda versão do Polars, e normaliza no próprio kernel o que o `eq_missing` trata como igual
// (-0.0 e 0.0; todo NaN; nulo com ou sem bitmap de validade; texto inline ou num buffer).
//
// Construção: h = semente; para cada coluna c, h = fmix64((h ^ v) + sal[c] + (nulo ? NULO : 0)), com v o valor da
// célula como 64 bits (inteiros estendidos, reais normalizados, texto e bytes largos pelo próprio hash). O fmix64 é
// uma bijeção: duas linhas que diferem numa coluna só nunca colidem; colidir exige diferenças em duas ou mais colunas
// que se cancelem, com chance ~2^-64 por par. `dc_diff_pairs` com `verify` reconfere célula a célula os pares de hash
// igual, para quem precisa de certeza.
//
// Por blocos de linhas (o Python divide e paraleliza): cada chamada passa por todas as colunas das linhas
// [start, end), com os hashes do bloco no cache.
#include "columns.hpp"

namespace {

const u64 SEED = 0x243F6A8885A308D3ULL;    // dígitos de pi
const u64 NULL_SALT = 0x13198A2E03707344ULL;

inline u64 column_salt(int c) { return mix(0xA4093822299F31D0ULL + (u64)c * 0x9E3779B97F4A7C15ULL); }

inline u64 step(u64 h, u64 v, u64 salt) { return mix((h ^ v) + salt); }

inline u64 float_bits(double d) {
  if (std::isnan(d)) return 0x7FF8000000000000ULL;   // todo NaN é igual
  if (d == 0.0) d = 0.0;                             // -0.0 == 0.0
  u64 bits;
  std::memcpy(&bits, &d, 8);
  return bits;
}

// o valor de uma célula não nula como 64 bits (mesmo valor nos dois lados => mesmos 64 bits)
inline u64 cell_value(const Col& c, i64 i) {
  switch (c.kind) {
    case SIGNED: return (u64)load_signed(c.data + c.width * i, c.width);
    case UNSIGNED: return load_unsigned(c.data + c.width * i, c.width);
    case BOOL: return load_bool(c, i) ? 1 : 0;
    case FLOAT: return float_bits(load_float(c.data + c.width * i, c.width));
    case TEXT: { Str s = load_text(c, i); return hash_bytes(s.p, s.n); }
    default:
      if (c.width <= 8) return load_unsigned(c.data + c.width * i, c.width);
      return hash_bytes(c.data + c.width * i, (u64)c.width);
  }
}

// laço direto para inteiros e reais de largura fixa (o tipo decidido uma vez por coluna)
template <typename T, bool IsFloat>
void hash_fixed(const Col& c, i64 start, i64 end, u64 salt, u64* h) {
  const T* v = (const T*)c.data;
  if (c.valid == nullptr) {
    for (i64 i = start; i < end; i++) {
      T x;
      std::memcpy(&x, v + i, sizeof(T));
      const u64 bits = IsFloat ? float_bits((double)x) : (u64)(i64)x;
      h[i - start] = step(h[i - start], bits, salt);
    }
    return;
  }
  for (i64 i = start; i < end; i++) {
    if (!bit_at(c.valid, c.bit_off + i)) {
      h[i - start] = step(h[i - start], 0, salt + NULL_SALT);
      continue;
    }
    T x;
    std::memcpy(&x, v + i, sizeof(T));
    const u64 bits = IsFloat ? float_bits((double)x) : (u64)(i64)x;
    h[i - start] = step(h[i - start], bits, salt);
  }
}

template <typename T>
void hash_unsigned(const Col& c, i64 start, i64 end, u64 salt, u64* h) {
  const T* v = (const T*)c.data;
  for (i64 i = start; i < end; i++) {
    if (c.is_null(i)) {
      h[i - start] = step(h[i - start], 0, salt + NULL_SALT);
      continue;
    }
    T x;
    std::memcpy(&x, v + i, sizeof(T));
    h[i - start] = step(h[i - start], (u64)x, salt);
  }
}

void hash_generic(const Col& c, i64 start, i64 end, u64 salt, u64* h) {
  for (i64 i = start; i < end; i++) {
    if (c.is_null(i)) h[i - start] = step(h[i - start], 0, salt + NULL_SALT);
    else h[i - start] = step(h[i - start], cell_value(c, i), salt);
  }
}

bool supported(const Col& c) {
  switch (c.kind) {
    case SIGNED: return c.width == 1 || c.width == 2 || c.width == 4 || c.width == 8;
    case UNSIGNED: return c.width == 1 || c.width == 2 || c.width == 4 || c.width == 8;
    case FLOAT: return c.width == 4 || c.width == 8;
    case TEXT: case BOOL: return true;
    case BYTES: return c.width > 0;
    default: return false;
  }
}

void hash_column(const Col& c, i64 start, i64 end, u64 salt, u64* h) {
  if (c.kind == SIGNED) {
    switch (c.width) {
      case 1: hash_fixed<int8_t, false>(c, start, end, salt, h); return;
      case 2: hash_fixed<int16_t, false>(c, start, end, salt, h); return;
      case 4: hash_fixed<i32, false>(c, start, end, salt, h); return;
      default: hash_fixed<i64, false>(c, start, end, salt, h); return;
    }
  }
  if (c.kind == UNSIGNED) {
    switch (c.width) {
      case 1: hash_unsigned<u8>(c, start, end, salt, h); return;
      case 2: hash_unsigned<u16>(c, start, end, salt, h); return;
      case 4: hash_unsigned<u32>(c, start, end, salt, h); return;
      default: hash_unsigned<u64>(c, start, end, salt, h); return;
    }
  }
  if (c.kind == FLOAT) {
    if (c.width == 4) hash_fixed<float, true>(c, start, end, salt, h);
    else hash_fixed<double, true>(c, start, end, salt, h);
    return;
  }
  hash_generic(c, start, end, salt, h);
}

}  // namespace

// Hash das linhas [start, end) de um lado: `k` colunas (descritores de DESC inteiros, columns.hpp), que são as colunas
// first .. first + k - 1 da comparação. Com first = 0, começa do zero; com first > 0, continua o hash que já está em
// out (o das colunas 0 .. first - 1): ler as colunas em grupos dá os mesmos bits que ler todas de uma vez. Grava
// out[0 .. end - start). Devolve 0, ou -1 com um tipo/largura que o kernel não conhece (out fica indefinido).
DC_EXPORT int dc_row_hash(int k, const i64* desc, i64 start, i64 end, int first, u64* out) {
  Side s = read_side(desc, k, end);
  for (const Col& c : s.cols)
    if (!supported(c)) return -1;
  const i64 n = end - start;
  if (first == 0)
    for (i64 i = 0; i < n; i++) out[i] = SEED;
  for (int c = 0; c < k; c++) hash_column(s.cols[(size_t)c], start, end, column_salt(first + c), out);
  return 0;
}

// Pares p em [start, end) (linha li[p] da esquerda, ri[p] da direita): grava em diff[0 ..] os p divergentes, em ordem.
// Divergente = hashes diferentes (hl[li[p]] != hr[ri[p]]). Com `verify` != 0, os pares de hash igual são
// reconferidos célula a célula nas `k` colunas (descritores `left`/`right`, na mesma ordem do hash); os que diferem
// (colisões do hash) também entram em diff. stats[0] = divergentes gravados, stats[1] = colisões. Devolve stats[0].
DC_EXPORT i64 dc_diff_pairs(int k, const i64* left, const i64* right, const u64* hl, const u64* hr, const u32* li,
                            const u32* ri, i64 start, i64 end, int verify, u32* diff, i64* stats) {
  i64 nd = 0, collisions = 0;
  if (!verify) {
    for (i64 p = start; p < end; p++) {
      diff[nd] = (u32)p;
      nd += hl[li[p]] != hr[ri[p]];   // sem desvio: grava sempre, avança só no divergente
    }
  } else {
    Side L = read_side(left, k, 0), R = read_side(right, k, 0);
    for (i64 p = start; p < end; p++) {
      bool differs = hl[li[p]] != hr[ri[p]];
      if (!differs && !row_equal(L, li[p], R, ri[p])) {
        differs = true;
        collisions++;
      }
      if (differs) diff[nd++] = (u32)p;
    }
  }
  stats[0] = nd;
  stats[1] = collisions;
  return nd;
}
