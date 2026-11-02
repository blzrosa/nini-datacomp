// keys.cpp -- alinhamento das chaves (PK) dos dois lados de uma janela: pares de linhas e contagens.
//
// Entrada: as colunas da chave de cada lado, como buffers do Arrow lidos do Polars sem cópia (src/nini_datacomp/core/
// keys.py monta os descritores). Saída: os pares (linha esquerda, linha direita) das chaves presentes nos dois lados
// e as contagens que o datacompolars dá: em comum, só esquerda, só direita e linhas excedentes por chave repetida.
//
// Semântica (a do datacompolars, que faz o join com nulls_equal=True): uma chave nula casa com uma chave nula; numa
// chave composta, cada coluna conta separado (nulo só casa com nulo na mesma coluna). Reais: -0.0 == 0.0 e qualquer
// NaN casa com qualquer NaN.
//
// Duas estratégias:
//   merge  os dois lados já vêm ordenados pela chave (o caso dos Parquet gravados por PK, e das janelas por faixa):
//          uma passada linear, sem memória extra. Duplicata = duas linhas vizinhas iguais.
//   hash   qualquer ordem: tabela de endereçamento aberto (sondagem linear) com o hash da chave, uma por lado (as
//          duplicatas de cada lado), e a esquerda procura na direita.
// A estratégia "auto" confere a ordem dos dois lados (uma passada) e escolhe. Com chave repetida em qualquer lado, os
// pares não são gravados (o datacompolars aborta a comparação): só as contagens saem.
//
// Descritor de cada coluna (DESC inteiros de 64 bits, na ordem): tipo, largura em bytes, endereço dos valores (já no
// primeiro elemento; nos booleanos, o byte 0 dos bits), endereço da validade (0 = sem nulos), deslocamento em bits
// (validade e booleanos), endereço de um vetor com o endereço de cada buffer de texto (só texto).
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

Side read_side(const i64* desc, int k, i64 n) {
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

inline u64 hash_bytes(const u8* p, u64 n) {
  u64 h = 0x6A09E667F3BCC909ULL ^ (n * 0x9E3779B97F4A7C15ULL);
  u64 i = 0;
  for (; i + 8 <= n; i += 8) {
    u64 w;
    std::memcpy(&w, p + i, 8);
    h = mix(h ^ w) + 0x632BE59BD9B4E019ULL;
  }
  if (i < n) {
    u64 w = 0;
    std::memcpy(&w, p + i, (size_t)(n - i));
    h = mix(h ^ w ^ 0xA0761D6478BD642FULL);
  }
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

// hash de cada linha, coluna a coluna: o tipo é decidido uma vez por coluna, não a cada célula
void hash_rows(const Side& s, u64* h) {
  for (i64 i = 0; i < s.n; i++) h[i] = 0;
  for (size_t c = 0; c < s.cols.size(); c++) {
    const Col& col = s.cols[c];
    const u64 salt = (u64)c;
    if (col.valid == nullptr && (col.kind == SIGNED || col.kind == UNSIGNED) && col.width == 8) {
      for (i64 i = 0; i < s.n; i++) {
        u64 v;
        std::memcpy(&v, col.data + 8 * i, 8);
        h[i] = mix(h[i] * 0x100000001B3ULL ^ mix(v + 1) ^ salt);
      }
    } else if (col.valid == nullptr && col.kind == SIGNED && col.width == 4) {
      for (i64 i = 0; i < s.n; i++) {
        i32 v;
        std::memcpy(&v, col.data + 4 * i, 4);
        h[i] = mix(h[i] * 0x100000001B3ULL ^ mix((u64)(i64)v + 1) ^ salt);
      }
    } else {
      for (i64 i = 0; i < s.n; i++) h[i] = mix(h[i] * 0x100000001B3ULL ^ hash_cell(col, i) ^ salt);
    }
  }
}

// ------------------------------------------------------------------ igualdade e ordem
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

// <0, 0, >0. Nulos primeiro (a ordem padrão do sort do Polars); texto em ordem de bytes (a do Polars).
inline int cell_cmp(const Col& a, i64 i, const Col& b, i64 j) {
  bool na = a.is_null(i), nb = b.is_null(j);
  if (na || nb) return (int)nb - (int)na;
  switch (a.kind) {
    case SIGNED: {
      i64 x = load_signed(a.data + a.width * i, a.width), y = load_signed(b.data + b.width * j, b.width);
      return (x > y) - (x < y);
    }
    case UNSIGNED: {
      u64 x = load_unsigned(a.data + a.width * i, a.width), y = load_unsigned(b.data + b.width * j, b.width);
      return (x > y) - (x < y);
    }
    case BOOL: return (int)load_bool(a, i) - (int)load_bool(b, j);
    case TEXT: {
      Str x = load_text(a, i), y = load_text(b, j);
      u32 m = x.n < y.n ? x.n : y.n;
      int r = m ? std::memcmp(x.p, y.p, m) : 0;
      if (r) return r < 0 ? -1 : 1;
      return (x.n > y.n) - (x.n < y.n);
    }
    default: return 0;   // sem ordem (reais, bytes): o merge não é usado
  }
}

inline int row_cmp(const Side& a, i64 i, const Side& b, i64 j) {
  for (size_t c = 0; c < a.cols.size(); c++) {
    int r = cell_cmp(a.cols[c], i, b.cols[c], j);
    if (r) return r;
  }
  return 0;
}

bool orderable(const Side& s) {
  for (const Col& c : s.cols)
    if (c.kind == FLOAT || c.kind == BYTES) return false;
  return true;
}

// ordenado (não decrescente) pela chave? Conta de passagem as linhas repetidas (vizinhas iguais).
bool sorted_side(const Side& s, i64* dups) {
  i64 d = 0;
  for (i64 i = 1; i < s.n; i++) {
    int r = row_cmp(s, i - 1, s, i);
    if (r > 0) return false;
    d += (r == 0);
  }
  *dups = d;
  return true;
}

// ------------------------------------------------------------------ estatísticas
// out: [0] em comum, [1] só esquerda, [2] só direita, [3] excedentes à esquerda, [4] excedentes à direita,
//      [5] estratégia usada (1 merge, 2 hash), [6] pares gravados
enum Stat { COMMON = 0, ONLY_LEFT, ONLY_RIGHT, DUP_LEFT, DUP_RIGHT, STRATEGY, PAIRS, N_STATS };

// uma chave inteira sem nulos (o caso comum de PK): o laço lê os valores direto, sem decidir o tipo por linha
template <typename T>
bool sorted_typed(const T* v, i64 n, i64* dups) {
  i64 d = 0;
  for (i64 i = 1; i < n; i++) {
    if (v[i - 1] > v[i]) return false;
    d += (v[i - 1] == v[i]);
  }
  *dups = d;
  return true;
}

template <typename T>
i64 merge_typed(const T* a, i64 na, const T* b, i64 nb, u32* li, u32* ri) {
  i64 i = 0, j = 0, common = 0;
  while (i < na && j < nb) {
    T x = a[i], y = b[j];
    if (x < y) {
      i++;
    } else if (y < x) {
      j++;
    } else {
      li[common] = (u32)i;
      ri[common] = (u32)j;
      common++;
      i++;
      j++;
    }
  }
  return common;
}

// 8 = int64, 4 = int32, 0 = não serve (vários campos, nulos, outro tipo)
int simple_int_key(const Side& L, const Side& R) {
  if (L.cols.size() != 1) return 0;
  const Col &a = L.cols[0], &b = R.cols[0];
  if (a.kind != SIGNED || a.valid != nullptr || b.valid != nullptr) return 0;
  if (a.width == 8 && b.width == 8) return 8;
  if (a.width == 4 && b.width == 4) return 4;
  return 0;
}

// merge com a chave inteira simples; false = algum lado fora de ordem
template <typename T>
bool try_merge_typed(const Side& L, const Side& R, u32* li, u32* ri, i64* out) {
  const T* a = (const T*)L.cols[0].data;
  const T* b = (const T*)R.cols[0].data;
  i64 dl = 0, dr = 0;
  if (!sorted_typed(a, L.n, &dl) || !sorted_typed(b, R.n, &dr)) return false;
  out[DUP_LEFT] = dl;
  out[DUP_RIGHT] = dr;
  if (dl == 0 && dr == 0) {
    out[COMMON] = merge_typed(a, L.n, b, R.n, li, ri);
    out[PAIRS] = out[COMMON];
  }
  return true;
}

void merge_join(const Side& L, const Side& R, u32* li, u32* ri, i64* out) {
  i64 i = 0, j = 0, common = 0;
  while (i < L.n && j < R.n) {
    int r = row_cmp(L, i, R, j);
    if (r < 0) {
      i++;
    } else if (r > 0) {
      j++;
    } else {
      li[common] = (u32)i;
      ri[common] = (u32)j;
      common++;
      i++;
      j++;
    }
  }
  out[COMMON] = common;
  out[PAIRS] = common;
}

// Endereçamento aberto com sondagem linear. Cada slot guarda os 32 bits altos do hash da linha (etiqueta) e a linha + 1
// nos 32 baixos (0 = vazio): só uma etiqueta igual leva à comparação das chaves, que é a parte com acesso fora de
// ordem aos dados.
// Igualdade de duas linhas: genérica (qualquer chave) ou direta (uma coluna inteira sem nulos).
struct GenericEq {
  const Side& a;
  const Side& b;
  bool operator()(i64 i, i64 j) const { return row_equal(a, i, b, j); }
};

template <typename T>
struct IntEq {
  const T* a;
  const T* b;
  bool operator()(i64 i, i64 j) const { return a[i] == b[j]; }
};

struct Table {
  std::vector<u64> slot;
  u64 mask = 0;

  static u64 tag(u64 h) { return h & 0xFFFFFFFF00000000ULL; }

  template <typename Eq>   // eq(linha guardada, linha nova) compara duas linhas do mesmo lado
  void build(i64 n, const u64* h, Eq eq, i64* dups) {
    u64 cap = 16;
    while (cap < (u64)n + (u64)n / 2) cap <<= 1;   // ocupação de 1/3 a 2/3
    slot.assign(cap, 0);
    mask = cap - 1;
    i64 d = 0;
    for (i64 i = 0; i < n; i++) {
      const u64 t = tag(h[i]);
      for (u64 p = h[i] & mask;; p = (p + 1) & mask) {
        const u64 e = slot[p];
        if (e == 0) { slot[p] = t | (u64)(i + 1); break; }
        if ((e & 0xFFFFFFFF00000000ULL) == t && eq((i64)(u32)e - 1, i)) { d++; break; }   // repetida
      }
    }
    *dups = d;
  }

  template <typename Eq>   // eq(linha guardada, linha procurada do outro lado)
  i64 find(u64 h, i64 j, Eq eq) const {
    const u64 t = tag(h);
    for (u64 p = h & mask;; p = (p + 1) & mask) {
      const u64 e = slot[p];
      if (e == 0) return -1;
      if ((e & 0xFFFFFFFF00000000ULL) == t && eq((i64)(u32)e - 1, j)) return (i64)(u32)e - 1;
    }
  }
};

// hash join: a esquerda procura na tabela da direita; as duas tabelas contam as repetidas de cada lado
template <typename EqRR, typename EqRL, typename EqLL>
void hash_join(i64 nl, i64 nr, const u64* hl, const u64* hr, EqRR err, EqRL erl, EqLL ell, u32* li, u32* ri,
               i64* out) {
  i64 dl = 0, dr = 0;
  Table tr;
  tr.build(nr, hr, err, &dr);
  out[DUP_RIGHT] = dr;
  if (dr == 0) {   // com repetidas à direita a comparação aborta: não precisa dos pares
    i64 common = 0;
    const i64 cap = nl < nr ? nl : nr;
    for (i64 j = 0; j < nl; j++) {
      i64 r = tr.find(hl[j], j, erl);
      if (r >= 0 && common < cap) {
        li[common] = (u32)j;
        ri[common] = (u32)r;
        common++;
      }
    }
    out[COMMON] = common;
    out[PAIRS] = common;
  }
  Table tl;   // só para contar as repetidas da esquerda
  tl.build(nl, hl, ell, &dl);
  out[DUP_LEFT] = dl;
}

}  // namespace

// Alinha as chaves de uma janela. `left`/`right`: k descritores de DESC inteiros (veja o topo do arquivo). `strategy`:
// 0 auto, 1 merge (exige os dois lados ordenados; se não estiverem, cai no hash), 2 hash. `li`/`ri`: espaço para
// min(nl, nr) pares (índices de linha, uint32). `out`: N_STATS contagens. Devolve 0, ou -1 com mais de 2^32 - 1
// linhas num lado (a janela tem de ser menor).
DC_EXPORT int dc_align_keys(int k, const i64* left, i64 nl, const i64* right, i64 nr, int strategy, u32* li, u32* ri,
                            i64* out) {
  for (int s = 0; s < N_STATS; s++) out[s] = 0;
  if (nl >= (i64)0xFFFFFFFF || nr >= (i64)0xFFFFFFFF) return -1;
  Side L = read_side(left, k, nl), R = read_side(right, k, nr);

  i64 dl = 0, dr = 0;
  const int simple = simple_int_key(L, R);
  bool done = false;
  if (strategy != 2 && simple) {   // chave inteira simples: merge direto, se os dois lados estão em ordem
    done = simple == 8 ? try_merge_typed<i64>(L, R, li, ri, out) : try_merge_typed<i32>(L, R, li, ri, out);
    if (done) out[STRATEGY] = 1;
  }
  if (!done && strategy != 2 && !simple && orderable(L) && sorted_side(L, &dl) && sorted_side(R, &dr)) {
    done = true;
    out[STRATEGY] = 1;
    out[DUP_LEFT] = dl;
    out[DUP_RIGHT] = dr;
    if (dl == 0 && dr == 0) merge_join(L, R, li, ri, out);
  }
  if (!done) {
    out[STRATEGY] = 2;
    std::vector<u64> hl((size_t)nl), hr((size_t)nr);
    hash_rows(L, hl.data());
    hash_rows(R, hr.data());
    if (simple == 8) {
      const i64 *a = (const i64*)L.cols[0].data, *b = (const i64*)R.cols[0].data;
      hash_join(nl, nr, hl.data(), hr.data(), IntEq<i64>{b, b}, IntEq<i64>{b, a}, IntEq<i64>{a, a}, li, ri, out);
    } else if (simple == 4) {
      const i32 *a = (const i32*)L.cols[0].data, *b = (const i32*)R.cols[0].data;
      hash_join(nl, nr, hl.data(), hr.data(), IntEq<i32>{b, b}, IntEq<i32>{b, a}, IntEq<i32>{a, a}, li, ri, out);
    } else {
      hash_join(nl, nr, hl.data(), hr.data(), GenericEq{R, R}, GenericEq{R, L}, GenericEq{L, L}, li, ri, out);
    }
  }
  if (out[DUP_LEFT] || out[DUP_RIGHT]) {   // aborta: sem pares; em comum = chaves distintas presentes nos dois lados
    out[PAIRS] = 0;
    out[COMMON] = 0;
    return 0;
  }
  out[ONLY_LEFT] = nl - out[COMMON];
  out[ONLY_RIGHT] = nr - out[COMMON];
  return 0;
}
