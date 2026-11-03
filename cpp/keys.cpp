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
// Descritores das colunas, leitura dos valores e igualdade: columns.hpp.
#include "columns.hpp"
#include "pool.hpp"

namespace {

// hash das linhas [start, end) de cada linha, coluna a coluna: o tipo é decidido uma vez por coluna, não a cada célula
void hash_rows(const Side& s, u64* h, i64 start, i64 end) {
  for (i64 i = start; i < end; i++) h[i] = 0;
  for (size_t c = 0; c < s.cols.size(); c++) {
    const Col& col = s.cols[c];
    const u64 salt = (u64)c;
    if (col.valid == nullptr && (col.kind == SIGNED || col.kind == UNSIGNED) && col.width == 8) {
      for (i64 i = start; i < end; i++) {
        u64 v;
        std::memcpy(&v, col.data + 8 * i, 8);
        h[i] = mix(h[i] * 0x100000001B3ULL ^ mix(v + 1) ^ salt);
      }
    } else if (col.valid == nullptr && col.kind == SIGNED && col.width == 4) {
      for (i64 i = start; i < end; i++) {
        i32 v;
        std::memcpy(&v, col.data + 4 * i, 4);
        h[i] = mix(h[i] * 0x100000001B3ULL ^ mix((u64)(i64)v + 1) ^ salt);
      }
    } else {
      for (i64 i = start; i < end; i++) h[i] = mix(h[i] * 0x100000001B3ULL ^ hash_cell(col, i) ^ salt);
    }
  }
}

void hash_rows_par(const Side& s, u64* h, int threads) {
  const i64 step = dc::task_rows(s.n, threads);
  dc::parallel_for(threads, (s.n + step - 1) / step, [&](i64 t) {
    const i64 start = t * step;
    hash_rows(s, h, start, start + step < s.n ? start + step : s.n);
  });
}

// ------------------------------------------------------------------ ordem
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

  // as linhas rows[0 .. m) (com o hash h[linha]); eq(linha guardada, linha nova) compara duas linhas do mesmo lado.
  // dups = linhas cuja chave já estava na tabela.
  template <typename Eq>
  void build(const u32* rows, i64 m, const u64* h, Eq eq, i64* dups) {
    u64 cap = 16;
    while (cap < (u64)m + (u64)m / 2) cap <<= 1;   // ocupação de 1/3 a 2/3
    slot.assign(cap, 0);
    mask = cap - 1;
    i64 d = 0;
    for (i64 q = 0; q < m; q++) {
      const i64 i = rows[q];
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

// As linhas de um lado separadas pelos `bits` de cima do hash (2^bits partes), em ordem dentro de cada parte:
// rows[off[p] .. off[p + 1]) são as da parte p. Duas passadas em paralelo (contar, espalhar), por trechos.
struct Partition {
  std::vector<i64> off;
  std::vector<u32> rows;
};

Partition partition(i64 n, const u64* h, int bits, int threads) {
  const i64 parts = (i64)1 << bits;
  const i64 step = dc::task_rows(n, threads);
  const i64 chunks = n ? (n + step - 1) / step : 0;
  auto part_of = [&](u64 x) { return bits ? (i64)(x >> (64 - bits)) : 0; };
  std::vector<i64> count((size_t)(chunks * parts), 0);
  dc::parallel_for(threads, chunks, [&](i64 c) {
    i64* cnt = count.data() + c * parts;
    const i64 end = (c + 1) * step < n ? (c + 1) * step : n;
    for (i64 i = c * step; i < end; i++) cnt[part_of(h[i])]++;
  });
  Partition out;
  out.off.assign((size_t)parts + 1, 0);
  std::vector<i64> pos((size_t)(chunks * parts));
  i64 total = 0;
  for (i64 p = 0; p < parts; p++) {   // parte a parte, trecho a trecho: a ordem das linhas fica a original
    out.off[(size_t)p] = total;
    for (i64 c = 0; c < chunks; c++) {
      pos[(size_t)(c * parts + p)] = total;
      total += count[(size_t)(c * parts + p)];
    }
  }
  out.off[(size_t)parts] = total;
  out.rows.resize((size_t)n);
  dc::parallel_for(threads, chunks, [&](i64 c) {
    i64* at = pos.data() + c * parts;
    const i64 end = (c + 1) * step < n ? (c + 1) * step : n;
    for (i64 i = c * step; i < end; i++) out.rows[(size_t)at[part_of(h[i])]++] = (u32)i;
  });
  return out;
}

// Hash join em partes: as linhas dos dois lados separadas pelos bits de cima do hash (a mesma chave cai sempre na
// mesma parte), e cada parte (uma tabela pequena, que cabe no cache) é juntada por uma thread. Cada parte conta as
// repetidas dos dois lados e grava, para cada linha da esquerda, a da direita com a mesma chave. No fim, os pares saem
// na ordem das linhas da esquerda: o mesmo resultado do hash join numa parte só.
template <typename EqRR, typename EqRL, typename EqLL>
void hash_join(i64 nl, i64 nr, const u64* hl, const u64* hr, EqRR err, EqRL erl, EqLL ell, u32* li, u32* ri, i64* out,
               int threads) {
  // ~16 partes por thread (equilíbrio), e nunca menores que ~4 mil linhas; numa thread só, uma parte (separar custa
  // mais do que o cache ganha: medido, 13 contra 17 milhões de chaves/s)
  int bits = 0;
  if (threads > 1)
    while (bits < 12 && (1 << bits) < 16 * threads && ((nl + nr) >> bits) > 8192) bits++;
  const Partition pl = partition(nl, hl, bits, threads), pr = partition(nr, hr, bits, threads);
  const i64 parts = (i64)1 << bits;
  const u32 none = 0xFFFFFFFFu;
  std::vector<u32> match((size_t)nl, none);
  std::vector<i64> dups((size_t)parts * 2, 0);
  dc::parallel_for(threads, parts, [&](i64 p) {
    const u32* lrows = pl.rows.data() + pl.off[(size_t)p];
    const u32* rrows = pr.rows.data() + pr.off[(size_t)p];
    const i64 ml = pl.off[(size_t)p + 1] - pl.off[(size_t)p], mr = pr.off[(size_t)p + 1] - pr.off[(size_t)p];
    Table t;
    t.build(rrows, mr, hr, err, &dups[(size_t)p * 2 + 1]);
    for (i64 q = 0; q < ml; q++) {
      const i64 j = lrows[q];
      const i64 r = t.find(hl[j], j, erl);
      if (r >= 0) match[(size_t)j] = (u32)r;
    }
    t.build(lrows, ml, hl, ell, &dups[(size_t)p * 2]);   // só para contar as repetidas da esquerda
  });
  i64 dl = 0, dr = 0;
  for (i64 p = 0; p < parts; p++) {
    dl += dups[(size_t)p * 2];
    dr += dups[(size_t)p * 2 + 1];
  }
  out[DUP_LEFT] = dl;
  out[DUP_RIGHT] = dr;
  if (dr == 0) {   // com repetidas à direita a comparação aborta: não precisa dos pares
    i64 common = 0;
    const i64 cap = nl < nr ? nl : nr;
    for (i64 j = 0; j < nl; j++) {
      if (match[(size_t)j] != none && common < cap) {
        li[common] = (u32)j;
        ri[common] = match[(size_t)j];
        common++;
      }
    }
    out[COMMON] = common;
    out[PAIRS] = common;
  }
}

}  // namespace

// Alinha as chaves de uma janela. `left`/`right`: k descritores de DESC inteiros (veja o topo do arquivo). `strategy`:
// 0 auto, 1 merge (exige os dois lados ordenados; se não estiverem, cai no hash), 2 hash. `li`/`ri`: espaço para
// min(nl, nr) pares (índices de linha, uint32). `out`: N_STATS contagens. `threads`: as do hash join (o merge é uma
// passada só). Devolve 0, ou -1 com mais de 2^32 - 1 linhas num lado (a janela tem de ser menor).
DC_EXPORT int dc_align_keys(int k, const i64* left, i64 nl, const i64* right, i64 nr, int strategy, u32* li, u32* ri,
                            i64* out, int threads) {
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
    hash_rows_par(L, hl.data(), threads);
    hash_rows_par(R, hr.data(), threads);
    if (simple == 8) {
      const i64 *a = (const i64*)L.cols[0].data, *b = (const i64*)R.cols[0].data;
      hash_join(nl, nr, hl.data(), hr.data(), IntEq<i64>{b, b}, IntEq<i64>{b, a}, IntEq<i64>{a, a}, li, ri, out,
                threads);
    } else if (simple == 4) {
      const i32 *a = (const i32*)L.cols[0].data, *b = (const i32*)R.cols[0].data;
      hash_join(nl, nr, hl.data(), hr.data(), IntEq<i32>{b, b}, IntEq<i32>{b, a}, IntEq<i32>{a, a}, li, ri, out,
                threads);
    } else {
      hash_join(nl, nr, hl.data(), hr.data(), GenericEq{R, R}, GenericEq{R, L}, GenericEq{L, L}, li, ri, out,
                threads);
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

namespace {

// Uma coluna inteira (sem nulos) em pedaços de memória, lida em sequência: o valor da linha atual e o avanço, sem
// juntar os pedaços antes (o rechunk copiava a chave inteira de cada janela).
template <typename T>
struct Cursor {
  const i64* desc;
  const i64* starts;
  int pieces;
  int piece = 0;
  i64 row = 0;   // linha global
  const T* data = nullptr;   // o primeiro valor do pedaço atual
  i64 first = 0, end = 0;     // primeira linha e fim (globais) do pedaço atual

  Cursor(const i64* d, const i64* s, int p) : desc(d), starts(s), pieces(p) { seek(); }

  void seek() {   // pula pedaços vazios
    while (piece < pieces && starts[piece + 1] <= row) piece++;
    if (piece < pieces) {
      data = (const T*)(uintptr_t)desc[(i64)piece * DESC + 2];
      first = starts[piece];
      end = starts[piece + 1];
    }
  }
  bool done() const { return piece >= pieces; }
  T value() const {
    T v;
    std::memcpy(&v, data + (row - first), sizeof(T));
    return v;
  }
  void next() {
    if (++row >= end) { piece++; seek(); }
  }
};

template <typename T>
bool sorted_pieces(const i64* desc, const i64* starts, int p, i64* dups) {
  Cursor<T> c(desc, starts, p);
  i64 d = 0;
  if (c.done()) { *dups = 0; return true; }
  T prev = c.value();
  for (c.next(); !c.done(); c.next()) {
    const T x = c.value();
    if (prev > x) return false;
    d += prev == x;
    prev = x;
  }
  *dups = d;
  return true;
}

template <typename T>
i64 merge_pieces(const i64* ld, const i64* ls, int pl, const i64* rd, const i64* rs, int pr, u32* li, u32* ri) {
  Cursor<T> a(ld, ls, pl), b(rd, rs, pr);
  i64 common = 0;
  while (!a.done() && !b.done()) {
    const T x = a.value(), y = b.value();
    if (x < y) {
      a.next();
    } else if (y < x) {
      b.next();
    } else {
      li[common] = (u32)a.row;
      ri[common] = (u32)b.row;
      common++;
      a.next();
      b.next();
    }
  }
  return common;
}

// a chave é uma coluna inteira simples (int64 ou int32, sem nulos) em todos os pedaços dos dois lados? 8, 4 ou 0
int simple_pieces(const i64* desc, int p, int* width) {
  for (int i = 0; i < p; i++) {
    const i64* d = desc + (i64)i * DESC;
    if (d[0] != SIGNED || d[3] != 0 || (d[1] != 8 && d[1] != 4)) return 0;
    if (i == 0) *width = (int)d[1];
    else if (*width != d[1]) return 0;
  }
  return p > 0;
}

}  // namespace

// O merge das chaves direto dos pedaços de memória da janela, sem juntá-los: a chave é uma coluna só, int64 ou int32,
// sem nulos, nos dois lados (pl/pr pedaços, descritores em ldesc/rdesc, a primeira linha de cada um mais o total em
// lstarts/rstarts). Mesma saída de dc_align_keys. Devolve 0; 1 se não serve (outra chave, ou algum lado fora de
// ordem): quem chama junta os pedaços e usa dc_align_keys; -1 com mais de 2^32 - 1 linhas num lado.
DC_EXPORT int dc_align_pieces(int pl, const i64* lstarts, const i64* ldesc, int pr, const i64* rstarts,
                              const i64* rdesc, u32* li, u32* ri, i64* out) {
  for (int s = 0; s < N_STATS; s++) out[s] = 0;
  const i64 nl = lstarts[pl], nr = rstarts[pr];
  if (nl >= (i64)0xFFFFFFFF || nr >= (i64)0xFFFFFFFF) return -1;
  int wl = 0, wr = 0;
  if (!simple_pieces(ldesc, pl, &wl) || !simple_pieces(rdesc, pr, &wr) || wl != wr) return 1;
  i64 dl = 0, dr = 0;
  const bool sorted = wl == 8 ? sorted_pieces<i64>(ldesc, lstarts, pl, &dl) && sorted_pieces<i64>(rdesc, rstarts, pr, &dr)
                              : sorted_pieces<i32>(ldesc, lstarts, pl, &dl) && sorted_pieces<i32>(rdesc, rstarts, pr, &dr);
  if (!sorted) return 1;
  out[STRATEGY] = 1;
  out[DUP_LEFT] = dl;
  out[DUP_RIGHT] = dr;
  if (dl || dr) return 0;   // aborta: sem pares (em comum = 0, como em dc_align_keys)
  out[COMMON] = wl == 8 ? merge_pieces<i64>(ldesc, lstarts, pl, rdesc, rstarts, pr, li, ri)
                        : merge_pieces<i32>(ldesc, lstarts, pl, rdesc, rstarts, pr, li, ri);
  out[PAIRS] = out[COMMON];
  out[ONLY_LEFT] = nl - out[COMMON];
  out[ONLY_RIGHT] = nr - out[COMMON];
  return 0;
}
