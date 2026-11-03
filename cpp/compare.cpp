// compare.cpp -- comparação de uma coluna, par a par: igualdade exata ou com tolerância (o caminho coluna a coluna).
//
// Entra: a coluna dos dois lados (descritores de columns.hpp) e os pares de linhas de dc_align_keys. Sai: um byte por
// par (1 = diverge) e quantos divergem. Usado para (a) as colunas com tolerância, em todos os pares, e (b) a contagem
// por coluna das colunas exatas, só nos pares que o hash de linha marcou como divergentes.
//
// Semântica (a do datacompolars 0.4.4, que é a do Polars):
//   - nulo dos dois lados = igual; nulo de um lado só = diferente (nas duas formas);
//   - exata: `eq_missing` do Polars, com NaN = NaN e -0.0 = 0.0 (cell_equal, em columns.hpp);
//   - tolerância (só colunas numéricas: o Python decide): igual se exata, ou se |l - r| <= abs_tol + rel_tol * |r|,
//     com a aritmética do Polars: inteiros no próprio tipo (a subtração e o abs dão a volta, como no Polars), depois
//     comparados como double; Float32 em float; Float64 em double; Decimal pelo inteiro de 128 bits, dividido por
//     10^escala. A comparação final usa a ordem do Polars para reais (NaN maior que tudo; veja polars_le).
#include "columns.hpp"
#include "pool.hpp"

namespace {

struct Tol {
  double abs_tol, rel_tol;
};

template <typename T, typename U>
bool int_within(const Col& a, i64 i, const Col& b, i64 j, Tol t) {
  T l, r;
  std::memcpy(&l, a.data + sizeof(T) * i, sizeof(T));
  std::memcpy(&r, b.data + sizeof(T) * j, sizeof(T));
  if (l == r) return true;
  const T d = (T)(U)((U)l - (U)r);   // subtração com volta, no tipo da coluna
  const T ad = d < 0 ? (T)(U)((U)0 - (U)d) : d;
  const T ar = r < 0 ? (T)(U)((U)0 - (U)r) : r;
  return (double)ad <= t.abs_tol + t.rel_tol * (double)ar;
}

template <typename T>
bool uint_within(const Col& a, i64 i, const Col& b, i64 j, Tol t) {
  T l, r;
  std::memcpy(&l, a.data + sizeof(T) * i, sizeof(T));
  std::memcpy(&r, b.data + sizeof(T) * j, sizeof(T));
  if (l == r) return true;
  const T d = (T)(l - r);   // sem sinal: dá a volta e o abs não muda nada (como no Polars)
  return (double)d <= t.abs_tol + t.rel_tol * (double)r;
}

// `a <= b` como o Polars compara reais: NaN é maior que tudo e igual a NaN (ordem total), não o IEEE (em que qualquer
// comparação com NaN é falsa). Importa na tolerância: com r infinito, rel_tol * |r| pode dar 0 * inf = NaN, e com r
// NaN a tolerância é NaN; para o Polars, x <= NaN é verdade (o datacompolars dá esses pares como iguais).
template <typename F>
inline bool polars_le(F x, F y) {
  if (std::isnan(y)) return true;
  if (std::isnan(x)) return false;
  return x <= y;
}

bool f64_within(const Col& a, i64 i, const Col& b, i64 j, Tol t) {
  double l, r;
  std::memcpy(&l, a.data + 8 * i, 8);
  std::memcpy(&r, b.data + 8 * j, 8);
  if (l == r || (std::isnan(l) && std::isnan(r))) return true;
  return polars_le(std::fabs(l - r), t.abs_tol + t.rel_tol * std::fabs(r));
}

bool f32_within(const Col& a, i64 i, const Col& b, i64 j, Tol t) {
  float l, r;
  std::memcpy(&l, a.data + 4 * i, 4);
  std::memcpy(&r, b.data + 4 * j, 4);
  if (l == r || (std::isnan(l) && std::isnan(r))) return true;
  const float tol = (float)t.abs_tol + (float)t.rel_tol * std::fabs(r);   // o Polars faz tudo em Float32
  return polars_le(std::fabs(l - r), tol);
}

// Decimal (inteiro de 128 bits, little-endian, em complemento de dois)
struct I128 {
  u64 lo;
  i64 hi;
};

inline I128 load128(const u8* p) {
  I128 v;
  std::memcpy(&v.lo, p, 8);
  std::memcpy(&v.hi, p + 8, 8);
  return v;
}

inline I128 sub128(I128 a, I128 b) {
  I128 out;
  out.lo = a.lo - b.lo;
  out.hi = (i64)((u64)a.hi - (u64)b.hi - (u64)(a.lo < b.lo));
  return out;
}

inline I128 abs128(I128 v) {
  if (v.hi >= 0) return v;
  I128 out;   // -v = ~v + 1
  out.lo = ~v.lo + 1;
  out.hi = (i64)(~(u64)v.hi + (u64)(out.lo == 0));
  return out;
}

inline double to_double(I128 v) { return (double)v.hi * 18446744073709551616.0 + (double)v.lo; }

bool dec_within(const Col& a, i64 i, const Col& b, i64 j, Tol t, double scale) {
  const I128 l = load128(a.data + 16 * i), r = load128(b.data + 16 * j);
  if (l.lo == r.lo && l.hi == r.hi) return true;
  const double d = to_double(abs128(sub128(l, r))) / scale;
  return d <= t.abs_tol + t.rel_tol * (to_double(abs128(r)) / scale);
}

// igual com tolerância? (os dois lados não nulos)
bool within(const Col& a, i64 i, const Col& b, i64 j, Tol t, double scale) {
  switch (a.kind) {
    case SIGNED:
      switch (a.width) {
        case 1: return int_within<int8_t, u8>(a, i, b, j, t);
        case 2: return int_within<int16_t, u16>(a, i, b, j, t);
        case 4: return int_within<i32, u32>(a, i, b, j, t);
        default: return int_within<i64, u64>(a, i, b, j, t);
      }
    case UNSIGNED:
      switch (a.width) {
        case 1: return uint_within<u8>(a, i, b, j, t);
        case 2: return uint_within<u16>(a, i, b, j, t);
        case 4: return uint_within<u32>(a, i, b, j, t);
        default: return uint_within<u64>(a, i, b, j, t);
      }
    case FLOAT: return a.width == 4 ? f32_within(a, i, b, j, t) : f64_within(a, i, b, j, t);
    case BYTES: return a.width == 16 ? dec_within(a, i, b, j, t, scale) : cell_equal(a, i, b, j);
    default: return cell_equal(a, i, b, j);
  }
}

}  // namespace

// Compara a coluna `left` x `right` (um descritor de DESC inteiros cada, mesmo tipo e largura) nos pares q em
// [start, end): o par é p = subset[q] (ou p = q, sem subset), as linhas são li[p] e ri[p]. Marca differs[p] = 1 nos
// que divergem (não apaga os outros: quem chama zera) e devolve quantos. `tolerance` != 0: igualdade com tolerância
// (abs_tol, rel_tol; `scale` = 10^escala do Decimal, 1 nos outros tipos); 0: exata.
DC_EXPORT i64 dc_compare_column(const i64* left, const i64* right, const u32* li, const u32* ri, const u32* subset,
                                i64 start, i64 end, int tolerance, double abs_tol, double rel_tol, double scale,
                                u8* differs) {
  Side L = read_side(left, 1, 0), R = read_side(right, 1, 0);
  const Col &a = L.cols[0], &b = R.cols[0];
  const Tol t{abs_tol, rel_tol};
  i64 count = 0;
  for (i64 q = start; q < end; q++) {
    const i64 p = subset ? (i64)subset[q] : q;
    const i64 i = li[p], j = ri[p];
    bool equal;
    const bool na = a.is_null(i), nb = b.is_null(j);
    if (na || nb) {
      equal = na && nb;
    } else {
      equal = tolerance ? within(a, i, b, j, t, scale) : cell_equal(a, i, b, j);
    }
    if (!equal) {
      differs[p] = 1;
      count++;
    }
  }
  return count;
}

// k colunas lado a lado, já reunidas nos pares (a linha i da esquerda com a linha i da direita), nas linhas
// [start, end): differs[c * n + i] = 1 onde a coluna c diverge (igualdade exata, a de cell_equal). É a contagem por
// coluna das linhas que o hash de linha marcou: uma chamada por janela no lugar de uma `dc_compare_column` (e duas
// exportações do Polars) por coluna. Devolve 0.
DC_EXPORT int dc_compare_rows(int k, const i64* left, const i64* right, i64 n, i64 start, i64 end, u8* differs) {
  Side L = read_side(left, k, n), R = read_side(right, k, n);
  for (int c = 0; c < k; c++) {
    const Col &a = L.cols[c], &b = R.cols[c];
    u8* d = differs + (i64)c * n;
    for (i64 i = start; i < end; i++) d[i] = cell_equal(a, i, b, i) ? 0 : 1;
  }
  return 0;
}

namespace {

// Um lado da janela em pedaços de memória (um por row group, como o Polars entrega): os descritores de cada pedaço
// (k colunas cada) e a primeira linha de cada um. `locate` acha o pedaço de uma linha (busca binária: são poucos).
struct Pieces {
  std::vector<Side> sides;
  std::vector<i64> starts;   // p + 1 posições: a primeira linha de cada pedaço e o total

  Pieces(int k, int p, const i64* first, const i64* desc) : starts(first, first + p + 1) {
    for (int i = 0; i < p; i++) sides.push_back(read_side(desc + (i64)i * k * DESC, k, 0));
  }

  const Side& locate(i64 row, i64* local) const {
    size_t lo = 0, hi = sides.size();
    while (hi - lo > 1) {
      size_t mid = (lo + hi) / 2;
      if (starts[mid] <= row) lo = mid; else hi = mid;
    }
    *local = row - starts[lo];
    return sides[lo];
  }
};

}  // namespace

// k colunas, igualdade exata, nos pares q em [start, end) de n: o par é p = subset[q] (ou q), as linhas li[p] e ri[p]
// da janela, que vem em pedaços (pl à esquerda, pr à direita: os descritores de cada pedaço em sequência, pedaço a
// pedaço, e a primeira linha de cada um mais o total em lstarts/rstarts). differs[c * n + q] = 1 onde a coluna c
// diverge. É a contagem por coluna dos pares que o hash de linha marcou, direto das colunas da janela (as mesmas que o
// hash leu), sem reunir as linhas antes no Polars. Devolve 0.
DC_EXPORT int dc_compare_pairs(int k, int pl, const i64* lstarts, const i64* ldesc, int pr, const i64* rstarts,
                               const i64* rdesc, const u32* li, const u32* ri, const u32* subset, i64 n, i64 start,
                               i64 end, u8* differs) {
  const Pieces L(k, pl, lstarts, ldesc), R(k, pr, rstarts, rdesc);
  for (i64 q = start; q < end; q++) {
    const i64 p = subset ? (i64)subset[q] : q;
    i64 a, b;
    const Side& x = L.locate(li[p], &a);
    const Side& y = R.locate(ri[p], &b);
    for (int c = 0; c < k; c++) differs[(i64)c * n + q] = cell_equal(x.cols[c], a, y.cols[c], b) ? 0 : 1;
  }
  return 0;
}

// dc_compare_column sobre os `n` pares (de `subset`, ou todos) numa chamada só, em `threads` threads. Devolve quantos
// divergem.
DC_EXPORT i64 dc_compare_column_all(const i64* left, const i64* right, const u32* li, const u32* ri, const u32* subset,
                                    i64 n, int tolerance, double abs_tol, double rel_tol, double scale, u8* differs,
                                    int threads) {
  const i64 step_rows = dc::task_rows(n, threads);
  const i64 tasks = (n + step_rows - 1) / step_rows;
  std::vector<i64> counts((size_t)tasks, 0);
  dc::parallel_for(threads, tasks, [&](i64 t) {
    const i64 start = t * step_rows, end = start + step_rows < n ? start + step_rows : n;
    counts[(size_t)t] = dc_compare_column(left, right, li, ri, subset, start, end, tolerance, abs_tol, rel_tol, scale,
                                          differs);
  });
  i64 total = 0;
  for (i64 c : counts) total += c;
  return total;
}

// dc_compare_pairs sobre os `n` pares numa chamada só, em `threads` threads.
DC_EXPORT int dc_compare_pairs_all(int k, int pl, const i64* lstarts, const i64* ldesc, int pr, const i64* rstarts,
                                   const i64* rdesc, const u32* li, const u32* ri, const u32* subset, i64 n, u8* differs,
                                   int threads) {
  const i64 step_rows = dc::task_rows(n, threads);
  const i64 tasks = (n + step_rows - 1) / step_rows;
  dc::parallel_for(threads, tasks, [&](i64 t) {
    const i64 start = t * step_rows, end = start + step_rows < n ? start + step_rows : n;
    dc_compare_pairs(k, pl, lstarts, ldesc, pr, rstarts, rdesc, li, ri, subset, n, start, end, differs);
  });
  return 0;
}
