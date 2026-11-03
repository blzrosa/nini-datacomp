// expr.cpp -- o interpretador do dc.expr: um comparador próprio por coluna, compilado no Python
// (src/nini_datacomp/expr.py) para um programa de pilha, rodado aqui por blocos de pares (como o numexpr: cada
// instrução passa por um bloco inteiro de linhas, e o custo do despacho se divide pelo bloco).
//
// Regra de nulos (a do datacompolars, a mesma dos comparadores em Python): nulo/nulo = igual; nulo de um lado só =
// diferente; o programa só decide os pares com os dois lados não nulos. Dentro dele não há nulos.
//
// Tipos (o compilador do Python escolhe a instrução certa; aqui só se confere a coluna): I = inteiro de 64 bits
// (aritmética com estouro circular), F = real de 64 bits (Float32 é lido como double), B = booleano, S = texto UTF-8.
// Reais comparados em ordem total, como no Polars: NaN == NaN, NaN maior que tudo, -0.0 == 0.0.
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "columns.hpp"
#include "pool.hpp"
#include "casemap.inc"

namespace {

enum Type : i64 { T_I = 0, T_F = 1, T_B = 2, T_S = 3 };

enum Op : i64 {
  LOAD_L = 1, LOAD_R = 2, CONST_I = 3, CONST_F = 4, CONST_B = 5, CONST_S = 6, TO_F = 7,
  ADD_I = 10, SUB_I = 11, MUL_I = 12, ADD_F = 13, SUB_F = 14, MUL_F = 15, DIV_F = 16, NEG_I = 17, NEG_F = 18,
  ABS_I = 19, ABS_F = 20,
  EQ_I = 30, NE_I, LT_I, LE_I, GT_I, GE_I,
  EQ_F = 40, NE_F, LT_F, LE_F, GT_F, GE_F,
  EQ_S = 50, NE_S, LT_S, LE_S, GT_S, GE_S,
  EQ_B = 60, NE_B = 61,
  AND = 70, OR = 71, NOT = 72,
  LOWER = 80, UPPER = 81, STRIP = 82, NOSPACE = 83, LEN = 84,
};

constexpr i64 BLOCK = 256;

struct Reg {
  Type type = T_B;
  std::vector<i64> i;
  std::vector<double> f;
  std::vector<u8> b;
  std::vector<std::string> s;
};

struct Program {
  const i64* code;
  i64 n;
  const double* fconst;
  const i64* iconst;
  const u8* sdata;
  const i64* soff;
};

// ------------------------------------------------------------------ UTF-8 e Unicode
// lê o código em s[i], avança i (UTF-8 já válido: o Polars garante)
inline u32 next_cp(const std::string& s, size_t& i) {
  const u8 c = (u8)s[i];
  if (c < 0x80) { i += 1; return c; }
  if (c < 0xE0) { u32 cp = ((u32)(c & 0x1F) << 6) | ((u8)s[i + 1] & 0x3F); i += 2; return cp; }
  if (c < 0xF0) {
    u32 cp = ((u32)(c & 0x0F) << 12) | ((u32)((u8)s[i + 1] & 0x3F) << 6) | ((u8)s[i + 2] & 0x3F);
    i += 3;
    return cp;
  }
  u32 cp = ((u32)(c & 0x07) << 18) | ((u32)((u8)s[i + 1] & 0x3F) << 12) | ((u32)((u8)s[i + 2] & 0x3F) << 6) |
           ((u8)s[i + 3] & 0x3F);
  i += 4;
  return cp;
}

inline void put_cp(std::string& out, u32 cp) {
  if (cp < 0x80) {
    out += (char)cp;
  } else if (cp < 0x800) {
    out += (char)(0xC0 | (cp >> 6));
    out += (char)(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    out += (char)(0xE0 | (cp >> 12));
    out += (char)(0x80 | ((cp >> 6) & 0x3F));
    out += (char)(0x80 | (cp & 0x3F));
  } else {
    out += (char)(0xF0 | (cp >> 18));
    out += (char)(0x80 | ((cp >> 12) & 0x3F));
    out += (char)(0x80 | ((cp >> 6) & 0x3F));
    out += (char)(0x80 | (cp & 0x3F));
  }
}

inline const uint32_t* lookup(const uint32_t* keys, const uint32_t (*vals)[3], size_t n, u32 cp) {
  size_t lo = 0, hi = n;
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    if (keys[mid] < cp) lo = mid + 1;
    else hi = mid;
  }
  return lo < n && keys[lo] == cp ? vals[lo] : nullptr;
}

// letra com caixa (tem minúscula ou maiúscula diferente): aproxima a propriedade Cased do sigma final
inline bool cased(u32 cp) {
  return lookup(dc_lower_keys, dc_lower_vals, dc_lower_count, cp) != nullptr ||
         lookup(dc_upper_keys, dc_upper_vals, dc_upper_count, cp) != nullptr;
}

// White_Space do Unicode (o char::is_whitespace do Rust e o \s do regex, que o Polars usa)
inline bool space(u32 cp) {
  return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
         (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
         cp == 0x3000;
}

std::string to_case(const std::string& s, bool lower) {
  std::string out;
  out.reserve(s.size());
  std::vector<u32> cps;
  for (size_t i = 0; i < s.size();) cps.push_back(next_cp(s, i));
  for (size_t k = 0; k < cps.size(); k++) {
    const u32 cp = cps[k];
    if (cp < 0x80) {  // ASCII: o caminho comum
      out += (char)(lower ? (cp >= 'A' && cp <= 'Z' ? cp + 32 : cp) : (cp >= 'a' && cp <= 'z' ? cp - 32 : cp));
      continue;
    }
    if (lower && cp == 0x3A3) {  // Σ: ς no fim de palavra (antes há letra com caixa, depois não), σ no resto
      bool before = k > 0 && cased(cps[k - 1]);
      bool after = k + 1 < cps.size() && cased(cps[k + 1]);
      put_cp(out, before && !after ? 0x3C2 : 0x3C3);
      continue;
    }
    const uint32_t* to = lower ? lookup(dc_lower_keys, dc_lower_vals, dc_lower_count, cp)
                               : lookup(dc_upper_keys, dc_upper_vals, dc_upper_count, cp);
    if (to == nullptr) {
      put_cp(out, cp);
      continue;
    }
    for (int j = 0; j < 3 && to[j]; j++) put_cp(out, to[j]);
  }
  return out;
}

std::string strip(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b) {
    size_t i = a;
    if (!space(next_cp(s, i))) break;
    a = i;
  }
  // do fim: volta até o começo de cada código (bytes 10xxxxxx são continuação)
  while (b > a) {
    size_t start = b - 1;
    while (start > a && ((u8)s[start] & 0xC0) == 0x80) start--;
    size_t i = start;
    if (!space(next_cp(s, i))) break;
    b = start;
  }
  return s.substr(a, b - a);
}

std::string no_space(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size();) {
    size_t start = i;
    u32 cp = next_cp(s, i);
    if (!space(cp)) out.append(s, start, i - start);
  }
  return out;
}

i64 len_chars(const std::string& s) {
  i64 n = 0;
  for (unsigned char c : s) n += (c & 0xC0) != 0x80;
  return n;
}

// ------------------------------------------------------------------ comparações
inline int cmp_f(double x, double y) {  // ordem total do Polars
  bool nx = std::isnan(x), ny = std::isnan(y);
  if (nx || ny) return nx == ny ? 0 : (nx ? 1 : -1);
  return x < y ? -1 : (x > y ? 1 : 0);
}

inline int cmp_s(const std::string& x, const std::string& y) {
  int c = x.compare(y);  // byte a byte (a ordem dos códigos, em UTF-8)
  return c < 0 ? -1 : (c > 0 ? 1 : 0);
}

inline bool by_op(int c, i64 op_offset) {
  switch (op_offset) {
    case 0: return c == 0;
    case 1: return c != 0;
    case 2: return c < 0;
    case 3: return c <= 0;
    case 4: return c > 0;
    default: return c >= 0;
  }
}

// ------------------------------------------------------------------ carga
// -1: a coluna não é do tipo que o programa espera
int load(Reg& r, const Col& c, Type t, const u32* rows, i64 m) {
  r.type = t;
  switch (t) {
    case T_I:
      if (c.kind != SIGNED && c.kind != UNSIGNED) return -1;
      r.i.resize((size_t)m);
      for (i64 k = 0; k < m; k++) {
        const u8* p = c.data + c.width * rows[k];
        r.i[(size_t)k] = c.kind == SIGNED ? load_signed(p, c.width) : (i64)load_unsigned(p, c.width);
      }
      return 0;
    case T_F:
      r.f.resize((size_t)m);
      if (c.kind == FLOAT) {
        for (i64 k = 0; k < m; k++) r.f[(size_t)k] = load_float(c.data + c.width * rows[k], c.width);
      } else if (c.kind == SIGNED || c.kind == UNSIGNED) {
        for (i64 k = 0; k < m; k++) {
          const u8* p = c.data + c.width * rows[k];
          r.f[(size_t)k] = c.kind == SIGNED ? (double)load_signed(p, c.width) : (double)load_unsigned(p, c.width);
        }
      } else {
        return -1;
      }
      return 0;
    case T_B:
      if (c.kind != BOOL) return -1;
      r.b.resize((size_t)m);
      for (i64 k = 0; k < m; k++) r.b[(size_t)k] = load_bool(c, rows[k]);
      return 0;
    default:
      if (c.kind != TEXT) return -1;
      r.s.resize((size_t)m);
      for (i64 k = 0; k < m; k++) {
        Str v = load_text(c, rows[k]);
        r.s[(size_t)k].assign((const char*)v.p, v.n);
      }
      return 0;
  }
}

// ------------------------------------------------------------------ execução de um bloco
// 0 ou -1 (coluna do tipo errado) / -2 (programa inválido); `out` = resultado booleano do bloco
int run_block(const Program& p, const Col& left, const Col& right, const u32* li, const u32* ri, i64 m,
              std::vector<Reg>& stack, std::vector<u8>& out) {
  size_t top = 0;
  auto push = [&]() -> Reg& {
    if (stack.size() <= top) stack.emplace_back();
    return stack[top++];
  };
  for (i64 pc = 0; pc < p.n; pc += 2) {
    const i64 op = p.code[pc], arg = p.code[pc + 1];
    if (op == LOAD_L || op == LOAD_R) {
      if (load(push(), op == LOAD_L ? left : right, (Type)arg, op == LOAD_L ? li : ri, m) != 0) return -1;
      continue;
    }
    if (op >= CONST_I && op <= CONST_S) {
      Reg& r = push();
      switch (op) {
        case CONST_I: r.type = T_I; r.i.assign((size_t)m, p.iconst[arg]); break;
        case CONST_F: r.type = T_F; r.f.assign((size_t)m, p.fconst[arg]); break;
        case CONST_B: r.type = T_B; r.b.assign((size_t)m, (u8)(arg != 0)); break;
        default: {
          r.type = T_S;
          std::string v((const char*)p.sdata + p.soff[arg], (size_t)(p.soff[arg + 1] - p.soff[arg]));
          r.s.assign((size_t)m, v);
        }
      }
      continue;
    }
    if (top == 0) return -2;
    Reg& a = stack[top - 1];
    // um operando
    switch (op) {
      case TO_F:
        if (a.type != T_I) return -2;
        a.f.resize((size_t)m);
        for (i64 k = 0; k < m; k++) a.f[(size_t)k] = (double)a.i[(size_t)k];
        a.type = T_F;
        continue;
      case NEG_I: for (i64 k = 0; k < m; k++) a.i[(size_t)k] = (i64)(0ULL - (u64)a.i[(size_t)k]); continue;
      case NEG_F: for (i64 k = 0; k < m; k++) a.f[(size_t)k] = -a.f[(size_t)k]; continue;
      case ABS_I:
        for (i64 k = 0; k < m; k++) {
          i64 v = a.i[(size_t)k];
          a.i[(size_t)k] = v < 0 ? (i64)(0ULL - (u64)v) : v;
        }
        continue;
      case ABS_F: for (i64 k = 0; k < m; k++) a.f[(size_t)k] = std::fabs(a.f[(size_t)k]); continue;
      case NOT: for (i64 k = 0; k < m; k++) a.b[(size_t)k] = !a.b[(size_t)k]; continue;
      case LOWER: case UPPER:
        for (i64 k = 0; k < m; k++) a.s[(size_t)k] = to_case(a.s[(size_t)k], op == LOWER);
        continue;
      case STRIP: for (i64 k = 0; k < m; k++) a.s[(size_t)k] = strip(a.s[(size_t)k]); continue;
      case NOSPACE: for (i64 k = 0; k < m; k++) a.s[(size_t)k] = no_space(a.s[(size_t)k]); continue;
      case LEN:
        a.i.resize((size_t)m);
        for (i64 k = 0; k < m; k++) a.i[(size_t)k] = len_chars(a.s[(size_t)k]);
        a.type = T_I;
        continue;
      default: break;
    }
    // dois operandos: x (abaixo) op y (topo); o resultado fica no lugar de x
    if (top < 2) return -2;
    Reg& x = stack[top - 2];
    Reg& y = stack[top - 1];
    top--;
    if (op >= ADD_I && op <= MUL_I) {
      for (i64 k = 0; k < m; k++) {
        u64 u = (u64)x.i[(size_t)k], v = (u64)y.i[(size_t)k];
        x.i[(size_t)k] = (i64)(op == ADD_I ? u + v : op == SUB_I ? u - v : u * v);
      }
    } else if (op >= ADD_F && op <= DIV_F) {
      for (i64 k = 0; k < m; k++) {
        double u = x.f[(size_t)k], v = y.f[(size_t)k];
        x.f[(size_t)k] = op == ADD_F ? u + v : op == SUB_F ? u - v : op == MUL_F ? u * v : u / v;
      }
    } else if (op >= EQ_I && op <= GE_I) {
      x.b.resize((size_t)m);
      for (i64 k = 0; k < m; k++) {
        i64 u = x.i[(size_t)k], v = y.i[(size_t)k];
        x.b[(size_t)k] = by_op(u < v ? -1 : (u > v ? 1 : 0), op - EQ_I);
      }
      x.type = T_B;
    } else if (op >= EQ_F && op <= GE_F) {
      x.b.resize((size_t)m);
      for (i64 k = 0; k < m; k++) x.b[(size_t)k] = by_op(cmp_f(x.f[(size_t)k], y.f[(size_t)k]), op - EQ_F);
      x.type = T_B;
    } else if (op >= EQ_S && op <= GE_S) {
      x.b.resize((size_t)m);
      for (i64 k = 0; k < m; k++) x.b[(size_t)k] = by_op(cmp_s(x.s[(size_t)k], y.s[(size_t)k]), op - EQ_S);
      x.type = T_B;
    } else if (op == EQ_B || op == NE_B) {
      for (i64 k = 0; k < m; k++) x.b[(size_t)k] = (u8)((x.b[(size_t)k] == y.b[(size_t)k]) == (op == EQ_B));
    } else if (op == AND || op == OR) {
      for (i64 k = 0; k < m; k++)
        x.b[(size_t)k] = (u8)(op == AND ? (x.b[(size_t)k] && y.b[(size_t)k]) : (x.b[(size_t)k] || y.b[(size_t)k]));
    } else {
      return -2;
    }
  }
  if (top != 1 || stack[0].type != T_B) return -2;
  out.assign(stack[0].b.begin(), stack[0].b.end());
  return 0;
}

}  // namespace

// Os `n` pares (li[p], ri[p]): differs[p] = 1 onde o programa diz que os valores não são iguais (ou um lado só é
// nulo). `left`/`right`: um descritor de coluna cada (columns.hpp). Devolve quantos pares divergem; -1 se uma coluna
// não é do tipo que o programa espera, -2 se o programa é inválido.
DC_EXPORT i64 dc_expr_differs(const i64* left, const i64* right, const u32* li, const u32* ri, i64 n, const i64* code,
                              i64 ncode, const double* fconst, const i64* iconst, const u8* sdata, const i64* soff,
                              u8* differs, int threads) {
  const Side ls = read_side(left, 1, 0), rs = read_side(right, 1, 0);
  const Col& lc = ls.cols[0];
  const Col& rc = rs.cols[0];
  const Program prog{code, ncode, fconst, iconst, sdata, soff};
  const i64 step_rows = dc::task_rows(n, threads);
  const i64 tasks = (n + step_rows - 1) / step_rows;
  std::vector<i64> counts((size_t)tasks, 0);
  std::vector<int> errors((size_t)tasks, 0);
  dc::parallel_for(threads, tasks, [&](i64 t) {
    const i64 start = t * step_rows, end = start + step_rows < n ? start + step_rows : n;
    std::vector<Reg> stack;
    std::vector<u8> result;
    std::vector<u32> bl, br;
    std::vector<i64> where;
    i64 count = 0;
    for (i64 s = start; s < end; s += BLOCK) {
      const i64 e = s + BLOCK < end ? s + BLOCK : end;
      // os pares com os dois lados não nulos vão para o programa; os outros se decidem pela regra de nulos
      bl.clear();
      br.clear();
      where.clear();
      for (i64 p = s; p < e; p++) {
        bool nl = lc.is_null(li[p]), nr = rc.is_null(ri[p]);
        if (nl || nr) {
          differs[p] = (u8)(nl != nr);
          count += nl != nr;
          continue;
        }
        bl.push_back(li[p]);
        br.push_back(ri[p]);
        where.push_back(p);
      }
      if (where.empty()) continue;
      int rc_ = run_block(prog, lc, rc, bl.data(), br.data(), (i64)where.size(), stack, result);
      if (rc_ != 0) {
        errors[(size_t)t] = rc_;
        return;
      }
      for (size_t k = 0; k < where.size(); k++) {
        u8 d = (u8)!result[k];
        differs[where[k]] = d;
        count += d;
      }
    }
    counts[(size_t)t] = count;
  });
  i64 total = 0;
  for (size_t t = 0; t < counts.size(); t++) {
    if (errors[t] != 0) return errors[t];
    total += counts[t];
  }
  return total;
}
