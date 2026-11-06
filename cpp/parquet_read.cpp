// parquet_read.cpp -- leitor de Parquet em C++ (ReadOptions.reader = "native"), entregue ao Polars pela interface C do
// Arrow, sem pyarrow e sem cópia: cada (row group, coluna) vira um ArrowArray cujos buffers vêm do pool de memória
// (memory.cpp) e voltam a ele no `release`.
//
// Tradução da lógica do leitor Parquet do Polars (polars-parquet, Rust, licença MIT; crédito em NOTICE) para este
// formato de saída; o formato do arquivo é o de https://github.com/apache/parquet-format. Não é cópia de código.
//
// Escopo: colunas planas (sem lista, struct ou mapa; max_rep = 0 e max_def <= 1) dos tipos INT32, INT64, FLOAT,
// DOUBLE, BOOLEAN, texto (BYTE_ARRAY UTF8, exportado como string_view "vu"), DATE e TIMESTAMP ms/us/ns sem fuso.
// Encodings PLAIN, PLAIN_DICTIONARY/RLE_DICTIONARY e RLE (booleano); níveis RLE/bit-packed; páginas v1 e v2; codecs
// UNCOMPRESSED, SNAPPY (snappy.cpp), ZSTD (third_party/zstd) e GZIP (tinfl do miniz). Qualquer outra coisa devolve -1
// (o Python usa o Polars). Arquivo corrompido devolve -2, com a mensagem em dc_parquet_error. Todo tamanho, offset e
// contagem lido do arquivo é conferido contra o buffer antes de ser usado, e o texto é validado como UTF-8 (o Polars
// confia nele).
//
// As tarefas são (row group, coluna), distribuídas pelo parallel_for de pool.hpp; cada uma lê só o byte range do seu
// column chunk.
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#include "common.hpp"
#include "memory.hpp"
#include "pool.hpp"
#include "snappy.hpp"
#include "thrift.hpp"

extern "C" {
#include "third_party/miniz/miniz_tinfl.h"
}
#include "zstd.h"

namespace {

using namespace thrift;

// Estruturas: https://arrow.apache.org/docs/format/CDataInterface.html (as mesmas de batch.cpp)
struct ArrowSchema {
  const char* format;
  const char* name;
  const char* metadata;
  i64 flags;
  i64 n_children;
  ArrowSchema** children;
  ArrowSchema* dictionary;
  void (*release)(ArrowSchema*);
  void* private_data;
};

struct ArrowArray {
  i64 length;
  i64 null_count;
  i64 offset;
  i64 n_buffers;
  i64 n_children;
  const void** buffers;
  ArrowArray** children;
  ArrowArray* dictionary;
  void (*release)(ArrowArray*);
  void* private_data;
};

thread_local std::string g_error;

// Falha dentro de uma tarefa: code -1 = fora do suporte (usar o Polars), -2 = erro de E/S ou arquivo corrompido
struct Fail {
  int code;
  std::string msg;
};
[[noreturn]] void unsupported(const std::string& why) { throw Fail{-1, why}; }
[[noreturn]] void corrupt(const std::string& why) { throw Fail{-2, why}; }

// Parquet: enums do parquet.thrift
enum Phys { P_BOOL = 0, P_I32 = 1, P_I64 = 2, P_I96 = 3, P_FLOAT = 4, P_DOUBLE = 5, P_BYTES = 6, P_FIXED = 7 };
enum Codec { C_NONE = 0, C_SNAPPY = 1, C_GZIP = 2, C_ZSTD = 6 };
enum Enc { E_PLAIN = 0, E_PLAIN_DICT = 2, E_RLE = 3, E_RLE_DICT = 8 };
enum PageType { PG_DATA = 0, PG_INDEX = 1, PG_DICT = 2, PG_DATA_V2 = 3 };
enum Logical { L_NONE = 0, L_STRING, L_DATE, L_TS, L_INT, L_OTHER };
enum Kind { K_I32, K_I64, K_F32, K_F64, K_BOOL, K_STRING, K_DATE, K_TS_MS, K_TS_US, K_TS_NS };

// ----------------------------------------------------------------------------- rodapé (FileMetaData)
struct Elem {
  std::string name;
  int type = -1;
  int rep = 0;
  int nchild = 0;
  int converted = -1;
  int logical = L_NONE;
  int ts_unit = 0;  // 1 ms, 2 us, 3 ns
  bool ts_utc = false;
  int int_bits = 0;
  bool int_signed = true;
};

struct Chunk {
  bool present = false;
  int type = -1;
  int codec = -1;
  i64 num_values = -1, comp = -1, uncomp = -1, data_off = -1, dict_off = -1;
  bool external = false;  // file_path ou criptografia: fora do suporte
};

struct RowGroup {
  i64 rows = -1;
  std::vector<Chunk> chunks;  // uma por coluna pedida
};

struct ColInfo {
  int phys = -1;
  int kind = 0;
  int max_def = 0;
  int width = 0;  // bytes por valor (0: booleano e texto)
  const char* fmt = "";
};

struct Footer {
  std::vector<Elem> schema;
  std::vector<ColInfo> cols;
  std::vector<int> slot_of_leaf;
  std::vector<RowGroup> rgs;
  bool have_schema = false;
};

i64 rint(Reader& r, int type) {
  if (type == T_I16 || type == T_I32 || type == T_I64) return r.zigzag();
  r.ok = false;
  return 0;
}

std::string rstr(Reader& r) {
  u64 n = r.varint();
  if (!r.ok || (u64)(r.end - r.p) < n) {
    r.ok = false;
    return {};
  }
  std::string s((const char*)r.p, (size_t)n);
  r.p += n;
  return s;
}

void parse_timestamp(Reader& r, Elem& e) {
  int last = 0, id, type;
  while (next_field(r, &last, &id, &type)) {
    if (id == 1) {
      e.ts_utc = (type == T_TRUE);
    } else if (id == 2 && type == T_STRUCT) {  // TimeUnit: união de MILLIS(1), MICROS(2), NANOS(3)
      int l2 = 0, i2, t2;
      while (next_field(r, &l2, &i2, &t2)) {
        if (i2 >= 1 && i2 <= 3) e.ts_unit = i2;
        r.skip(t2, 2);
      }
    } else {
      r.skip(type, 1);
    }
  }
}

void parse_integer(Reader& r, Elem& e) {
  int last = 0, id, type;
  while (next_field(r, &last, &id, &type)) {
    if (id == 1 && type == T_BYTE) e.int_bits = r.byte();
    else if (id == 2) e.int_signed = (type == T_TRUE);
    else r.skip(type, 1);
  }
}

void parse_logical(Reader& r, Elem& e) {  // LogicalType: união, um campo struct só
  int last = 0, id, type;
  while (next_field(r, &last, &id, &type)) {
    if (type != T_STRUCT) {
      r.skip(type, 1);
      e.logical = L_OTHER;
      continue;
    }
    switch (id) {
      case 1: e.logical = L_STRING; r.skip(type, 1); break;
      case 6: e.logical = L_DATE; r.skip(type, 1); break;
      case 8: e.logical = L_TS; parse_timestamp(r, e); break;
      case 10: e.logical = L_INT; parse_integer(r, e); break;
      default: e.logical = L_OTHER; r.skip(type, 1); break;
    }
  }
}

void parse_elem(Reader& r, Elem& e) {
  int last = 0, id, type;
  while (next_field(r, &last, &id, &type)) {
    switch (id) {
      case 1: e.type = (int)rint(r, type); break;
      case 3: e.rep = (int)rint(r, type); break;
      case 4:
        if (type == T_BINARY) e.name = rstr(r);
        else r.skip(type, 1);
        break;
      case 5: e.nchild = (int)rint(r, type); break;
      case 6: e.converted = (int)rint(r, type); break;
      case 10:
        if (type == T_STRUCT) parse_logical(r, e);
        else r.skip(type, 1);
        break;
      default: r.skip(type, 1); break;
    }
  }
}

bool walk_leaves(const std::vector<Elem>& s, size_t& idx, int depth, int& leaves) {
  if (idx >= s.size() || depth > 32) return false;
  const Elem& e = s[idx++];
  if (e.nchild < 0) return false;
  if (e.nchild == 0) {
    leaves++;
    return true;
  }
  for (int i = 0; i < e.nchild; i++)
    if (!walk_leaves(s, idx, depth + 1, leaves)) return false;
  return true;
}

// Tipo do Arrow e largura de uma coluna plana; fora do suporte: unsupported()
ColInfo describe(const Elem& e, const std::string& name) {
  ColInfo c;
  if (e.rep == 2) unsupported("coluna repetida (lista): " + name);
  if (e.rep != 0 && e.rep != 1) corrupt("repetition_type inválido em " + name);
  c.phys = e.type;
  c.max_def = e.rep == 1 ? 1 : 0;
  const bool lg = e.logical != L_NONE;
  switch (e.type) {
    case P_I32:
      c.width = 4;
      if (lg ? (e.logical == L_INT && e.int_bits == 32 && e.int_signed) : (e.converted == -1 || e.converted == 17)) {
        c.kind = K_I32;
        c.fmt = "i";
      } else if (lg ? e.logical == L_DATE : e.converted == 6) {
        c.kind = K_DATE;
        c.fmt = "tdD";
      } else {
        unsupported("INT32 com anotação fora do suporte: " + name);
      }
      break;
    case P_I64:
      c.width = 8;
      if (lg ? (e.logical == L_INT && e.int_bits == 64 && e.int_signed) : (e.converted == -1 || e.converted == 18)) {
        c.kind = K_I64;
        c.fmt = "l";
      } else if (e.logical == L_TS) {
        if (e.ts_utc) unsupported("timestamp com fuso: " + name);
        c.kind = e.ts_unit == 1 ? K_TS_MS : e.ts_unit == 2 ? K_TS_US : e.ts_unit == 3 ? K_TS_NS : -1;
        if (c.kind < 0) unsupported("unidade de timestamp desconhecida: " + name);
        c.fmt = e.ts_unit == 1 ? "tsm:" : e.ts_unit == 2 ? "tsu:" : "tsn:";
      } else {
        unsupported("INT64 com anotação fora do suporte: " + name);
      }
      break;
    case P_FLOAT:
      if (lg || e.converted != -1) unsupported("FLOAT anotado: " + name);
      c.kind = K_F32, c.fmt = "f", c.width = 4;
      break;
    case P_DOUBLE:
      if (lg || e.converted != -1) unsupported("DOUBLE anotado: " + name);
      c.kind = K_F64, c.fmt = "g", c.width = 8;
      break;
    case P_BOOL:
      if (lg || e.converted != -1) unsupported("BOOLEAN anotado: " + name);
      c.kind = K_BOOL, c.fmt = "b", c.width = 0;
      break;
    case P_BYTES:
      if (lg ? e.logical == L_STRING : e.converted == 0) {
        c.kind = K_STRING, c.fmt = "vu", c.width = 0;
      } else {
        unsupported("BYTE_ARRAY que não é texto: " + name);
      }
      break;
    default: unsupported("tipo físico fora do suporte (INT96 ou FIXED_LEN_BYTE_ARRAY): " + name);
  }
  return c;
}

void parse_chunk_meta(Reader& r, Chunk& c) {
  int last = 0, id, type;
  while (next_field(r, &last, &id, &type)) {
    switch (id) {
      case 1: c.type = (int)rint(r, type); break;
      case 4: c.codec = (int)rint(r, type); break;
      case 5: c.num_values = rint(r, type); break;
      case 6: c.uncomp = rint(r, type); break;
      case 7: c.comp = rint(r, type); break;
      case 9: c.data_off = rint(r, type); break;
      case 11: c.dict_off = rint(r, type); break;
      default: r.skip(type, 1); break;
    }
  }
}

void parse_chunk(Reader& r, Chunk& c) {
  int last = 0, id, type;
  c.present = true;
  while (next_field(r, &last, &id, &type)) {
    if (id == 1) {
      c.external = true;
      r.skip(type, 1);
    } else if (id == 3 && type == T_STRUCT) {
      parse_chunk_meta(r, c);
    } else if (id == 8 || id == 9) {  // crypto_metadata / encrypted_column_metadata
      c.external = true;
      r.skip(type, 1);
    } else {
      r.skip(type, 1);
    }
  }
}

void parse_row_group(Reader& r, Footer& F, RowGroup& rg) {
  int last = 0, id, type;
  rg.chunks.assign(F.cols.size(), Chunk{});
  while (next_field(r, &last, &id, &type)) {
    if (id == 1 && type == T_LIST) {
      u64 n;
      int et;
      r.list_header(&n, &et);
      if (et != T_STRUCT) corrupt("rodapé inválido (colunas do row group)");
      for (u64 leaf = 0; leaf < n && r.ok; leaf++) {
        const int slot = leaf < F.slot_of_leaf.size() ? F.slot_of_leaf[leaf] : -1;
        if (slot >= 0) parse_chunk(r, rg.chunks[slot]);
        else r.skip(T_STRUCT, 1);
      }
    } else if (id == 3 && type == T_I64) {
      rg.rows = r.zigzag();
    } else {
      r.skip(type, 1);
    }
  }
}

void parse_footer(const u8* p, i64 n, const std::vector<std::string>& names, Footer& F) {
  Reader r{p, p + n};
  int last = 0, id, type;
  while (next_field(r, &last, &id, &type)) {
    if (id == 2 && type == T_LIST) {
      u64 cnt;
      int et;
      r.list_header(&cnt, &et);
      if (et != T_STRUCT || cnt == 0 || cnt > (u64)n) corrupt("rodapé inválido (schema)");
      F.schema.resize((size_t)cnt);
      for (u64 i = 0; i < cnt && r.ok; i++) parse_elem(r, F.schema[(size_t)i]);
      if (!r.ok) break;
      // colunas do primeiro nível, e a posição da folha de cada uma
      const Elem& root = F.schema[0];
      std::vector<std::pair<size_t, int>> top;  // (índice do elemento, índice da primeira folha)
      size_t idx = 1;
      int leaf = 0;
      for (int c = 0; c < root.nchild; c++) {
        const size_t start = idx;
        int nl = 0;
        if (!walk_leaves(F.schema, idx, 0, nl)) corrupt("rodapé inválido (árvore do schema)");
        top.emplace_back(start, leaf);
        leaf += nl;
      }
      F.slot_of_leaf.assign((size_t)leaf, -1);
      for (size_t c = 0; c < names.size(); c++) {
        size_t found = top.size();
        for (size_t t = 0; t < top.size(); t++)
          if (F.schema[top[t].first].name == names[c]) {
            found = t;
            break;
          }
        if (found == top.size()) unsupported("coluna ausente no arquivo: " + names[c]);
        const Elem& e = F.schema[top[found].first];
        if (e.nchild != 0) unsupported("coluna aninhada: " + names[c]);
        if (F.slot_of_leaf[(size_t)top[found].second] != -1) unsupported("coluna repetida na lista: " + names[c]);
        F.slot_of_leaf[(size_t)top[found].second] = (int)c;
        F.cols.resize(names.size());
        F.cols[c] = describe(e, names[c]);
      }
      F.cols.resize(names.size());
      F.have_schema = true;
    } else if (id == 4 && type == T_LIST) {
      if (!F.have_schema) corrupt("rodapé inválido (row groups antes do schema)");
      u64 cnt;
      int et;
      r.list_header(&cnt, &et);
      if (et != T_STRUCT || cnt > (u64)n) corrupt("rodapé inválido (row groups)");
      F.rgs.resize((size_t)cnt);
      for (u64 i = 0; i < cnt && r.ok; i++) parse_row_group(r, F, F.rgs[(size_t)i]);
    } else {
      r.skip(type, 1);
    }
  }
  if (!r.ok) corrupt("rodapé do Parquet truncado ou inválido");
  if (!F.have_schema) corrupt("rodapé sem schema");
}

// ----------------------------------------------------------------------------- bits, níveis e índices
inline int popcnt64(u64 x) {
  x = x - ((x >> 1) & 0x5555555555555555ull);
  x = (x & 0x3333333333333333ull) + ((x >> 2) & 0x3333333333333333ull);
  x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0Full;
  return (int)((x * 0x0101010101010101ull) >> 56);
}

// k <= 56 bits de src a partir do bit sbit (o chamador garante que estão dentro do buffer)
inline u64 load_bits(const u8* src, i64 sbit, int k) {
  const u8* p = src + (sbit >> 3);
  const int sh = (int)(sbit & 7);
  const int need = (sh + k + 7) >> 3;
  u64 w = 0;
  for (int i = 0; i < need; i++) w |= (u64)p[i] << (8 * i);
  w >>= sh;
  return k >= 64 ? w : (w & ((1ull << k) - 1));
}

// OR de k <= 56 bits em dst a partir do bit dbit (dst já zerado nessa faixa)
inline void or_bits(u8* dst, i64 dbit, u64 w, int k) {
  const int sh = (int)(dbit & 7);
  u8* p = dst + (dbit >> 3);
  w <<= sh;
  const int cnt = (sh + k + 7) >> 3;
  for (int i = 0; i < cnt; i++) p[i] |= (u8)(w >> (8 * i));
}

// copia n bits de src (a partir de sbit) para dst (a partir de dbit); devolve quantos eram 1
i64 copy_bits(u8* dst, i64 dbit, const u8* src, i64 sbit, i64 n) {
  i64 ones = 0;
  while (n > 0) {
    const int k = (int)std::min<i64>(n, 56);
    const u64 w = load_bits(src, sbit, k);
    or_bits(dst, dbit, w, k);
    ones += popcnt64(w);
    sbit += k;
    dbit += k;
    n -= k;
  }
  return ones;
}

void set_ones(u8* dst, i64 dbit, i64 n) {
  while (n > 0) {
    const int k = (int)std::min<i64>(n, 56);
    or_bits(dst, dbit, (1ull << k) - 1, k);
    dbit += k;
    n -= k;
  }
}

bool rd_varint(const u8* p, i64 len, i64& pos, u64& v) {
  v = 0;
  for (int shift = 0; shift < 64; shift += 7) {
    if (pos >= len) return false;
    const u8 b = p[pos++];
    v |= (u64)(b & 0x7F) << shift;
    if (!(b & 0x80)) return true;
  }
  return false;
}

// RLE/bit-packed híbrido com largura de 1 bit (níveis de definição, booleano RLE): grava `nvals` bits em dst a partir
// de dbit0 (dst zerado) e devolve quantos eram 1. Falha com corrupt() se o bloco não tiver exatamente nvals valores.
i64 decode_bits1(const u8* p, i64 len, i64 nvals, u8* dst, i64 dbit0) {
  i64 pos = 0, done = 0, ones = 0;
  while (done < nvals) {
    u64 h;
    if (!rd_varint(p, len, pos, h)) corrupt("níveis truncados");
    if ((h & 1) == 0) {  // run repetido
      const u64 count = h >> 1;
      if (count == 0 || pos >= len) corrupt("run RLE inválido");
      const u8 v = p[pos++];
      if (v > 1) corrupt("nível de definição inválido");
      if (count > (u64)(nvals - done)) corrupt("run RLE passa do número de valores");
      if (v) {
        set_ones(dst, dbit0 + done, (i64)count);
        ones += (i64)count;
      }
      done += (i64)count;
    } else {  // grupos de 8 valores, 1 bit cada
      const u64 groups = h >> 1;
      if (groups == 0 || groups > (u64)(len - pos)) corrupt("run bit-packed inválido");
      const i64 take = std::min<i64>((i64)groups * 8, nvals - done);
      ones += copy_bits(dst, dbit0 + done, p + pos, 0, take);
      pos += (i64)groups;
      done += take;
    }
  }
  return ones;
}

// índices de dicionário: largura `bw` (0..32), `count` valores em out
void decode_indices(const u8* p, i64 len, int bw, i64 count, u32* out) {
  if (bw < 0 || bw > 32) corrupt("largura de índice inválida");
  const u64 mask = (1ull << bw) - 1;
  const int vbytes = (bw + 7) / 8;
  i64 pos = 0, done = 0;
  while (done < count) {
    u64 h;
    if (!rd_varint(p, len, pos, h)) corrupt("índices truncados");
    if ((h & 1) == 0) {
      const u64 c = h >> 1;
      if (c == 0 || len - pos < vbytes) corrupt("run RLE inválido nos índices");
      u64 v = 0;
      for (int k = 0; k < vbytes; k++) v |= (u64)p[pos + k] << (8 * k);
      pos += vbytes;
      if (c > (u64)(count - done)) corrupt("run RLE passa do número de índices");
      const u32 val = (u32)(v & mask);
      for (u64 k = 0; k < c; k++) out[done + (i64)k] = val;
      done += (i64)c;
    } else {
      const u64 groups = h >> 1;
      if (groups == 0 || (bw > 0 && groups > (u64)(len - pos) / (u64)bw)) corrupt("run bit-packed inválido nos índices");
      const i64 take = std::min<i64>((i64)groups * 8, count - done);
      const u8* s = p + pos;
      u64 acc = 0;
      int nb = 0;
      for (i64 i = 0; i < take; i++) {
        while (nb < bw) {
          acc |= (u64)*s++ << nb;
          nb += 8;
        }
        out[done + i] = (u32)(acc & mask);
        acc >>= bw;
        nb -= bw;
      }
      pos += (i64)groups * bw;
      done += take;
    }
  }
}

// corre os trechos de presentes (bit 1) e nulos (bit 0) de n linhas a partir de row0 em `valid`
template <class FP, class FN>
void for_runs(const u8* valid, i64 row0, i64 n, FP&& present, FN&& null) {
  i64 pos = 0, dense = 0;
  while (pos < n) {
    const i64 bit = row0 + pos;
    const bool v = bit_at(valid, bit);
    const i64 limit = n - pos;
    i64 len = 1;
    while (len < limit) {
      const i64 b = bit + len;
      if ((b & 7) == 0 && len + 8 <= limit) {
        const u8 byte = valid[b >> 3];
        if (byte == (v ? 0xFF : 0x00)) {
          len += 8;
          continue;
        }
      }
      if (bit_at(valid, b) != v) break;
      len++;
    }
    if (v) {
      present(row0 + pos, dense, len);
      dense += len;
    } else {
      null(row0 + pos, len);
    }
    pos += len;
  }
}

bool valid_utf8(const u8* s, i64 n) {
  i64 i = 0;
  while (i < n) {
    while (i + 8 <= n) {
      u64 w;
      std::memcpy(&w, s + i, 8);
      if (w & 0x8080808080808080ull) break;
      i += 8;
    }
    if (i >= n) break;
    const u8 c = s[i];
    if (c < 0x80) {
      i++;
      continue;
    }
    int more;
    u8 lo = 0x80, hi = 0xBF;
    if (c >= 0xC2 && c <= 0xDF) more = 1;
    else if (c >= 0xE0 && c <= 0xEF) {
      more = 2;
      if (c == 0xE0) lo = 0xA0;
      if (c == 0xED) hi = 0x9F;
    } else if (c >= 0xF0 && c <= 0xF4) {
      more = 3;
      if (c == 0xF0) lo = 0x90;
      if (c == 0xF4) hi = 0x8F;
    } else {
      return false;
    }
    if (n - i <= more) return false;
    if (s[i + 1] < lo || s[i + 1] > hi) return false;
    for (int k = 2; k <= more; k++)
      if ((s[i + k] & 0xC0) != 0x80) return false;
    i += more + 1;
  }
  return true;
}

// ----------------------------------------------------------------------------- codecs
bool gunzip(const u8* src, i64 n, u8* dst, i64 dst_len) {
  tinfl_decompressor* d = tinfl_decompressor_alloc();
  if (d == nullptr) throw std::bad_alloc();
  i64 pos = 0, o = 0;
  bool ok = true;
  while (ok && pos < n) {
    if (n - pos < 10 || src[pos] != 0x1F || src[pos + 1] != 0x8B || src[pos + 2] != 8) {
      ok = false;
      break;
    }
    const u8 flg = src[pos + 3];
    pos += 10;
    if (flg & 4) {
      if (n - pos < 2) { ok = false; break; }
      const i64 xlen = src[pos] | (src[pos + 1] << 8);
      pos += 2;
      if (n - pos < xlen) { ok = false; break; }
      pos += xlen;
    }
    for (const int bit : {8, 16}) {
      if (!(flg & bit)) continue;
      while (pos < n && src[pos] != 0) pos++;
      if (pos >= n) { ok = false; break; }
      pos++;
    }
    if (!ok) break;
    if (flg & 2) pos += 2;
    if (pos >= n) { ok = false; break; }
    tinfl_init(d);
    size_t in = (size_t)(n - pos), out = (size_t)(dst_len - o);
    const tinfl_status st = tinfl_decompress(d, src + pos, &in, dst, dst + o, &out, TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF);
    if (st != TINFL_STATUS_DONE) { ok = false; break; }
    pos += (i64)in;
    o += (i64)out;
    if (n - pos < 8) { ok = false; break; }
    pos += 8;  // CRC-32 e tamanho: os tamanhos já são conferidos pelo chamador
  }
  tinfl_decompressor_free(d);
  return ok && o == dst_len;
}

void decompress(int codec, const u8* src, i64 n, u8* dst, i64 dst_len) {
  switch (codec) {
    case C_SNAPPY:
      if (!dc::snappy_decompress(src, n, dst, dst_len)) corrupt("página snappy corrompida");
      return;
    case C_ZSTD: {
      if (dst_len == 0 && n == 0) return;
      const size_t r = ZSTD_decompress(dst, (size_t)dst_len, src, (size_t)n);
      if (ZSTD_isError(r) || r != (size_t)dst_len) corrupt("página zstd corrompida");
      return;
    }
    case C_GZIP:
      if (!gunzip(src, n, dst, dst_len)) corrupt("página gzip corrompida");
      return;
    default: unsupported("codec fora do suporte");
  }
}


// ----------------------------------------------------------------------------- saída (buffers do pool)
void* alloc_block(i64 bytes, bool zero) {
  const i64 b = bytes <= 0 ? 64 : bytes;
  void* p = dc::pool_alloc(b);
  if (p == nullptr) corrupt("sem memória para as colunas");
  if (zero) std::memset(p, 0, (size_t)b);
  return p;
}

constexpr i64 MAX_ROWS = (i64)1 << 40;
constexpr i64 MAX_TEXT = 0x7FFFFFFF;  // as views do Arrow endereçam o buffer com 32 bits

struct Out {
  const ColInfo* ci;
  i64 n;
  u8* values = nullptr;
  u8* valid = nullptr;
  u8* sdata = nullptr;
  i64 scap = 0, slen = 0, nulls = 0;
  bool moved = false;

  Out(const ColInfo* c, i64 rows) : ci(c), n(rows) {
    if (ci->max_def > 0) valid = (u8*)alloc_block((n + 7) / 8, true);
    if (ci->kind == K_BOOL) values = (u8*)alloc_block((n + 7) / 8, true);
    else if (ci->kind == K_STRING) values = (u8*)alloc_block(n * 16, true);
    else values = (u8*)alloc_block(n * ci->width, false);
  }
  ~Out() {
    if (moved) return;
    dc::pool_free(values);
    dc::pool_free(valid);
    dc::pool_free(sdata);
  }
  Out(const Out&) = delete;
  Out& operator=(const Out&) = delete;

  void reserve_text(i64 need) {
    if (need <= scap) return;
    if (need > MAX_TEXT) unsupported("texto de uma coluna passa de 2 GiB num row group");
    const i64 nc = std::min<i64>(std::max<i64>({need, scap * 2, (i64)1 << 16}), MAX_TEXT);
    u8* nb = (u8*)alloc_block(nc, false);
    if (slen > 0) std::memcpy(nb, sdata, (size_t)slen);
    dc::pool_free(sdata);
    sdata = nb;
    scap = nc;
  }

  // view de 16 bytes: tamanho + (até 12 bytes inline | 4 de prefixo, índice do buffer e offset)
  void put_view(u8* d, const u8* s, i64 len) {
    const u32 l = (u32)len;
    std::memcpy(d, &l, 4);
    if (len <= 12) {
      std::memcpy(d + 4, s, (size_t)len);
      std::memset(d + 4 + len, 0, (size_t)(12 - len));
    } else {
      reserve_text(slen + len);
      std::memcpy(sdata + slen, s, (size_t)len);
      const u32 zero = 0, off = (u32)slen;
      std::memcpy(d + 4, s, 4);
      std::memcpy(d + 8, &zero, 4);
      std::memcpy(d + 12, &off, 4);
      slen += len;
    }
  }
};

struct Private {
  std::vector<void*> blocks;
  std::vector<const void*> bufs;
  std::vector<i64> sizes;
};

void release_array(ArrowArray* a) {
  if (a == nullptr || a->release == nullptr) return;
  Private* p = (Private*)a->private_data;
  if (p != nullptr) {
    for (void* b : p->blocks) dc::pool_free(b);
    delete p;
  }
  a->release = nullptr;
}

void release_schema(ArrowSchema* s) {
  if (s != nullptr) s->release = nullptr;
}

void finish(Out& o, ArrowSchema* sc, ArrowArray* ar) {
  std::unique_ptr<Private> pv(new Private);
  if (o.nulls == 0 && o.valid != nullptr) {
    dc::pool_free(o.valid);
    o.valid = nullptr;
  }
  if (o.valid) pv->blocks.push_back(o.valid);
  pv->blocks.push_back(o.values);
  pv->sizes.reserve(1);
  pv->bufs.push_back(o.valid);
  pv->bufs.push_back(o.values);
  if (o.ci->kind == K_STRING) {
    if (o.slen > 0) {
      pv->blocks.push_back(o.sdata);
      pv->bufs.push_back(o.sdata);
      pv->sizes.push_back(o.slen);
    } else {
      dc::pool_free(o.sdata);
    }
    pv->sizes.push_back(0);  // garante um endereço válido mesmo sem buffer de dados
    pv->bufs.push_back(pv->sizes.data());  // tamanhos dos buffers de dados (ao fim, na ordem do Arrow)
  }
  ar->length = o.n;
  ar->null_count = o.nulls;
  ar->offset = 0;
  ar->n_buffers = (i64)pv->bufs.size();
  ar->n_children = 0;
  ar->buffers = pv->bufs.data();
  ar->children = nullptr;
  ar->dictionary = nullptr;
  ar->private_data = pv.release();
  ar->release = release_array;
  o.moved = true;
  sc->format = o.ci->fmt;
  sc->name = "";
  sc->metadata = nullptr;
  sc->flags = 2;  // ARROW_FLAG_NULLABLE
  sc->n_children = 0;
  sc->children = nullptr;
  sc->dictionary = nullptr;
  sc->private_data = nullptr;
  sc->release = release_schema;
}

// ----------------------------------------------------------------------------- páginas
struct PageHdr {
  int type = -1;
  i64 usize = -1, csize = -1;
  i64 nvals = -1, enc = -1, def_enc = E_RLE, def_len = 0, rep_len = 0;
  bool compressed = true;
};

PageHdr parse_page_header(Reader& r) {
  PageHdr h;
  int last = 0, id, type;
  while (next_field(r, &last, &id, &type)) {
    if (id == 1) h.type = (int)rint(r, type);
    else if (id == 2) h.usize = rint(r, type);
    else if (id == 3) h.csize = rint(r, type);
    else if ((id == 5 || id == 7 || id == 8) && type == T_STRUCT) {
      int l2 = 0, i2, t2;
      while (next_field(r, &l2, &i2, &t2)) {
        if (i2 == 1) h.nvals = rint(r, t2);
        else if (id == 8 && i2 == 4) h.enc = rint(r, t2);
        else if (id != 8 && i2 == 2) h.enc = rint(r, t2);
        else if (id == 5 && i2 == 3) h.def_enc = rint(r, t2);
        else if (id == 8 && i2 == 5) h.def_len = rint(r, t2);
        else if (id == 8 && i2 == 6) h.rep_len = rint(r, t2);
        else if (id == 8 && i2 == 7) h.compressed = (t2 == T_TRUE);
        else r.skip(t2, 2);
      }
    } else {
      r.skip(type, 1);
    }
  }
  if (!r.ok) corrupt("cabeçalho de página truncado ou inválido");
  return h;
}

inline u32 le32(const u8* p) {
  u32 v;
  std::memcpy(&v, p, 4);
  return v;
}

struct Decoder {
  const ColInfo& ci;
  const Chunk& cm;
  Out& out;
  i64 n;
  i64 row = 0;
  std::vector<u8> tmp, dict, bits;
  std::vector<u32> idx;
  i64 dict_count = -1;

  Decoder(const ColInfo& c, const Chunk& m, Out& o, i64 rows) : ci(c), cm(m), out(o), n(rows) {}

  bool is_text() const { return ci.kind == K_STRING; }
  bool is_bool() const { return ci.kind == K_BOOL; }

  const u8* uncompressed(const u8* body, i64 csize, i64 usize) {
    if (cm.codec == C_NONE) {
      if (csize != usize) corrupt("tamanhos da página não conferem (sem compressão)");
      return body;
    }
    tmp.resize((size_t)usize);
    decompress(cm.codec, body, csize, tmp.data(), usize);
    return tmp.data();
  }

  void dict_page(const PageHdr& h, const u8* body) {
    if (dict_count >= 0) corrupt("duas páginas de dicionário no mesmo chunk");
    if (is_bool()) unsupported("dicionário de booleano");
    if (h.enc != E_PLAIN && h.enc != E_PLAIN_DICT) unsupported("encoding do dicionário fora do suporte");
    const i64 nv = h.nvals;
    const u8* d = uncompressed(body, h.csize, h.usize);
    const i64 len = h.usize;
    if (nv < 0) corrupt("dicionário com contagem negativa");
    if (is_text()) {
      if (nv > len / 4) corrupt("dicionário maior que a página");
      dict.assign((size_t)nv * 16, 0);
      i64 pos = 0;
      for (i64 i = 0; i < nv; i++) {
        if (len - pos < 4) corrupt("dicionário de texto truncado");
        const i64 l = le32(d + pos);
        pos += 4;
        if (l > len - pos) corrupt("texto do dicionário passa do fim da página");
        if (!valid_utf8(d + pos, l)) corrupt("texto do dicionário não é UTF-8 válido");
        out.put_view(dict.data() + (size_t)i * 16, d + pos, l);
        pos += l;
      }
    } else {
      if (nv > len / ci.width) corrupt("dicionário maior que a página");
      dict.assign(d, d + (size_t)nv * ci.width);
    }
    dict_count = nv;
  }

  // chama fp(linha, posição densa, tamanho) para cada trecho de valores e fn(linha, tamanho) para os nulos
  template <class FP, class FN>
  void scatter(i64 nv, i64 present, FP&& fp, FN&& fn) {
    if (out.valid == nullptr || present == nv) fp(row, (i64)0, nv);
    else for_runs(out.valid, row, nv, fp, fn);
  }

  void data_page(const PageHdr& h, const u8* body) {
    const i64 nv = h.nvals;
    if (nv < 0 || nv > n - row) corrupt("página com mais valores que o row group");
    const u8 *defp = nullptr, *vals;
    i64 deflen = 0, vlen;
    if (h.type == PG_DATA_V2) {
      const i64 lv = h.rep_len + h.def_len;
      if (h.rep_len < 0 || h.def_len < 0 || lv > h.csize || lv > h.usize) corrupt("níveis da página v2 inválidos");
      if (h.rep_len != 0) unsupported("níveis de repetição");
      if (ci.max_def == 0 && h.def_len != 0) corrupt("níveis de definição numa coluna obrigatória");
      defp = body;
      deflen = h.def_len;
      const u8* vsrc = body + lv;
      const i64 vc = h.csize - lv, vu = h.usize - lv;
      if (cm.codec == C_NONE || !h.compressed) {
        if (vc != vu) corrupt("tamanhos da página v2 não conferem");
        vals = vsrc;
      } else {
        tmp.resize((size_t)vu);
        decompress(cm.codec, vsrc, vc, tmp.data(), vu);
        vals = tmp.data();
      }
      vlen = vu;
    } else {
      const u8* all = uncompressed(body, h.csize, h.usize);
      i64 pos = 0;
      if (ci.max_def > 0) {
        if (h.def_enc != E_RLE) unsupported("níveis de definição fora de RLE");
        if (h.usize < 4) corrupt("página sem níveis de definição");
        const i64 l = le32(all);
        if (l > h.usize - 4) corrupt("níveis de definição passam do fim da página");
        defp = all + 4;
        deflen = l;
        pos = 4 + l;
      }
      vals = all + pos;
      vlen = h.usize - pos;
    }
    i64 present = nv;
    if (ci.max_def > 0) {
      present = decode_bits1(defp, deflen, nv, out.valid, row);
      out.nulls += nv - present;
    }
    const bool dict_enc = h.enc == E_PLAIN_DICT || h.enc == E_RLE_DICT;
    if (dict_enc) {
      if (dict_count < 0) corrupt("página de dicionário ausente");
      if (is_bool()) unsupported("booleano com dicionário");
      if (vlen < 1) corrupt("página de índices vazia");
      idx.resize((size_t)present);
      decode_indices(vals + 1, vlen - 1, vals[0], present, idx.data());
      gather(nv, present);
    } else if (h.enc == E_PLAIN) {
      plain(vals, vlen, nv, present);
    } else if (h.enc == E_RLE && is_bool()) {
      if (vlen < 4) corrupt("booleano RLE truncado");
      const i64 l = le32(vals);
      if (l > vlen - 4) corrupt("booleano RLE passa do fim da página");
      bits.assign((size_t)(present + 7) / 8 + 8, 0);
      decode_bits1(vals + 4, l, present, bits.data(), 0);
      bool_scatter(bits.data(), nv, present);
    } else {
      unsupported("encoding de página fora do suporte");
    }
    row += nv;
  }

  void bool_scatter(const u8* src, i64 nv, i64 present) {
    scatter(nv, present, [&](i64 r, i64 d, i64 len) { copy_bits(out.values, r, src, d, len); }, [&](i64, i64) {});
  }

  template <int W>
  void plain_fixed(const u8* vals, i64 vlen, i64 nv, i64 present) {
    if (present > vlen / W) corrupt("página com menos bytes que valores");
    u8* dst = out.values;
    scatter(
        nv, present, [&](i64 r, i64 d, i64 len) { std::memcpy(dst + r * W, vals + d * W, (size_t)(len * W)); },
        [&](i64 r, i64 len) { std::memset(dst + r * W, 0, (size_t)(len * W)); });
  }

  void plain(const u8* vals, i64 vlen, i64 nv, i64 present) {
    if (is_bool()) {
      if ((present + 7) / 8 > vlen) corrupt("página de booleanos truncada");
      bool_scatter(vals, nv, present);
    } else if (is_text()) {
      i64 pos = 0;
      u8* views = out.values;
      scatter(
          nv, present,
          [&](i64 r, i64, i64 len) {
            for (i64 k = 0; k < len; k++) {
              if (vlen - pos < 4) corrupt("página de texto truncada");
              const i64 l = le32(vals + pos);
              pos += 4;
              if (l > vlen - pos) corrupt("texto passa do fim da página");
              if (!valid_utf8(vals + pos, l)) corrupt("texto não é UTF-8 válido");
              out.put_view(views + (r + k) * 16, vals + pos, l);
              pos += l;
            }
          },
          [&](i64 r, i64 len) { std::memset(views + r * 16, 0, (size_t)(len * 16)); });
    } else if (ci.width == 4) {
      plain_fixed<4>(vals, vlen, nv, present);
    } else {
      plain_fixed<8>(vals, vlen, nv, present);
    }
  }

  template <int W>
  void gather_fixed(i64 nv, i64 present) {
    const u8* dv = dict.data();
    u8* dst = out.values;
    const u32* ix = idx.data();
    const i64 dcount = dict_count;
    scatter(
        nv, present,
        [&](i64 r, i64 d, i64 len) {
          for (i64 k = 0; k < len; k++) {
            const u32 i = ix[d + k];
            if ((i64)i >= dcount) corrupt("índice fora do dicionário");
            std::memcpy(dst + (r + k) * W, dv + (size_t)i * W, W);
          }
        },
        [&](i64 r, i64 len) { std::memset(dst + r * W, 0, (size_t)(len * W)); });
  }

  void gather(i64 nv, i64 present) {
    if (is_text()) {
      const u8* dv = dict.data();
      u8* dst = out.values;
      const u32* ix = idx.data();
      const i64 dcount = dict_count;
      scatter(
          nv, present,
          [&](i64 r, i64 d, i64 len) {
            for (i64 k = 0; k < len; k++) {
              const u32 i = ix[d + k];
              if ((i64)i >= dcount) corrupt("índice fora do dicionário");
              std::memcpy(dst + (r + k) * 16, dv + (size_t)i * 16, 16);
            }
          },
          [&](i64 r, i64 len) { std::memset(dst + r * 16, 0, (size_t)(len * 16)); });
    } else if (ci.width == 4) {
      gather_fixed<4>(nv, present);
    } else {
      gather_fixed<8>(nv, present);
    }
  }

  void run(const u8* chunk, i64 size) {
    const u8* p = chunk;
    const u8* end = chunk + size;
    while (row < n) {
      if (p >= end) corrupt("o column chunk termina antes das linhas esperadas");
      Reader r{p, end};
      const PageHdr h = parse_page_header(r);
      const u8* body = r.p;
      if (h.csize < 0 || h.csize > end - body || h.usize < 0 || h.usize > 0x7FFFFFFF)
        corrupt("tamanho de página inválido");
      if (h.type == PG_DICT) {
        if (row != 0) corrupt("página de dicionário depois dos dados");
        dict_page(h, body);
      } else if (h.type == PG_DATA || h.type == PG_DATA_V2) {
        data_page(h, body);
      } else if (h.type != PG_INDEX) {
        corrupt("tipo de página desconhecido");
      }
      p = body + h.csize;
    }
  }
};

void read_chunk(const std::filesystem::path& path, const ColInfo& ci, const Chunk& cm, i64 start, i64 rows,
                ArrowSchema* sc, ArrowArray* ar) {
  Out out(&ci, rows);
  if (rows > 0) {
    std::vector<u8> buf((size_t)cm.comp);
    {
      std::ifstream f(path, std::ios::binary);
      if (!f) corrupt("não foi possível abrir o arquivo");
      f.seekg((std::streamoff)start);
      f.read((char*)buf.data(), (std::streamsize)cm.comp);
      if (!f || f.gcount() != (std::streamsize)cm.comp) corrupt("leitura incompleta do column chunk");
    }
    Decoder d(ci, cm, out, rows);
    d.run(buf.data(), cm.comp);
  }
  finish(out, sc, ar);
}

}  // namespace

// Mensagem do último erro (-1 ou -2) de dc_parquet_read nesta thread; devolve o tamanho gravado em buf (sem NUL).
DC_EXPORT i64 dc_parquet_error(u8* buf, i64 cap) {
  const i64 n = std::min<i64>((i64)g_error.size(), cap);
  if (n > 0) std::memcpy(buf, g_error.data(), (size_t)n);
  return n;
}

// Lê as colunas `names` dos row groups `row_groups` do Parquet em `path`; preenche n_rg * n_col pares (schema, array)
// em ordem rg-major (um por row group e coluna) e devolve o total de linhas. -1: fora do suporte (usar o Polars); -2:
// erro de E/S ou arquivo corrompido (mensagem em dc_parquet_error). Em erro, nenhum par fica preenchido.
DC_EXPORT i64 dc_parquet_read(const u8* path, i64 path_len, const i32* row_groups, i64 n_rg, const u8* names,
                              const i64* name_lens, i64 n_col, int threads, u8* schemas, u8* arrays) {
  g_error.clear();
  ArrowSchema* sch = (ArrowSchema*)schemas;
  ArrowArray* arr = (ArrowArray*)arrays;
  if (n_rg < 0 || n_col <= 0 || path_len <= 0) {
    g_error = "argumentos inválidos";
    return -2;
  }
  const i64 total = n_rg * n_col;
  std::memset(sch, 0, sizeof(ArrowSchema) * (size_t)total);
  std::memset(arr, 0, sizeof(ArrowArray) * (size_t)total);
  std::atomic<int> code{0};
  std::mutex mu;
  std::string msg;
  auto fail = [&](int c, const std::string& m) {
    std::lock_guard<std::mutex> lock(mu);
    if (code.load() == 0) {
      msg = m;
      code.store(c);
    }
  };
  i64 rows_total = 0;
  try {
    std::vector<std::string> cols;
    i64 off = 0;
    for (i64 i = 0; i < n_col; i++) {
      if (name_lens[i] < 0) corrupt("argumentos inválidos");
      cols.emplace_back((const char*)names + off, (size_t)name_lens[i]);
      off += name_lens[i];
    }
    const std::filesystem::path fpath = std::filesystem::u8path(std::string((const char*)path, (size_t)path_len));
    i64 fsize;
    std::vector<u8> foot;
    {
      std::ifstream f(fpath, std::ios::binary);
      if (!f) corrupt("não foi possível abrir o arquivo");
      f.seekg(0, std::ios::end);
      fsize = (i64)f.tellg();
      if (fsize < 12) corrupt("arquivo curto demais para ser Parquet");
      u8 tail[8];
      f.seekg((std::streamoff)(fsize - 8));
      f.read((char*)tail, 8);
      if (!f) corrupt("leitura do fim do arquivo falhou");
      if (std::memcmp(tail + 4, "PARE", 4) == 0) unsupported("Parquet com rodapé criptografado");
      if (std::memcmp(tail + 4, "PAR1", 4) != 0) corrupt("não é um arquivo Parquet (falta PAR1 no fim)");
      const i64 flen = le32(tail);
      if (flen == 0 || flen > fsize - 12) corrupt("tamanho do rodapé inválido");
      foot.resize((size_t)flen);
      f.seekg((std::streamoff)(fsize - 8 - flen));
      f.read((char*)foot.data(), (std::streamsize)flen);
      if (!f) corrupt("leitura do rodapé falhou");
    }
    Footer F;
    parse_footer(foot.data(), (i64)foot.size(), cols, F);
    // confere os row groups pedidos e os metadados de cada column chunk antes de ler qualquer dado
    std::vector<i64> starts((size_t)total, 0);
    for (i64 ri = 0; ri < n_rg; ri++) {
      const i32 g = row_groups[ri];
      if (g < 0 || (size_t)g >= F.rgs.size()) corrupt("row group fora do arquivo");
      const RowGroup& rg = F.rgs[(size_t)g];
      if (rg.rows < 0 || rg.rows > MAX_ROWS) corrupt("número de linhas do row group inválido");
      rows_total += rg.rows;
      for (i64 c = 0; c < n_col; c++) {
        const Chunk& cm = rg.chunks[(size_t)c];
        const ColInfo& ci = F.cols[(size_t)c];
        if (!cm.present || cm.type < 0) corrupt("column chunk sem metadados");
        if (cm.external) unsupported("column chunk em outro arquivo ou criptografado");
        if (cm.type != ci.phys) corrupt("tipo do column chunk difere do schema");
        if (cm.codec != C_NONE && cm.codec != C_SNAPPY && cm.codec != C_GZIP && cm.codec != C_ZSTD)
          unsupported("codec " + std::to_string(cm.codec) + " fora do suporte");
        if (rg.rows == 0) continue;
        if (cm.num_values != rg.rows) corrupt("número de valores do chunk difere do row group");
        i64 start = cm.data_off;
        if (cm.dict_off > 0 && cm.dict_off < start) start = cm.dict_off;
        if (start < 4 || cm.comp < 0 || cm.comp > fsize || start > fsize - 8 - cm.comp)
          corrupt("column chunk fora do arquivo");
        starts[(size_t)(ri * n_col + c)] = start;
      }
    }
    dc::parallel_for(threads < 1 ? 1 : threads, total, [&](i64 t) {
      if (code.load() != 0) return;
      try {
        const i64 ri = t / n_col, c = t % n_col;
        const RowGroup& rg = F.rgs[(size_t)row_groups[ri]];
        read_chunk(fpath, F.cols[(size_t)c], rg.chunks[(size_t)c], starts[(size_t)t], rg.rows, &sch[t], &arr[t]);
      } catch (const Fail& e) {
        fail(e.code, e.msg);
      } catch (const std::bad_alloc&) {
        fail(-2, "sem memória");
      } catch (const std::exception& e) {
        fail(-2, e.what());
      }
    });
  } catch (const Fail& e) {
    fail(e.code, e.msg);
  } catch (const std::bad_alloc&) {
    fail(-2, "sem memória");
  } catch (const std::exception& e) {
    fail(-2, e.what());
  }
  if (code.load() != 0) {
    for (i64 i = 0; i < total; i++) {
      release_array(&arr[i]);
      release_schema(&sch[i]);
    }
    g_error = msg;
    return code.load();
  }
  return rows_total;
}
