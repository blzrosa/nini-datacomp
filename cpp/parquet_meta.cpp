// parquet_meta.cpp -- linhas de cada row group de um arquivo Parquet, lidas do rodapé (Thrift, protocolo compacto).
//
// O planejamento das janelas (src/nini_datacomp/windows.py) alinha as fronteiras ao começo dos row groups, para nenhum
// row group ser decodificado duas vezes. O Polars não expõe os row groups, e o pacote não depende do pyarrow: este
// leitor percorre o FileMetaData (https://github.com/apache/parquet-format, parquet.thrift) e só guarda o num_rows de
// cada RowGroup (campo 4 do FileMetaData, campo 3 do RowGroup); o resto é pulado sem ser interpretado.
//
// Protocolo compacto do Thrift: cabeçalho de campo = 1 byte (delta do id nos 4 bits altos, tipo nos 4 baixos; delta 0
// = id em varint zigzag a seguir); inteiros em varint zigzag; binário = tamanho em varint + bytes; lista = 1 byte
// (tamanho nos 4 bits altos, 15 = tamanho em varint a seguir; tipo do elemento nos baixos); mapa = tamanho em varint e,
// se > 0, 1 byte com os tipos de chave e valor; struct = campos até o byte 0. Booleanos de campo vêm no próprio tipo
// (1 = true, 2 = false); dentro de listas, 1 byte por elemento.
#include "common.hpp"

namespace {

enum TType { T_TRUE = 1, T_FALSE = 2, T_BYTE = 3, T_I16 = 4, T_I32 = 5, T_I64 = 6, T_DOUBLE = 7, T_BINARY = 8,
             T_LIST = 9, T_SET = 10, T_MAP = 11, T_STRUCT = 12 };
const int MAX_DEPTH = 64;

struct Reader {
  const u8* p;
  const u8* end;
  bool ok = true;

  u8 byte() {
    if (p >= end) { ok = false; return 0; }
    return *p++;
  }

  u64 varint() {
    u64 v = 0;
    for (int shift = 0; shift < 64; shift += 7) {
      u8 b = byte();
      if (!ok) return 0;
      v |= (u64)(b & 0x7F) << shift;
      if (!(b & 0x80)) return v;
    }
    ok = false;
    return 0;
  }

  i64 zigzag() {
    u64 v = varint();
    return (i64)(v >> 1) ^ -(i64)(v & 1);
  }

  void advance(u64 n) {
    if ((u64)(end - p) < n) { ok = false; p = end; return; }
    p += n;
  }

  // tamanho e tipo do elemento de uma lista/conjunto
  void list_header(u64* size, int* type) {
    u8 h = byte();
    *type = h & 0x0F;
    *size = h >> 4;
    if (*size == 15) *size = varint();
  }

  void skip(int type, int depth);

  void skip_elem(int type, int depth) {
    if (type == T_TRUE || type == T_FALSE) { byte(); return; }   // booleano dentro de lista: 1 byte
    skip(type, depth);
  }

  void skip_struct(int depth) {
    if (depth > MAX_DEPTH) { ok = false; return; }
    while (ok) {
      u8 h = byte();
      if (!ok || h == 0) return;
      if ((h >> 4) == 0) zigzag();   // id do campo por extenso
      skip(h & 0x0F, depth + 1);
    }
  }
};

void Reader::skip(int type, int depth) {
  if (depth > MAX_DEPTH) { ok = false; return; }
  switch (type) {
    case T_TRUE: case T_FALSE: return;
    case T_BYTE: byte(); return;
    case T_I16: case T_I32: case T_I64: varint(); return;
    case T_DOUBLE: advance(8); return;
    case T_BINARY: advance(varint()); return;
    case T_LIST: case T_SET: {
      u64 n; int et;
      list_header(&n, &et);
      for (u64 i = 0; i < n && ok; i++) skip_elem(et, depth + 1);
      return;
    }
    case T_MAP: {
      u64 n = varint();
      if (n == 0) return;
      u8 kv = byte();
      for (u64 i = 0; i < n && ok; i++) {
        skip_elem(kv >> 4, depth + 1);
        skip_elem(kv & 0x0F, depth + 1);
      }
      return;
    }
    case T_STRUCT: skip_struct(depth + 1); return;
    default: ok = false; return;
  }
}

// próximo campo de um struct: false no fim (byte 0); `last` é o id anterior (para o delta)
bool next_field(Reader& r, int* last, int* id, int* type) {
  u8 h = r.byte();
  if (!r.ok || h == 0) return false;
  *type = h & 0x0F;
  int delta = h >> 4;
  *id = delta ? *last + delta : (int)r.zigzag();
  *last = *id;
  return r.ok;
}

i64 row_group_rows(Reader& r) {
  i64 rows = -1;
  int last = 0, id, type;
  while (next_field(r, &last, &id, &type)) {
    if (id == 3 && type == T_I64) rows = r.zigzag();
    else r.skip(type, 1);
  }
  return rows;
}

}  // namespace

// `footer`: os `n` bytes do FileMetaData (sem os 8 finais: tamanho + "PAR1"). Grava em `rows` as linhas de cada row
// group (até `cap`) e devolve quantos row groups o arquivo tem, ou -1 se o rodapé não pôde ser lido.
DC_EXPORT i64 dc_parquet_row_groups(const u8* footer, i64 n, i64* rows, i64 cap) {
  Reader r{footer, footer + n};
  i64 count = -1;
  int last = 0, id, type;
  while (next_field(r, &last, &id, &type)) {
    if (id == 4 && type == T_LIST) {
      u64 size;
      int et;
      r.list_header(&size, &et);
      if (et != T_STRUCT) return -1;
      count = 0;
      for (u64 i = 0; i < size && r.ok; i++) {
        i64 v = row_group_rows(r);
        if (v < 0) return -1;
        if (count < cap) rows[count] = v;
        count++;
      }
    } else {
      r.skip(type, 1);
    }
  }
  return r.ok && count >= 0 ? count : -1;   // sem a lista de row groups (obrigatória): não é um rodapé válido
}

namespace {

// As estatísticas de uma coluna inteira (INT32/INT64 físico) num row group: mínimo, máximo e nulos, se o arquivo as
// gravou. Statistics: 1 max, 2 min (antigos, ordem com sinal: a mesma dos inteiros), 3 null_count, 5 max_value,
// 6 min_value (binários com o valor em little-endian).
struct KeyStats {
  i64 min = 0, max = 0, nulls = 0;
  bool has_min = false, has_max = false, has_nulls = false;
};

bool read_int(const u8* p, u64 len, int physical, i64* out) {
  if (physical == 1 && len == 4) { i32 v; std::memcpy(&v, p, 4); *out = v; return true; }
  if (physical == 2 && len == 8) { std::memcpy(out, p, 8); return true; }
  return false;
}

void read_statistics(Reader& r, int physical, KeyStats* s) {
  int last = 0, id, type;
  bool old_min = false, old_max = false;
  i64 omin = 0, omax = 0;
  while (next_field(r, &last, &id, &type)) {
    if (type == T_BINARY && (id == 1 || id == 2 || id == 5 || id == 6)) {
      const u64 len = r.varint();
      const u8* p = r.p;
      r.advance(len);
      if (!r.ok) return;
      i64 v;
      if (!read_int(p, len, physical, &v)) continue;
      if (id == 6) { s->min = v; s->has_min = true; }
      else if (id == 5) { s->max = v; s->has_max = true; }
      else if (id == 2) { omin = v; old_min = true; }
      else { omax = v; old_max = true; }
    } else if (id == 3 && type == T_I64) {
      s->nulls = r.zigzag();
      s->has_nulls = true;
    } else {
      r.skip(type, 1);
    }
  }
  if (!s->has_min && old_min) { s->min = omin; s->has_min = true; }
  if (!s->has_max && old_max) { s->max = omax; s->has_max = true; }
}

// ColumnMetaData: 1 type, 3 path_in_schema (lista de nomes), 12 statistics. `match` = o caminho é só `name`.
void read_column_meta(Reader& r, const u8* name, u64 name_len, bool* match, KeyStats* s) {
  int last = 0, id, type;
  int physical = -1;
  const u8* stats_at = nullptr;
  const u8* stats_end = nullptr;
  *match = false;
  while (next_field(r, &last, &id, &type)) {
    if (id == 1 && type == T_I32) {
      physical = (int)r.zigzag();
    } else if (id == 3 && type == T_LIST) {
      u64 size;
      int et;
      r.list_header(&size, &et);
      for (u64 i = 0; i < size && r.ok; i++) {
        const u64 len = r.varint();
        const u8* p = r.p;
        r.advance(len);
        if (size == 1 && len == name_len && std::memcmp(p, name, (size_t)len) == 0) *match = true;
      }
    } else if (id == 12 && type == T_STRUCT) {
      stats_at = r.p;   // o tipo físico (campo 1) vem antes, mas guardo a posição por garantia
      r.skip_struct(1);
      stats_end = r.p;
    } else {
      r.skip(type, 1);
    }
  }
  if (*match && stats_at != nullptr && r.ok) {
    Reader sr{stats_at, stats_end};
    read_statistics(sr, physical, s);
  }
}

// RowGroup: 1 columns (lista de ColumnChunk: 3 meta_data), 3 num_rows
i64 row_group_stats(Reader& r, const u8* name, u64 name_len, KeyStats* s) {
  i64 rows = -1;
  int last = 0, id, type;
  while (next_field(r, &last, &id, &type)) {
    if (id == 1 && type == T_LIST) {
      u64 size;
      int et;
      r.list_header(&size, &et);
      if (et != T_STRUCT) { r.ok = false; return -1; }
      for (u64 i = 0; i < size && r.ok; i++) {
        int clast = 0, cid, ctype;
        while (next_field(r, &clast, &cid, &ctype)) {
          if (cid == 3 && ctype == T_STRUCT) {
            bool match;
            KeyStats got;
            read_column_meta(r, name, name_len, &match, &got);
            if (match) *s = got;
          } else {
            r.skip(ctype, 1);
          }
        }
      }
    } else if (id == 3 && type == T_I64) {
      rows = r.zigzag();
    } else {
      r.skip(type, 1);
    }
  }
  return rows;
}

}  // namespace

// As estatísticas da coluna `name` (UTF-8, `name_len` bytes; uma coluna do nível de cima, inteira) em cada row group:
// out[5 * g ..] = linhas, mínimo, máximo, nulos e marcas (bit 0: tem mínimo e máximo; bit 1: tem a contagem de nulos).
// Até `cap` row groups; devolve quantos o arquivo tem, ou -1 se o rodapé não pôde ser lido. Uma coluna que não existe
// ou não é inteira sai sem marcas.
DC_EXPORT i64 dc_parquet_key_stats(const u8* footer, i64 n, const u8* name, i64 name_len, i64* out, i64 cap) {
  Reader r{footer, footer + n};
  i64 count = -1;
  int last = 0, id, type;
  while (next_field(r, &last, &id, &type)) {
    if (id == 4 && type == T_LIST) {
      u64 size;
      int et;
      r.list_header(&size, &et);
      if (et != T_STRUCT) return -1;
      count = 0;
      for (u64 i = 0; i < size && r.ok; i++) {
        KeyStats s;
        const i64 rows = row_group_stats(r, name, (u64)name_len, &s);
        if (rows < 0) return -1;
        if (count < cap) {
          i64* o = out + 5 * count;
          o[0] = rows;
          o[1] = s.min;
          o[2] = s.max;
          o[3] = s.nulls;
          o[4] = (s.has_min && s.has_max ? 1 : 0) | (s.has_nulls ? 2 : 0);
        }
        count++;
      }
    } else {
      r.skip(type, 1);
    }
  }
  return r.ok && count >= 0 ? count : -1;
}
