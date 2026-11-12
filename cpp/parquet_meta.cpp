// parquet_meta.cpp -- linhas de cada row group de um arquivo Parquet, lidas do rodapé (Thrift, protocolo compacto).
//
// O planejamento das janelas (src/nini_datacomp/windows.py) alinha as fronteiras ao começo dos row groups, para nenhum
// row group ser decodificado duas vezes. O Polars não expõe os row groups, e o pacote não depende do pyarrow: este
// leitor percorre o FileMetaData (https://github.com/apache/parquet-format, parquet.thrift) e só guarda o num_rows de
// cada RowGroup (campo 4 do FileMetaData, campo 3 do RowGroup); o resto é pulado sem ser interpretado.
// O protocolo compacto do Thrift (leitura e salto de campos) está em thrift.hpp.
#include "common.hpp"
#include "thrift.hpp"

namespace {

using namespace thrift;

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

namespace {

// As estatísticas de uma coluna de texto (BYTE_ARRAY físico) num row group. O mínimo e o máximo ficam como posição e
// tamanho em bytes dentro do rodapé (nada é copiado). Só os campos 5 (max_value) e 6 (min_value): os antigos 1 e 2 não
// valem para texto (a ordem deles era indefinida). Os campos de exatidão (7/8) são ignorados: um mínimo truncado vira
// um piso e um máximo truncado um teto, o que só deixa o teste "em ordem" mais conservador.
struct TextStats {
  i64 min_pos = 0, min_len = 0, max_pos = 0, max_len = 0, nulls = 0;
  bool has_min = false, has_max = false, has_nulls = false;
};

void read_text_statistics(Reader& r, const u8* base, TextStats* s) {
  int last = 0, id, type;
  while (next_field(r, &last, &id, &type)) {
    if (type == T_BINARY && (id == 5 || id == 6)) {
      const u64 len = r.varint();
      const u8* p = r.p;
      r.advance(len);
      if (!r.ok) return;
      if (id == 6) { s->min_pos = p - base; s->min_len = (i64)len; s->has_min = true; }
      else { s->max_pos = p - base; s->max_len = (i64)len; s->has_max = true; }
    } else if (id == 3 && type == T_I64) {
      s->nulls = r.zigzag();
      s->has_nulls = true;
    } else {
      r.skip(type, 1);
    }
  }
}

void read_text_column_meta(Reader& r, const u8* base, const u8* name, u64 name_len, bool* match, TextStats* s) {
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
      stats_at = r.p;
      r.skip_struct(1);
      stats_end = r.p;
    } else {
      r.skip(type, 1);
    }
  }
  if (*match && physical == 6 && stats_at != nullptr && r.ok) {
    Reader sr{stats_at, stats_end};
    read_text_statistics(sr, base, s);
  }
}

i64 row_group_text_stats(Reader& r, const u8* base, const u8* name, u64 name_len, TextStats* s) {
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
            TextStats got;
            read_text_column_meta(r, base, name, name_len, &match, &got);
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

// Como `dc_parquet_key_stats`, para uma coluna de texto (BYTE_ARRAY): out[7 * g ..] = linhas, posição e tamanho do
// mínimo, posição e tamanho do máximo (posições em bytes dentro de `footer`), nulos e marcas (bit 0: tem mínimo e
// máximo; bit 1: tem a contagem de nulos). Até `cap` row groups; devolve quantos o arquivo tem, ou -1 se o rodapé não
// pôde ser lido. Uma coluna que não existe ou não é de texto sai sem marcas.
DC_EXPORT i64 dc_parquet_text_stats(const u8* footer, i64 n, const u8* name, i64 name_len, i64* out, i64 cap) {
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
        TextStats s;
        const i64 rows = row_group_text_stats(r, footer, name, (u64)name_len, &s);
        if (rows < 0) return -1;
        if (count < cap) {
          i64* o = out + 7 * count;
          o[0] = rows;
          o[1] = s.min_pos;
          o[2] = s.min_len;
          o[3] = s.max_pos;
          o[4] = s.max_len;
          o[5] = s.nulls;
          o[6] = (s.has_min && s.has_max ? 1 : 0) | (s.has_nulls ? 2 : 0);
        }
        count++;
      }
    } else {
      r.skip(type, 1);
    }
  }
  return r.ok && count >= 0 ? count : -1;
}
