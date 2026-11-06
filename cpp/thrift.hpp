// thrift.hpp -- leitor do protocolo compacto do Thrift, compartilhado por parquet_meta.cpp (rodapé: linhas e
// estatísticas) e parquet_read.cpp (rodapé e cabeçalhos de página). Só lê e pula; nada é interpretado aqui.
//
// Protocolo compacto do Thrift: cabeçalho de campo = 1 byte (delta do id nos 4 bits altos, tipo nos 4 baixos; delta 0
// = id em varint zigzag a seguir); inteiros em varint zigzag; binário = tamanho em varint + bytes; lista = 1 byte
// (tamanho nos 4 bits altos, 15 = tamanho em varint a seguir; tipo do elemento nos baixos); mapa = tamanho em varint e,
// se > 0, 1 byte com os tipos de chave e valor; struct = campos até o byte 0. Booleanos de campo vêm no próprio tipo
// (1 = true, 2 = false); dentro de listas, 1 byte por elemento.
#pragma once
#include "common.hpp"

namespace thrift {

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

inline void Reader::skip(int type, int depth) {
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
inline bool next_field(Reader& r, int* last, int* id, int* type) {
  u8 h = r.byte();
  if (!r.ok || h == 0) return false;
  *type = h & 0x0F;
  int delta = h >> 4;
  *id = delta ? *last + delta : (int)r.zigzag();
  *last = *id;
  return r.ok;
}

}  // namespace thrift
