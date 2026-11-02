// batch.cpp -- descritores das colunas de um lote do Arrow (um DataFrame exportado de uma vez), lidos no C++.
//
// Ler as colunas uma a uma pelo Python (`import_column`: uma exportação do Polars e uma dúzia de chamadas do ctypes por
// coluna) custava ~0,3 ms por coluna: numa janela de 300 colunas, mais que o próprio hash. Aqui o Python exporta o
// pedaço do DataFrame inteiro (um struct do Arrow com uma coluna por filho) e uma chamada só preenche os descritores
// (o formato de columns.hpp) de todas as colunas, sem copiar nada.
//
// Estruturas: https://arrow.apache.org/docs/format/CDataInterface.html (layout fixo, igual em toda plataforma).
#include "columns.hpp"

namespace {

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

bool starts(const char* s, const char* prefix) { return std::strncmp(s, prefix, std::strlen(prefix)) == 0; }

bool eq(const char* s, const char* t) { return std::strcmp(s, t) == 0; }

// (tipo, largura) pelo formato do Arrow; tipo 0 = não suportado. Os mesmos de core/keys.py (key_kind) e de
// core/arrow.py (value_width).
void kind_of(const char* f, i64* kind, i64* width) {
  *kind = 0;
  *width = 0;
  struct Fixed { const char* fmt; i64 kind, width; };
  static const Fixed fixed[] = {
      {"c", SIGNED, 1}, {"s", SIGNED, 2}, {"i", SIGNED, 4}, {"l", SIGNED, 8},
      {"C", UNSIGNED, 1}, {"S", UNSIGNED, 2}, {"I", UNSIGNED, 4}, {"L", UNSIGNED, 8},
      {"f", FLOAT, 4}, {"g", FLOAT, 8},
      {"tdD", SIGNED, 4}, {"tdm", SIGNED, 8}, {"tts", SIGNED, 4}, {"ttm", SIGNED, 4}, {"ttu", SIGNED, 8},
      {"ttn", SIGNED, 8}, {"b", BOOL, 0}, {"vu", TEXT, 16}, {"vz", TEXT, 16},
  };
  for (const Fixed& x : fixed) {
    if (eq(f, x.fmt)) {
      *kind = x.kind;
      *width = x.width;
      return;
    }
  }
  if (starts(f, "ts") || starts(f, "tD")) {   // Datetime ("tsu:UTC") e Duration ("tDu"): int64
    *kind = SIGNED;
    *width = 8;
  } else if (starts(f, "d:")) {   // decimal128 "d:precisão,escala"; decimal256 traz ",256" no fim
    const char* last = std::strrchr(f, ',');
    const char* first = std::strchr(f, ',');
    *kind = BYTES;
    *width = (last && last != first && eq(last, ",256")) ? 32 : 16;
  }
}

}  // namespace

// Preenche `desc` (k x DESC inteiros) com as k colunas do lote: `schema`/`array` são o struct exportado (um filho por
// coluna). Devolve 0; -1 se o lote não é um struct de k filhos sem deslocamento; -(2 + i) se a coluna i não é
// suportada (dicionário, tipo ou buffers inesperados): o Python cai na leitura coluna a coluna.
DC_EXPORT int dc_describe_batch(const u8* schema, const u8* array, int k, i64* desc) {
  const ArrowSchema* s = (const ArrowSchema*)schema;
  const ArrowArray* a = (const ArrowArray*)array;
  if (!s || !a || !eq(s->format, "+s") || s->n_children != k || a->n_children != k || a->offset != 0) return -1;
  for (int c = 0; c < k; c++) {
    const ArrowSchema* cs = s->children[c];
    const ArrowArray* ca = a->children[c];
    i64* d = desc + (i64)c * DESC;
    i64 kind, width;
    kind_of(cs->format, &kind, &width);
    const i64 need = kind == TEXT ? 3 : 2;   // validade, valores (views) [, buffers de texto..., tamanhos]
    if (kind == 0 || cs->dictionary != nullptr || ca->n_buffers < need || ca->length < a->length) return -(2 + c);
    const i64 off = ca->offset;
    const u8* values = (const u8*)ca->buffers[1];
    d[0] = kind;
    d[1] = width;
    d[2] = (i64)(uintptr_t)(values == nullptr ? nullptr : (kind == BOOL ? values : values + width * off));
    d[3] = (i64)(uintptr_t)ca->buffers[0];   // pode ser 0: sem nulos
    d[4] = off;
    // texto: os buffers de dados são buffers[2 .. n_buffers-2] (o último guarda os tamanhos), já em sequência
    d[5] = kind == TEXT && ca->n_buffers > 3 ? (i64)(uintptr_t)(ca->buffers + 2) : 0;
  }
  return 0;
}
