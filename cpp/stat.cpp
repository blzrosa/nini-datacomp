// stat.cpp -- leitura de arquivos de programas estatísticos pelo ReadStat (cpp/third_party/readstat, licença MIT):
// SAS (.sas7bdat, .xpt), Stata (.dta) e SPSS (.sav, .zsav, .por).
//
// O ReadStat lê o arquivo do começo ao fim chamando uma função por célula. Uma thread lê (produtora) e monta pedaços
// de linhas em colunas no formato do Arrow (valores, bitmap de validade e, no texto, offsets de 64 bits); o Python
// pega um pedaço de cada vez (dc_stat_next), copia as colunas e devolve (dc_stat_release). Assim o arquivo é lido uma
// vez só, a leitura do próximo pedaço anda enquanto o Python grava o anterior, e a memória fica em ~2 pedaços.
//
// Uso (src/nini_datacomp/core/stat.py):
//   h = dc_stat_open(...)       abre e começa a ler (em outra thread)
//   dc_stat_schema(h)           espera as colunas: quantas (ou -1 se o arquivo não abriu)
//   dc_stat_column(h, i, ...)   nome, tipo e formato de exibição da coluna i
//   dc_stat_next(h)             espera o próximo pedaço: linhas (0 = acabou, -1 = erro)
//   dc_stat_buffers(h, c, out)  os buffers da coluna c no pedaço atual
//   dc_stat_release(h)          o Python já copiou: a thread pode entregar o próximo
//   dc_stat_error(h, ...)       a mensagem de erro, se houve
//   dc_stat_close(h)            interrompe a leitura (se ainda não acabou) e libera tudo
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include "common.hpp"

extern "C" {
#include "third_party/readstat/readstat.h"
}
#include "third_party/shim/iconv.h"

namespace {

// Formatos (o argumento `format` de dc_stat_open)
enum Format { F_SAS7BDAT = 0, F_XPORT = 1, F_DTA = 2, F_SAV = 3, F_POR = 4 };
// Tipos das colunas devolvidas (o Python escolhe o tipo do Polars por eles)
enum Kind { K_STRING = 0, K_I8 = 1, K_I16 = 2, K_I32 = 3, K_F32 = 4, K_F64 = 5 };
constexpr int FLAG_BLANK_NULL = 1;  // texto vazio (ou só espaços) vira nulo

int width_of(int kind) {
  switch (kind) {
    case K_I8: return 1;
    case K_I16: return 2;
    case K_I32: case K_F32: return 4;
    case K_F64: return 8;
    default: return 0;
  }
}

int kind_of(readstat_type_t t) {
  switch (t) {
    case READSTAT_TYPE_STRING: case READSTAT_TYPE_STRING_REF: return K_STRING;
    case READSTAT_TYPE_INT8: return K_I8;
    case READSTAT_TYPE_INT16: return K_I16;
    case READSTAT_TYPE_INT32: return K_I32;
    case READSTAT_TYPE_FLOAT: return K_F32;
    default: return K_F64;
  }
}

struct ColumnInfo {
  std::string name, format;
  int kind = K_F64;
};

// Uma coluna de um pedaço, no formato do Arrow
struct Buffer {
  std::vector<u8> data;     // valores de largura fixa, ou os bytes do texto
  std::vector<u8> valid;    // bitmap de validade (LSB primeiro)
  std::vector<i64> offsets; // só texto: rows + 1 posições em `data`
  i64 rows = 0, nulls = 0;
};

struct Chunk {
  std::vector<Buffer> cols;
  i64 rows = 0, bytes = 0;

  void reset(const std::vector<ColumnInfo>& info) {
    cols.assign(info.size(), Buffer());
    for (size_t c = 0; c < info.size(); c++)
      if (info[c].kind == K_STRING) cols[c].offsets.push_back(0);
    rows = bytes = 0;
  }
};

void push_valid(Buffer& b, bool ok) {
  if ((b.rows & 7) == 0) b.valid.push_back(0);
  if (ok) b.valid.back() |= (u8)(1u << (b.rows & 7));
  else b.nulls++;
  b.rows++;
}

void push_null(Buffer& b, int kind) {
  if (kind == K_STRING) b.offsets.push_back((i64)b.data.size());
  else b.data.resize(b.data.size() + width_of(kind), 0);
  push_valid(b, false);
}

// Arquivo aberto pelas funções de E/S que o ReadStat chama (caminho em UTF-8 também no Windows)
struct Io {
  FILE* f = nullptr;
  std::atomic<bool>* abort = nullptr;
};

struct State {
  // entrada
  std::string path, encoding;
  int format = 0, flags = 0;
  i64 chunk_bytes = 0, max_rows = 0;

  // lido pela thread produtora; publicado (com `mu`) quando `schema_ready`
  std::vector<ColumnInfo> columns;
  i64 row_count = -1;  // do cabeçalho; -1 = o formato não diz
  Chunk building;
  i64 obs = -1;        // linha (do arquivo) que está sendo montada; -1 = nenhuma
  Io io;

  // troca entre as threads
  std::mutex mu;
  std::condition_variable cv;
  bool schema_ready = false, done = false, failed = false;
  bool has_ready = false;   // `ready` tem um pedaço que o Python ainda não pegou
  bool taken = false;       // o Python pegou `ready` (dc_stat_next) e ainda não devolveu (dc_stat_release)
  Chunk ready;
  std::string error;
  std::atomic<bool> abort{false};
  std::thread worker;

  ~State() {
    abort = true;
    {
      std::lock_guard<std::mutex> lock(mu);
    }
    cv.notify_all();
    if (worker.joinable()) worker.join();
  }

  void fail(const std::string& msg) {
    std::lock_guard<std::mutex> lock(mu);
    if (!failed) {
      failed = true;
      error = msg;
    }
  }

  void publish_schema() {
    std::lock_guard<std::mutex> lock(mu);
    if (!schema_ready) {
      schema_ready = true;
      cv.notify_all();
    }
  }

  // Entrega `building` ao Python (espera ele terminar o anterior). false: interrompido.
  bool hand_off() {
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [&] { return (!has_ready && !taken) || abort.load(); });
    if (abort) return false;
    ready = std::move(building);
    has_ready = true;
    cv.notify_all();
    lock.unlock();
    building.reset(columns);
    return true;
  }

  // A linha `obs` terminou: completa com nulo as colunas que não receberam valor e entrega o pedaço se encheu.
  bool finish_row() {
    if (obs < 0) return true;
    for (size_t c = 0; c < columns.size(); c++) {
      Buffer& b = building.cols[c];
      if (b.rows <= building.rows) push_null(b, columns[c].kind);
    }
    building.rows++;
    obs = -1;
    if (building.bytes >= chunk_bytes || building.rows >= max_rows) return hand_off();
    return true;
  }
};

// ----------------------------------------------------------------------------- E/S
int io_open(const char* path, void* ctx) {
  Io* io = static_cast<Io*>(ctx);
#if defined(_WIN32)
  int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, nullptr, 0);
  if (n <= 0) return -1;
  std::wstring wide((size_t)n, L'\0');
  MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, &wide[0], n);
  io->f = _wfopen(wide.c_str(), L"rb");
#else
  io->f = std::fopen(path, "rb");
#endif
  if (!io->f) return -1;
  std::setvbuf(io->f, nullptr, _IOFBF, 1 << 20);
  return 0;
}

int io_close(void* ctx) {
  Io* io = static_cast<Io*>(ctx);
  int rc = 0;
  if (io->f) rc = std::fclose(io->f);
  io->f = nullptr;
  return rc;
}

readstat_off_t io_seek(readstat_off_t offset, readstat_io_flags_t whence, void* ctx) {
  Io* io = static_cast<Io*>(ctx);
  int w = whence == READSTAT_SEEK_SET ? SEEK_SET : whence == READSTAT_SEEK_CUR ? SEEK_CUR : SEEK_END;
#if defined(_WIN32)
  if (_fseeki64(io->f, (__int64)offset, w) != 0) return -1;
  return (readstat_off_t)_ftelli64(io->f);
#else
  if (fseeko(io->f, (off_t)offset, w) != 0) return -1;
  return (readstat_off_t)ftello(io->f);
#endif
}

ssize_t io_read(void* buf, size_t nbyte, void* ctx) {
  Io* io = static_cast<Io*>(ctx);
  size_t got = std::fread(buf, 1, nbyte, io->f);
  if (got < nbyte && std::ferror(io->f)) return -1;
  return (ssize_t)got;
}

readstat_error_t io_update(long, readstat_progress_handler, void*, void* ctx) {
  Io* io = static_cast<Io*>(ctx);
  return io->abort->load() ? READSTAT_ERROR_USER_ABORT : READSTAT_OK;
}

// ----------------------------------------------------------------------------- callbacks do ReadStat
int on_metadata(readstat_metadata_t* metadata, void* ctx) {
  State* s = static_cast<State*>(ctx);
  s->row_count = readstat_get_row_count(metadata);
  int vars = readstat_get_var_count(metadata);
  if (vars > 0) s->columns.reserve((size_t)vars);
  return s->abort ? READSTAT_HANDLER_ABORT : READSTAT_HANDLER_OK;
}

int on_variable(int index, readstat_variable_t* variable, const char*, void* ctx) {
  State* s = static_cast<State*>(ctx);
  if (index != (int)s->columns.size()) {
    s->fail("ReadStat entregou as colunas fora de ordem");
    return READSTAT_HANDLER_ABORT;
  }
  ColumnInfo info;
  const char* name = readstat_variable_get_name(variable);
  const char* fmt = readstat_variable_get_format(variable);
  info.name = name ? name : "";
  info.format = fmt ? fmt : "";
  info.kind = kind_of(readstat_variable_get_type(variable));
  s->columns.push_back(std::move(info));
  return s->abort ? READSTAT_HANDLER_ABORT : READSTAT_HANDLER_OK;
}

// O texto é UTF-8 válido? (o Polars exige; um arquivo sem a codificação certa entrega os bytes como estão)
bool valid_utf8(const unsigned char* s, size_t n) {
  size_t i = 0;
  while (i < n) {
    unsigned char c = s[i];
    if (c < 0x80) { i++; continue; }
    size_t len;
    u32 cp;
    if (c >= 0xC2 && c <= 0xDF) len = 2, cp = c & 0x1F;
    else if (c >= 0xE0 && c <= 0xEF) len = 3, cp = c & 0x0F;
    else if (c >= 0xF0 && c <= 0xF4) len = 4, cp = c & 0x07;
    else return false;
    if (i + len > n) return false;
    for (size_t k = 1; k < len; k++) {
      if ((s[i + k] & 0xC0) != 0x80) return false;
      cp = (cp << 6) | (s[i + k] & 0x3F);
    }
    if ((len == 3 && cp < 0x800) || (len == 4 && (cp < 0x10000 || cp > 0x10FFFF)) || (cp >= 0xD800 && cp <= 0xDFFF))
      return false;
    i += len;
  }
  return true;
}

bool blank(const char* str) {
  if (!str) return true;
  for (; *str; str++)
    if (*str != ' ' && *str != '\t' && *str != '\r' && *str != '\n') return false;
  return true;
}

int on_value(int obs_index, readstat_variable_t* variable, readstat_value_t value, void* ctx) {
  State* s = static_cast<State*>(ctx);
  if (s->abort) return READSTAT_HANDLER_ABORT;
  if (!s->schema_ready) {
    s->building.reset(s->columns);
    s->publish_schema();
  }
  if ((i64)obs_index != s->obs) {
    if (!s->finish_row()) return READSTAT_HANDLER_ABORT;
    s->obs = obs_index;
  }
  int c = readstat_variable_get_index_after_skipping(variable);
  if (c < 0 || c >= (int)s->columns.size()) return READSTAT_HANDLER_OK;
  Buffer& b = s->building.cols[(size_t)c];
  int kind = s->columns[(size_t)c].kind;
  // uma coluna que pulou linhas (não deveria acontecer) fica com nulo nelas
  while (b.rows < s->building.rows) push_null(b, kind);
  if (b.rows > s->building.rows) return READSTAT_HANDLER_OK;  // valor repetido na mesma linha: fica o primeiro

  if (kind == K_STRING) {
    const char* str = readstat_string_value(value);
    if (str == nullptr || ((s->flags & FLAG_BLANK_NULL) && blank(str))) {
      push_null(b, kind);
    } else {
      size_t n = std::strlen(str);
      if (!valid_utf8(reinterpret_cast<const unsigned char*>(str), n)) {
        s->fail("o texto da coluna '" + s->columns[(size_t)c].name + "' (linha " + std::to_string(obs_index + 1) +
                ") não é UTF-8 válido: o arquivo não declara a codificação, ou declara errado (informe encoding=, "
                "por exemplo 'latin1' ou 'cp1252')");
        return READSTAT_HANDLER_ABORT;
      }
      b.data.insert(b.data.end(), str, str + n);
      b.offsets.push_back((i64)b.data.size());
      push_valid(b, true);
      s->building.bytes += (i64)n + 8;
    }
    return READSTAT_HANDLER_OK;
  }
  if (readstat_value_is_missing(value, variable)) {
    push_null(b, kind);
    s->building.bytes += width_of(kind);
    return READSTAT_HANDLER_OK;
  }
  size_t at = b.data.size();
  b.data.resize(at + (size_t)width_of(kind));
  u8* p = b.data.data() + at;
  switch (kind) {
    case K_I8: { int8_t v = (int8_t)readstat_int8_value(value); std::memcpy(p, &v, 1); break; }
    case K_I16: { int16_t v = readstat_int16_value(value); std::memcpy(p, &v, 2); break; }
    case K_I32: { int32_t v = readstat_int32_value(value); std::memcpy(p, &v, 4); break; }
    case K_F32: { float v = readstat_float_value(value); std::memcpy(p, &v, 4); break; }
    default: { double v = readstat_double_value(value); std::memcpy(p, &v, 8); break; }
  }
  push_valid(b, true);
  s->building.bytes += width_of(kind);
  return READSTAT_HANDLER_OK;
}

void on_error(const char* message, void* ctx) {
  State* s = static_cast<State*>(ctx);
  std::string msg = message ? message : "";
  while (!msg.empty() && (msg.back() == '\n' || msg.back() == ' ')) msg.pop_back();
  s->fail(msg);
}

// ----------------------------------------------------------------------------- a thread produtora
void run(State* s) {
  readstat_parser_t* parser = nullptr;
  try {
    parser = readstat_parser_init();
    if (!parser) throw std::bad_alloc();
    s->io.abort = &s->abort;
    readstat_set_metadata_handler(parser, on_metadata);
    readstat_set_variable_handler(parser, on_variable);
    readstat_set_value_handler(parser, on_value);
    readstat_set_error_handler(parser, on_error);
    readstat_set_open_handler(parser, io_open);
    readstat_set_close_handler(parser, io_close);
    readstat_set_seek_handler(parser, io_seek);
    readstat_set_read_handler(parser, io_read);
    readstat_set_update_handler(parser, io_update);
    readstat_set_io_ctx(parser, &s->io);
    readstat_set_handler_character_encoding(parser, "UTF-8");
    if (!s->encoding.empty()) readstat_set_file_character_encoding(parser, s->encoding.c_str());

    readstat_error_t rc = READSTAT_OK;
    const char* path = s->path.c_str();
    switch (s->format) {
      case F_SAS7BDAT: rc = readstat_parse_sas7bdat(parser, path, s); break;
      case F_XPORT: rc = readstat_parse_xport(parser, path, s); break;
      case F_DTA: rc = readstat_parse_dta(parser, path, s); break;
      case F_SAV: rc = readstat_parse_sav(parser, path, s); break;
      case F_POR: rc = readstat_parse_por(parser, path, s); break;
      default: rc = READSTAT_ERROR_UNSUPPORTED_FILE_FORMAT_VERSION; break;
    }
    if (s->io.f) io_close(&s->io);  // o ReadStat fecha; isto é só para o caso de ele ter parado antes
    readstat_parser_free(parser);
    parser = nullptr;

    if (rc != READSTAT_OK && !s->abort) {
      std::string msg = readstat_error_message(rc);
      std::lock_guard<std::mutex> lock(s->mu);
      if (!s->failed) {
        s->failed = true;
        s->error = msg;
      } else if (rc != READSTAT_ERROR_USER_ABORT && s->error.find(msg) == std::string::npos) {
        s->error = msg + ": " + s->error;
      }
    }
    bool ok;
    {
      std::lock_guard<std::mutex> lock(s->mu);
      ok = !s->failed && !s->abort;
    }
    if (ok) {
      if (!s->schema_ready) {  // sem nenhuma linha: as colunas só ficam prontas no fim
        s->building.reset(s->columns);
      }
      if (s->obs >= 0) {
        // a última linha: completa e entrega o que sobrou
        for (size_t c = 0; c < s->columns.size(); c++) {
          Buffer& b = s->building.cols[c];
          if (b.rows <= s->building.rows) push_null(b, s->columns[c].kind);
        }
        s->building.rows++;
        s->obs = -1;
      }
      s->publish_schema();
      if (s->building.rows > 0) s->hand_off();
    }
  } catch (const std::bad_alloc&) {
    if (parser) readstat_parser_free(parser);
    s->fail("memória insuficiente ao ler o arquivo");
  } catch (...) {
    if (parser) readstat_parser_free(parser);
    s->fail("erro inesperado ao ler o arquivo");
  }
  std::lock_guard<std::mutex> lock(s->mu);
  s->done = true;
  s->schema_ready = s->schema_ready || s->failed;
  s->cv.notify_all();
}

// ----------------------------------------------------------------------------- registro de leituras abertas
std::mutex g_mu;
std::unordered_map<i64, std::shared_ptr<State>> g_open;
i64 g_next = 1;

std::shared_ptr<State> find(i64 h) {
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_open.find(h);
  return it == g_open.end() ? nullptr : it->second;
}

i64 copy_out(const std::string& s, u8* buf, i64 cap) {
  i64 n = (i64)s.size();
  if (buf && cap > 0) {
    i64 m = n < cap - 1 ? n : cap - 1;
    std::memcpy(buf, s.data(), (size_t)m);
    buf[m] = 0;
  }
  return n;
}

}  // namespace

// Abre `path` (UTF-8, `path_len` bytes) e começa a ler numa thread. `encoding` (`enc_len` bytes; 0 = a do arquivo)
// troca a codificação do texto que o arquivo declara. Um pedaço é entregue a cada `chunk_bytes` bytes (aproximados,
// em memória) ou `max_rows` linhas. `flags`: bit 0 = texto vazio vira nulo. Devolve o identificador (> 0), ou -1.
DC_EXPORT i64 dc_stat_open(const u8* path, i64 path_len, int format, const u8* encoding, i64 enc_len, i64 chunk_bytes,
                           i64 max_rows, int flags) {
  try {
    auto s = std::make_shared<State>();
    s->path.assign(reinterpret_cast<const char*>(path), (size_t)path_len);
    if (enc_len > 0) s->encoding.assign(reinterpret_cast<const char*>(encoding), (size_t)enc_len);
    s->format = format;
    s->flags = flags;
    s->chunk_bytes = chunk_bytes > 0 ? chunk_bytes : (i64)64 << 20;
    s->max_rows = max_rows > 0 ? max_rows : (i64)1 << 40;
    s->worker = std::thread(run, s.get());
    std::lock_guard<std::mutex> lock(g_mu);
    i64 h = g_next++;
    g_open[h] = s;
    return h;
  } catch (...) {
    return -1;
  }
}

// Espera as colunas: quantas são (>= 0), ou -1 se o arquivo não abriu (a mensagem em dc_stat_error).
DC_EXPORT int dc_stat_schema(i64 h) {
  auto s = find(h);
  if (!s) return -1;
  std::unique_lock<std::mutex> lock(s->mu);
  s->cv.wait(lock, [&] { return s->schema_ready || s->done; });
  if (s->failed && s->columns.empty()) return -1;
  return (int)s->columns.size();
}

// Linhas que o cabeçalho do arquivo declara (-1: o formato não diz). Só depois de dc_stat_schema.
DC_EXPORT i64 dc_stat_row_count(i64 h) {
  auto s = find(h);
  return s ? s->row_count : -1;
}

// A coluna `i`: nome e formato de exibição (UTF-8, terminados em 0) em `name`/`fmt` (com `cap` bytes cada) e, em
// `info`, [tipo (Kind), bytes do nome, bytes do formato]. Só depois de dc_stat_schema. 0, ou -1 se `i` não existe.
DC_EXPORT int dc_stat_column(i64 h, int i, u8* name, i64 name_cap, u8* fmt, i64 fmt_cap, i64* info) {
  auto s = find(h);
  if (!s || i < 0 || (size_t)i >= s->columns.size()) return -1;
  const ColumnInfo& c = s->columns[(size_t)i];
  info[0] = c.kind;
  info[1] = copy_out(c.name, name, name_cap);
  info[2] = copy_out(c.format, fmt, fmt_cap);
  return 0;
}

// Espera o próximo pedaço: as linhas dele (> 0), 0 quando o arquivo acabou, -1 se houve erro. O pedaço fica
// disponível em dc_stat_buffers até dc_stat_release.
DC_EXPORT i64 dc_stat_next(i64 h) {
  auto s = find(h);
  if (!s) return -1;
  std::unique_lock<std::mutex> lock(s->mu);
  if (s->taken) {  // o anterior não foi devolvido: devolve agora
    s->taken = false;
    s->ready = Chunk();
    s->cv.notify_all();
  }
  s->cv.wait(lock, [&] { return s->has_ready || s->done; });
  if (s->has_ready) {
    s->has_ready = false;
    s->taken = true;
    return s->ready.rows;
  }
  return s->failed ? -1 : 0;
}

// Os buffers da coluna `col` no pedaço atual, em `out`: [endereço dos valores (ou dos bytes do texto), bytes,
// endereço do bitmap de validade, endereço dos offsets (texto; 0 nas outras), nulos]. 0, ou -1.
DC_EXPORT int dc_stat_buffers(i64 h, int col, i64* out) {
  auto s = find(h);
  if (!s) return -1;
  std::lock_guard<std::mutex> lock(s->mu);
  if (!s->taken || col < 0 || (size_t)col >= s->ready.cols.size()) return -1;
  Buffer& b = s->ready.cols[(size_t)col];
  out[0] = (i64)(intptr_t)b.data.data();
  out[1] = (i64)b.data.size();
  out[2] = (i64)(intptr_t)b.valid.data();
  out[3] = b.offsets.empty() ? 0 : (i64)(intptr_t)b.offsets.data();
  out[4] = b.nulls;
  return 0;
}

// O Python terminou de copiar o pedaço atual: a thread pode entregar o próximo.
DC_EXPORT void dc_stat_release(i64 h) {
  auto s = find(h);
  if (!s) return;
  std::lock_guard<std::mutex> lock(s->mu);
  if (s->taken) {
    s->taken = false;
    s->ready = Chunk();
    s->cv.notify_all();
  }
}

// A mensagem de erro (UTF-8, terminada em 0) em `buf` com `cap` bytes; devolve o tamanho dela (0: sem erro).
DC_EXPORT i64 dc_stat_error(i64 h, u8* buf, i64 cap) {
  auto s = find(h);
  if (!s) return copy_out("leitura fechada ou inexistente", buf, cap);
  std::lock_guard<std::mutex> lock(s->mu);
  return copy_out(s->failed ? s->error : std::string(), buf, cap);
}

// Interrompe a leitura (se ainda não acabou), espera a thread e libera tudo.
DC_EXPORT void dc_stat_close(i64 h) {
  std::shared_ptr<State> s;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_open.find(h);
    if (it == g_open.end()) return;
    s = it->second;
    g_open.erase(it);
  }
  s->abort = true;
  {
    std::lock_guard<std::mutex> lock(s->mu);
    s->cv.notify_all();
  }
  if (s->worker.joinable()) s->worker.join();
}

// A codificação `name` (`len` bytes) é uma das que o leitor converte? 1 = sim, 0 = não.
DC_EXPORT int dc_stat_encoding_supported(const u8* name, i64 len) {
  std::string n(reinterpret_cast<const char*>(name), (size_t)len);
  return dc_iconv_supported(n.c_str());
}
