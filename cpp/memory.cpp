// memory.cpp -- pool dos buffers grandes (pares, hashes, bitmaps, colunas), que guarda a memória liberada para a
// próxima janela. Copiado do nini-datagen 0.3.0 (mesmas regras, prefixo dc_).
//
// Com o malloc do sistema, cada buffer grande pede páginas novas ao sistema operacional (entregues uma a uma na
// primeira escrita) e as devolve ao ser liberado: no Windows, ~39 ns/linha para receber e ~27 para devolver (medido no
// nini-datagen). Aqui os blocos liberados ficam guardados (até um limite) e a próxima alocação de tamanho parecido
// reaproveita um deles, com as páginas já entregues. Na comparação por janelas, a janela k+1 pede os mesmos tamanhos
// que a k: quase tudo sai do pool.
//
// Regras: tamanhos arredondados para múltiplos de 64 KiB; um bloco guardado serve a um pedido de até 25% menor (não
// desperdiça mais que isso); passando do limite de bytes guardados, devolve ao sistema os maiores primeiro. Seguro
// entre threads (um mutex; as alocações são de MB, poucas por janela). Alinhamento de 64 bytes (o do Arrow).
#include "common.hpp"
#include "memory.hpp"
#include <cstdlib>
#include <iterator>
#include <map>
#include <mutex>
#include <unordered_map>

namespace {

const i64 GRAIN = 64 * 1024;
const size_t ALIGN = 64;

void* raw_alloc(size_t n) {
#if defined(_WIN32)
  return _aligned_malloc(n, ALIGN);
#else
  void* p = nullptr;
  return posix_memalign(&p, ALIGN, n) == 0 ? p : nullptr;
#endif
}

void raw_free(void* p) {
#if defined(_WIN32)
  _aligned_free(p);
#else
  free(p);
#endif
}

struct Pool {
  std::mutex mu;
  std::multimap<i64, void*> cached;       // tamanho -> bloco guardado
  std::unordered_map<void*, i64> live;    // bloco em uso -> tamanho
  i64 cached_bytes = 0, live_bytes = 0, limit = (i64)1 << 30;   // 1 GiB guardado, no máximo
  i64 hits = 0, misses = 0;

  void shrink_to(i64 cap) {   // devolve ao sistema os maiores até caber (chamado com o mutex)
    while (cached_bytes > cap && !cached.empty()) {
      auto it = std::prev(cached.end());
      cached_bytes -= it->first;
      raw_free(it->second);
      cached.erase(it);
    }
  }
};

Pool& pool() {
  static Pool* p = new Pool();   // nunca destruído: blocos podem ser liberados depois do fim do programa
  return *p;
}

}  // namespace

namespace dc {

// Bloco de pelo menos nbytes (nullptr se faltou memória).
void* pool_alloc(i64 nbytes) {
  if (nbytes <= 0) nbytes = 1;
  const i64 size = (nbytes + GRAIN - 1) / GRAIN * GRAIN;
  Pool& P = pool();
  {
    std::lock_guard<std::mutex> lock(P.mu);
    auto it = P.cached.lower_bound(size);
    if (it != P.cached.end() && it->first <= size + size / 4) {
      void* p = it->second;
      const i64 got = it->first;
      P.cached.erase(it);
      P.cached_bytes -= got;
      P.live[p] = got;
      P.live_bytes += got;
      P.hits++;
      return p;
    }
    P.misses++;
  }
  void* p = raw_alloc((size_t)size);
  if (p == nullptr) {   // sem memória: devolve o que está guardado e tenta de novo
    {
      std::lock_guard<std::mutex> lock(P.mu);
      P.shrink_to(0);
    }
    p = raw_alloc((size_t)size);
    if (p == nullptr) return nullptr;
  }
  std::lock_guard<std::mutex> lock(P.mu);
  P.live[p] = size;
  P.live_bytes += size;
  return p;
}

// Devolve um bloco de pool_alloc ao pool (guardado se couber no limite). Endereço desconhecido: ignora.
void pool_free(void* p) {
  Pool& P = pool();
  std::lock_guard<std::mutex> lock(P.mu);
  auto it = P.live.find(p);
  if (it == P.live.end()) return;
  const i64 size = it->second;
  P.live.erase(it);
  P.live_bytes -= size;
  if (size > P.limit) { raw_free(p); return; }
  P.cached.emplace(size, p);
  P.cached_bytes += size;
  P.shrink_to(P.limit);
}

}  // namespace dc

// Bloco de pelo menos nbytes (endereço como inteiro; 0 se faltou memória).
DC_EXPORT i64 dc_pool_alloc(i64 nbytes) { return (i64)(uintptr_t)dc::pool_alloc(nbytes); }

// Devolve um bloco de dc_pool_alloc ao pool. Endereço desconhecido: ignora.
DC_EXPORT void dc_pool_free(i64 addr) { dc::pool_free((void*)(uintptr_t)addr); }

// Limite de bytes guardados (< 0: só consulta). 0 desliga a guarda (tudo volta ao sistema ao ser liberado).
DC_EXPORT i64 dc_pool_limit(i64 bytes) {
  Pool& P = pool();
  std::lock_guard<std::mutex> lock(P.mu);
  if (bytes >= 0) {
    P.limit = bytes;
    P.shrink_to(bytes);
  }
  return P.limit;
}

// Devolve ao sistema toda a memória guardada (os blocos em uso continuam válidos).
DC_EXPORT void dc_pool_trim() {
  Pool& P = pool();
  std::lock_guard<std::mutex> lock(P.mu);
  P.shrink_to(0);
}

// out[0..3]: bytes em uso, bytes guardados, alocações atendidas pelo pool, alocações novas.
DC_EXPORT void dc_pool_stats(i64* out) {
  Pool& P = pool();
  std::lock_guard<std::mutex> lock(P.mu);
  out[0] = P.live_bytes;
  out[1] = P.cached_bytes;
  out[2] = P.hits;
  out[3] = P.misses;
}
