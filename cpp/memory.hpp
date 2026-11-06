// memory.hpp -- o pool de memória (memory.cpp) para o C++ de dentro da biblioteca; o Python usa dc_pool_alloc/free.
#pragma once
#include "common.hpp"

namespace dc {

// Bloco de pelo menos `nbytes`, alinhado a 64 bytes (nullptr se faltou memória). Seguro entre threads.
void* pool_alloc(i64 nbytes);
// Devolve ao pool um bloco de pool_alloc (endereço desconhecido: ignora).
void pool_free(void* p);

}  // namespace dc
