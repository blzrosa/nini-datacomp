// pool.hpp -- pool de threads do C++: uma chamada do Python roda todas as tarefas de um kernel.
//
// Antes, cada bloco de linhas era uma tarefa do ThreadPoolExecutor do Python, e cada tarefa custa dezenas de
// microssegundos de fila e de GIL (em 10M x 10, ~1.300 tarefas por comparação só no hash). Aqui o Python faz uma
// chamada só por kernel (o ctypes solta o GIL durante ela), e as tarefas se dividem entre threads do C++, que ficam
// vivas entre as chamadas. A thread que chamou também trabalha.
//
// Uma chamada por vez: chamadas de threads diferentes do Python esperam a vez (os kernels são chamados da thread
// principal; isto só evita misturar duas). Uma tarefa que chama parallel_for de novo roda em série.
#pragma once
#include "common.hpp"

#include <functional>

namespace dc {

// fn(t) para t em [0, tasks), em até `threads` threads (a que chamou inclusive). Tarefas pequenas, muitas: o
// equilíbrio vem de cada thread pegar a próxima tarefa livre.
void parallel_for(int threads, i64 tasks, const std::function<void(i64)>& fn);

// linhas por tarefa para n linhas em `threads` threads: ~4 tarefas por thread (equilíbrio), entre 4.096 e 65.536
// linhas (o custo fixo de uma tarefa contra o cache)
i64 task_rows(i64 n, int threads);

}  // namespace dc
