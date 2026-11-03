// pool.cpp -- o pool de threads (veja pool.hpp).
#include "pool.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace dc {
namespace {

thread_local bool inside_task = false;

class Pool {
 public:
  void run(int threads, i64 tasks, const std::function<void(i64)>& fn) {
    std::lock_guard<std::mutex> call(call_mutex_);   // uma chamada por vez
    grow(threads - 1);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      fn_ = &fn;
      tasks_ = tasks;
      next_.store(0);
      helpers_ = std::min<int>(threads - 1, (int)workers_.size());
      busy_ = helpers_;
      generation_++;
    }
    wake_.notify_all();
    work();
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [&] { return busy_ == 0; });
    fn_ = nullptr;
  }

 private:
  void grow(int want) {
    while ((int)workers_.size() < want) {
      const int index = (int)workers_.size();
      workers_.emplace_back([this, index] { loop(index); });
      workers_.back().detach();   // nunca são juntadas: vivem até o fim do processo (sem travar na saída)
    }
  }

  void work() {
    inside_task = true;
    for (i64 t; (t = next_.fetch_add(1)) < tasks_;) (*fn_)(t);
    inside_task = false;
  }

  void loop(int index) {
    u64 seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        wake_.wait(lock, [&] { return generation_ != seen; });
        seen = generation_;
        if (index >= helpers_) continue;   // esta chamada pediu menos threads
      }
      work();
      std::lock_guard<std::mutex> lock(mutex_);
      if (--busy_ == 0) done_.notify_one();
    }
  }

  std::mutex call_mutex_, mutex_;
  std::condition_variable wake_, done_;
  std::vector<std::thread> workers_;
  const std::function<void(i64)>* fn_ = nullptr;
  i64 tasks_ = 0;
  std::atomic<i64> next_{0};
  int helpers_ = 0, busy_ = 0;
  u64 generation_ = 0;
};

Pool* pool() {
  // criado na primeira vez e nunca destruído (as threads seguem vivas até o processo acabar). Depois de um fork, o
  // filho não tem as threads do pai: um pool novo.
  static Pool* p = nullptr;
#ifndef _WIN32
  static pid_t owner = 0;
  if (p != nullptr && owner != getpid()) p = nullptr;
  if (p == nullptr) owner = getpid();
#endif
  if (p == nullptr) p = new Pool();
  return p;
}

std::mutex pool_mutex;

}  // namespace

void parallel_for(int threads, i64 tasks, const std::function<void(i64)>& fn) {
  if (tasks <= 0) return;
  if (threads <= 1 || tasks == 1 || inside_task) {
    for (i64 t = 0; t < tasks; t++) fn(t);
    return;
  }
  Pool* p;
  {
    std::lock_guard<std::mutex> lock(pool_mutex);
    p = pool();
  }
  p->run(std::min<i64>(threads, tasks) > 0 ? (int)std::min<i64>(threads, tasks) : 1, tasks, fn);
}

i64 task_rows(i64 n, int threads) {
  const i64 per = n / std::max<i64>(1, 4 * (i64)threads);
  return std::max<i64>(4096, std::min<i64>(65536, per));
}

}  // namespace dc
