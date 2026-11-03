# nini-datacomp

Comparação de tabelas em escala: a reescrita do [datacompolars](https://github.com/blzrosa/datacompolars) com a
comparação em kernels C++, que leem as colunas do Polars sem cópia (interface C do Arrow, sem pyarrow).

> **Em construção (fase 4).** A API já compara; o relatório e a publicação no PyPI fecham a 0.1.0.

```python
import nini_datacomp as dc  # antes do Polars (veja Memória)

result = dc.compare(
    "sas/tops_adm.parquet",  # caminho, glob, pasta, pl.DataFrame ou pl.LazyFrame
    "lake/tops_adm/",
    on=["id", "competencia"],
    abs_tol={"valor": 0.005},
    memory_limit="2GB",
)
result.is_match, result.rows.mismatched, result.columns["valor"].mismatches
result.columns["valor"].samples  # pl.DataFrame: chave, valor (left), valor (right)
```

## Memória

Importe o `nini_datacomp` antes do Polars. No import, ele configura o alocador do Polars (`MIMALLOC_PURGE_DELAY=0`)
para devolver na hora a memória que libera; sem isso, numa comparação grande no Windows o pico chega a ~1,9x (medido
na fase 1c). Se o Polars já tiver sido importado, não dá mais para configurar e o pacote avisa com `AllocatorWarning`;
dá para definir a variável antes de iniciar o Python, ou silenciar com `NINI_DATACOMP_ALLOCATOR=0`.

## Desenvolvimento

Precisa de Python ≥ 3.10, [uv](https://docs.astral.sh/uv/), CMake ≥ 3.20 e um compilador C++17 (no Windows, o Visual
Studio com as ferramentas de C++; os wheels publicados usam o clang-cl).

```bash
uv sync                                    # compila o C++ e instala o pacote no ambiente
uv run python tools/check.py               # formata, lint, tipos e testes, parando no primeiro que falhar
uv run python tools/check.py --then bench_hash   # ... e, se tudo passar, um benchmark
uv run python tools/gen_bindings.py        # depois de mudar a assinatura de uma função C++
```

Ferramentas (`tools/`):

| comando | o que faz |
|---|---|
| `uv run python tools/make_data.py --rows 1000000 --cols 50` | gera um par de tabelas com o resultado esperado (`data/bench/`) |
| `uv run python tools/make_golden.py` | regrava os gabaritos dos casos de borda (`tests/golden/`) com o datacompolars 0.4.4 |
| `uv run python tools/bench_baseline.py` | mede a linha de base (datacompolars 0.4.4) e escreve `docs/benchmarks/baseline.md` |

No Windows, para compilar como os wheels: `$env:DC_CLANGCL = "1"` antes do `uv sync` (precisa dos componentes
"C++ Clang Compiler for Windows" e "MSBuild support for LLVM (clang-cl) toolset" no Visual Studio Installer).

| variável | efeito |
|---|---|
| `NINI_DATACOMP_LIB` | caminho de uma biblioteca compilada à mão (ex. com sanitizers) |
| `NINI_DATACOMP_THREADS` | número de threads (padrão: todas as CPUs disponíveis) |
| `NINI_DATACOMP_POOL=0` | desliga o pool de memória do C++ (só para comparar) |

## Licença

Apache-2.0 (veja `LICENSE` e `NOTICE`).
