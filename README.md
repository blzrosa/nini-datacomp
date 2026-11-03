# nini-datacomp

Comparação de tabelas em escala: a reescrita do [datacompolars](https://github.com/blzrosa/datacompolars) com a
comparação em kernels C++, que leem as colunas do Polars sem cópia (interface C do Arrow, sem pyarrow).

> **0.1.0rc1.** Parquet (arquivo, pasta, glob), `pl.DataFrame` e `pl.LazyFrame`; inteiros, reais, booleano, texto
> (String, Categorical e Enum comparados como texto), Date, Datetime, Duration, Time e Decimal. CSV, SAS, List e
> Struct chegam na fase 6.

```bash
pip install nini-datacomp
```

```python
import nini_datacomp as dc  # antes do Polars (veja Memória)

result = dc.compare(
    "sas/tops_adm.parquet",  # caminho, glob, pasta, pl.DataFrame ou pl.LazyFrame
    "lake/tops_adm/",
    on=["id", "competencia"],
    abs_tol={"valor": 0.005},
    memory_limit="2GB",
)
print(result.report())  # texto; também "markdown", "html" e "json"
result.report(path="relatorio.html")  # o formato sai da extensão; devolve o texto e salva
result.is_match, result.rows.mismatched, result.columns["valor"].mismatches
result.columns["valor"].samples  # pl.DataFrame: id, competencia, valor (left), valor (right)
```

As opções vão como argumentos nomeados (como acima) ou num `dc.CompareOptions(...)` em `options=`. Erros fatais
(chave que falta, chave com tipos diferentes, chave repetida) não levantam exceção: ficam em `result.errors`, com
`result.aborted` verdadeiro.

## Vindo do datacompolars

Mesma semântica (os testes conferem os casos de borda contra o datacompolars 0.4.4, amostras inclusive); mudam os
nomes e o que era ajuste interno do motor vira um orçamento de memória.

| datacompolars 0.4.4 | nini-datacomp | observação |
|---|---|---|
| `compare(left, right, settings)` | `compare(left, right, on=..., options=...)` | `on` sai das opções; opções também como argumentos nomeados |
| `CompareSettings` | `CompareOptions` | pydantic, `extra="forbid"` (nome errado é erro) |
| `join_columns` | `on` | uma coluna (texto) ou várias |
| `left_name`, `right_name` | `names=("left", "right")` | |
| `lowercase_columns=True` | `case_sensitive_columns=False` | mesmo padrão, nome positivo |
| `casting` | `cast` | `"none"`, `"left"` ou `"right"` |
| `abs_tol`, `rel_tol` | `abs_tol`, `rel_tol` | número ou um por coluna (dict) |
| `ignore_spaces`, `ignore_case` | `ignore_whitespace`, `ignore_case` | |
| `check_duplicate_keys` | — | chave repetida sempre aborta (a checagem não custa nada) |
| `column_details`, `sample_count` | `details`, `samples` | |
| `window_rows`, `cache_windows`, `prefetch_windows`, `columns_per_batch` | `memory_limit="auto"` (+ `window_rows`) | o motor escolhe janelas e grupos de colunas pelo orçamento; `"auto"` = 1/4 da RAM livre, até 2 GB |
| — | `threads`, `verify_hash` | threads dos kernels C++; conferir o hash de linha célula a célula |
| `common_keys_only` | — | fase 6 |
| `custom_comparators` | — | fase 6 |
| `SourceOptions` | — | CSV e SAS na fase 6 |
| `result.fatal_errors` | `result.errors` | |
| `result.column_stats` | `result.columns.mismatched()` | mesma ordem: da coluna que mais diverge para a que menos |
| `result.samples` (lista de `SampleTable`) | `result.columns["c"].samples` | `pl.DataFrame`, mesmos cabeçalhos |
| `result.columns.compared` | `result.columns.compared` | `result.columns` também é um mapeamento por nome |
| `result.report(ReportSettings(...))` | `result.report(format=None, path=None, width=100, ...)` | os mesmos 4 formatos; não imprime sozinho (`print_output=True` imprime) |
| `result.to_json()`, `model_dump()` | `result.to_json()`, `result.to_dict()` | o resultado é uma dataclass, não pydantic |

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
