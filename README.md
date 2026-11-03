# nini-datacomp

[![PyPI](https://img.shields.io/pypi/v/nini-datacomp)](https://pypi.org/project/nini-datacomp/)
[![Python](https://img.shields.io/pypi/pyversions/nini-datacomp)](https://pypi.org/project/nini-datacomp/)
[![CI](https://github.com/blzrosa/nini-datacomp/actions/workflows/ci.yml/badge.svg)](https://github.com/blzrosa/nini-datacomp/actions/workflows/ci.yml)
[![Licença](https://img.shields.io/pypi/l/nini-datacomp)](https://github.com/blzrosa/nini-datacomp/blob/main/LICENSE)

**Compara duas tabelas grandes linha a linha, dentro de um orçamento de memória.** Diz quais linhas existem só de
um lado, quais divergem, em que colunas e com exemplos, num relatório em texto, Markdown, HTML ou JSON.

É a reescrita do [datacompolars](https://github.com/blzrosa/datacompolars): o mesmo resultado (os testes conferem
os casos de borda contra ele, amostra por amostra), com a comparação em kernels C++ que leem as colunas do Polars
sem cópia.

- **Memória sob controle:** a comparação anda em janelas pela chave, do tamanho que cabe no `memory_limit`
  (`"auto"` = 1/4 da RAM livre, até 2 GB). Tabelas maiores que a memória são comparadas sem problema.
- **Rápida:** 10 milhões de linhas x 300 colunas em 33 s com 1 GB, onde o datacompolars levava 50 s e usava
  quase 2 GB ([números](#desempenho)).
- **Sem surpresas:** igualdade exata pelo hash de linha (com `verify_hash=True` para conferir célula a célula),
  tolerâncias absolutas e relativas com a aritmética do Polars, texto sem caixa e sem espaços, nulos e NaN como no
  datacompolars.
- **Leve:** só `polars`, `numpy` e `pydantic`; wheels prontos para Linux, Windows e macOS (x86-64 e ARM), sem
  compilador.

## Instalação

```bash
pip install nini-datacomp
```

Python 3.10 ou mais novo e Polars 1.30 ou mais novo.

## Uso

```python
import nini_datacomp as dc  # importe antes do Polars (veja Memória)

result = dc.compare(
    "sas/tops_adm.parquet",  # arquivo, pasta, glob, pl.DataFrame ou pl.LazyFrame
    "lake/tops_adm/",
    on=["id", "competencia"],  # a chave: uma coluna ou várias
    abs_tol={"valor": 0.005},  # tolerância numa coluna (ou um número para todas)
    memory_limit="2GB",
)

result.is_match  # False
result.rows.mismatched  # linhas em comum com algum valor diferente
result.columns["valor"].mismatches  # divergências nessa coluna
result.columns["valor"].samples  # pl.DataFrame: id, competencia, valor (left), valor (right)

print(result.report())  # o relatório em texto
result.report(path="relatorio.html")  # ou em HTML (o formato sai da extensão)
```

As opções vão como argumentos nomeados, como acima, ou num `dc.CompareOptions(...)` passado em `options=`.

### O resultado

| atributo | o que traz |
|---|---|
| `is_match` | os dois lados são iguais: mesmas linhas, mesmos valores, mesmas colunas e tipos |
| `rows` | `left_total`, `right_total`, `common`, `left_only`, `right_only`, `mismatched`, `matched`, `mismatch_rate` |
| `columns` | mapeamento por nome das colunas comparadas (`dtype`, `mismatches`, `rate`, `samples`, `tolerance`), mais `left_only`, `right_only`, `type_mismatches` e `mismatched()` (da que mais diverge para a que menos) |
| `errors`, `aborted` | erros fatais (chave que falta, chave de tipos diferentes, chave repetida): ficam aqui, sem exceção |
| `warnings` | avisos (nenhuma coluna além da chave, tolerância para uma coluna que não existe) |
| `execution` | janelas, grupos de colunas, memória, tempo por etapa, versões |
| `report(...)`, `to_dict()`, `to_json()` | o relatório e a forma serializável |

### O relatório

`result.report(format=None, path=None, *, width=100, style="auto", max_columns=20, max_sample_columns=10,
print_output=False)`, nos formatos `"text"`, `"markdown"`, `"html"` e `"json"`. Sem `format`, o formato sai da
extensão de `path` (`.md`, `.html`, `.json`, `.txt`); com `path`, o relatório também é salvo. Texto e Markdown
mostram até `max_columns` colunas com divergência; HTML e JSON trazem tudo.

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ ✘ DIFFERENCES FOUND                                                          │
│ 2 of 3 common rows differ (66.67%); 1 rows only in sas; 1 rows only in lake. │
└──────────────────────────────────────────────────────────────────────────────┘

DIFFERENCES BY COLUMN
─────────────────────

  ┌────────┬─────────┬─────────────┬─────────────┬──────────────┐
  │ Column │ Type    │ Differences │ % of common │              │
  ├────────┼─────────┼─────────────┼─────────────┼──────────────┤
  │ valor  │ Float64 │           2 │      66.67% │ ████████░░░░ │
  └────────┴─────────┴─────────────┴─────────────┴──────────────┘

SAMPLE DIFFERING ROWS
─────────────────────

  valor: 2 differing rows, showing 2
  ┌────┬─────────────┬─────────────┬──────────────┐
  │ id │ competencia │ valor (sas) │ valor (lake) │
  ├────┼─────────────┼─────────────┼──────────────┤
  │ 2  │ 2026-01     │ 10          │ 10.5         │
  │ 3  │ 2026-01     │ 7.25        │ null         │
  └────┴─────────────┴─────────────┴──────────────┘
```

### Opções

| opção | padrão | o que faz |
|---|---|---|
| `names` | `("left", "right")` | os rótulos dos dois lados nas amostras e no relatório |
| `case_sensitive_columns` | `False` | `False`: os nomes das colunas são comparados em minúsculas |
| `cast` | `"none"` | colunas de tipos diferentes: `"left"` converte a direita, `"right"` a esquerda, `"none"` só as lista |
| `abs_tol`, `rel_tol` | `0.0` | tolerâncias nas colunas numéricas (um número ou um por coluna): igual se `\|l - r\| <= abs_tol + rel_tol * \|r\|` |
| `ignore_whitespace`, `ignore_case` | `False` | texto comparado sem espaços em branco e/ou em minúsculas |
| `details` | `True` | divergências por coluna e amostras (`False`: só as contagens de linhas) |
| `samples` | `5` | linhas divergentes guardadas por coluna, na ordem da chave |
| `memory_limit` | `"auto"` | orçamento de memória: bytes ou texto (`"512MB"`, `"2GB"`, `"1.5GiB"`) |
| `window_rows` | `None` | avançado: linhas por janela, no lugar do cálculo pelo `memory_limit` |
| `threads` | `None` | threads dos kernels C++ (padrão: todos os núcleos) |
| `verify_hash` | `False` | confere as colunas exatas célula a célula em todos os pares, além do hash |

Uma opção com nome errado é erro (`extra="forbid"`), em vez de ser ignorada.

### Tipos

Inteiros, reais, booleano, texto (String, e Categorical e Enum comparados como texto), Date, Datetime, Duration,
Time e Decimal. As fontes são Parquet (arquivo, pasta com subpastas, glob), `pl.DataFrame` e `pl.LazyFrame`. List,
Struct, CSV e SAS ainda não: uma coluna de tipo não suportado levanta `UnsupportedTypeError`.

## Desempenho

O caminho exato (hash de linha) contra o datacompolars 0.4.4, com `memory_limit="1GB"` (954 MiB), num Windows 11 com
16 threads e 15,7 GB. Média de 5 execuções, cada uma num processo novo; a razão vem com o IC de 95%, e um caso é `ok`
se o intervalo inteiro fica em até 1,10. As contagens (linhas e divergências por coluna) conferem em todos.

| linhas x colunas | nini-datacomp | pico | datacompolars | pico | razão (IC 95%) |
|---|---:|---:|---:|---:|---|
| 1M x 10 | 0,18 s | 401 MiB | 0,22 s | 698 MiB | 0,81 (0,75–0,88) |
| 1M x 50 | 0,45 s | 724 MiB | 0,43 s | 1070 MiB | 1,05 (1,01–1,09) |
| 1M x 300 | 2,72 s | 691 MiB | 3,35 s | 1709 MiB | 0,81 (0,72–0,91) |
| 10M x 10 | 1,11 s | 768 MiB | 1,16 s | 1631 MiB | 0,96 (0,94–0,99) |
| 10M x 50 | 5,10 s | 805 MiB | 4,26 s | 1679 MiB | 1,20 (1,05–1,35), inconclusivo |
| 10M x 300 | 32,9 s | 814 MiB | 50,0 s | 1929 MiB | 0,66 (0,65–0,66) |

O nini-datacomp fica dentro do `memory_limit` em todos os casos; o datacompolars não tem orçamento e usa até o dobro.
Os detalhes, com o caminho coluna a coluna (com tolerância), estão em
[`docs/benchmarks/`](https://github.com/blzrosa/nini-datacomp/tree/main/docs/benchmarks).

## Memória

Importe o `nini_datacomp` antes do Polars. No import, ele configura o alocador do Polars (`MIMALLOC_PURGE_DELAY=10`
e `MIMALLOC_RESET_DELAY=10`) para devolver em 10 ms a memória que libera: sem isso, numa comparação grande no Windows
o pico chega a ~1,9x; devolvendo na hora, cada janela pede de novo ao sistema a memória da anterior (até ~30% mais
lento). Se o Polars já tiver sido importado, não dá mais para configurar e o pacote avisa com `AllocatorWarning`; dá
para definir as variáveis antes de iniciar o Python, ou silenciar com `NINI_DATACOMP_ALLOCATOR=0`.

## Vindo do datacompolars

Mesma semântica; mudam os nomes, e o que era ajuste interno do motor vira um orçamento de memória.

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
| `uv run python tools/bench_hash.py` | mede o nini-datacomp nos mesmos casos (`--mode columnwise` para o caminho com tolerância) e escreve `docs/benchmarks/fase2.md` / `fase3.md` |
| `uv run python tools/diag_profile.py --case 1000000x50` | onde vai o tempo de uma comparação (cProfile e o planejamento por partes) |

No Windows, para compilar como os wheels: `$env:DC_CLANGCL = "1"` antes do `uv sync` (precisa dos componentes
"C++ Clang Compiler for Windows" e "MSBuild support for LLVM (clang-cl) toolset" no Visual Studio Installer).

| variável | efeito |
|---|---|
| `NINI_DATACOMP_LIB` | caminho de uma biblioteca compilada à mão (ex. com sanitizers) |
| `NINI_DATACOMP_THREADS` | número de threads (padrão: todas as CPUs disponíveis) |
| `NINI_DATACOMP_POOL=0` | desliga o pool de memória do C++ (só para comparar) |

## Licença

Apache-2.0 (veja [`LICENSE`](https://github.com/blzrosa/nini-datacomp/blob/main/LICENSE) e [`NOTICE`](https://github.com/blzrosa/nini-datacomp/blob/main/NOTICE)).
