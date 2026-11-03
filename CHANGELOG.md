# Changelog

Formato: [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/); versões: [SemVer](https://semver.org/lang/pt-BR/).

## [Não lançado]

## [0.1.0] — 2026-11-03

Primeira versão: a reescrita do datacompolars 0.4.4 com a comparação em kernels C++.

- `dc.compare(left, right, on, ...)` com `CompareOptions` e `ComparisonResult`: o mesmo resultado do datacompolars
  (contagens, divergências por coluna e amostras, conferidos contra ele em 21 casos de borda), com os nomes da
  tabela de migração do README.
- Comparação por janelas pela chave, dentro de um `memory_limit`: todos os casos da grade (1M e 10M linhas x 10, 50
  e 300 colunas) ficam abaixo de 1 GB, onde o datacompolars chegava a quase 2 GB.
- Igualdade exata pelo hash de linha (`verify_hash=True` confere célula a célula); tolerâncias, `ignore_case` e
  `ignore_whitespace` coluna a coluna, com a aritmética do Polars.
- Relatório em texto, Markdown, HTML e JSON (`result.report`).
- Desempenho (caminho exato, `memory_limit="1GB"`): de 0,66x a 1,05x o tempo do datacompolars, e 10M x 50 em 1,20x
  (inconclusivo: o IC de 1,05 a 1,35 passa de 1,10, mas não fica inteiro acima). Nenhum caso mais lento.
- Wheels para Linux, Windows e macOS (x86-64 e ARM64), Python 3.10 a 3.14, sem pyarrow.

O histórico abaixo, fase a fase, é o do desenvolvimento.

### Fase 0a — fundação
- Biblioteca C++ (`datacomp.dll` / `libdatacomp.so` / `.dylib`) com ABI em C de prefixo `dc_`, versão de ABI conferida
  no carregamento (`DC_ABI_VERSION`) e assinaturas do ctypes geradas dos `.cpp` (`tools/gen_bindings.py`).
- Do nini-datagen 0.3.0: build com scikit-build-core e CMake (clang-cl nos wheels do Windows, x86-64-v2, sem FMA,
  `/MT`), pool de threads (`core/blocks.py`), pool de memória (`cpp/memory.cpp`, `core/memory.py`), interface C do
  Arrow sem pyarrow (`core/arrow_c.py`) e os workflows de CI e de wheels.
- `core/arrow.py`: `import_column` lê qualquer coluna dos tipos da 0.1 (inteiros, reais, booleano, texto, Date,
  Datetime, Duration, Time, Decimal) como buffers do Arrow, sem cópia; outro tipo levanta `UnsupportedTypeError`.
- API pública por enquanto: `release_memory()`, `pool_limit()`, `pool_stats()`, `__version__`.

### Fase 0b — dados, gabaritos e linha de base
- `tools/make_data.py`: pares de tabelas Parquet gerados com o nini-datagen (semente fixa; inteiros, reais, texto,
  booleano, data e data e hora em ciclo, ~2% de nulos), com o lado direito mudado de forma controlada e o resultado
  esperado gravado pela própria construção (`expected.json`).
- `tools/cases.py` + `tools/make_golden.py`: 21 casos de borda (chave composta, de texto, nula, esparsa, fora de
  ordem; duplicatas; lado vazio; -0.0, NaN, infinitos; texto com acentos, NBSP e emoji; Categorical; larguras de
  inteiro; tolerâncias na borda; 300 colunas; tipos temporais e Decimal) com gabaritos do datacompolars 0.4.4 pelo
  caminho columnwise em `tests/golden/`. Alguns conferidos à mão nos testes.
- `tools/bench_baseline.py` + `tools/benchstats.py`: linha de base do datacompolars 0.4.4 na grade do plano (100 a 10M
  linhas x 10 a 300 colunas, caminhos hash e columnwise), cada execução num processo novo, média e IC de 95%,
  retomável; resultado em `docs/benchmarks/baseline.{md,json}`.

### Fase 0c — experimento de leitura
- `tools/read_experiment.py`: lê as mesmas janelas da comparação (faixas de ids, os dois lados) pelo Polars, pelo
  leitor C++ do Apache Arrow, pelo Polars sem compressão e só os bytes do disco, para medir o teto do ganho de um
  leitor de Parquet próprio antes de escrevê-lo. Grupo `bench` (pyarrow) só para isso; o pacote não usa o pyarrow.
- Gabaritos: o tipo divergente guarda só o nome do tipo (o texto do Categorical muda entre versões do Polars).
- Decimal: o Polars 1.30 não o exporta pela interface C do Arrow (nem pela representação física); exige um Polars
  mais novo.

### Fase 1a — alinhamento das chaves (ABI 2)
- Kernel `dc_align_keys` (`cpp/keys.cpp`) e `core/keys.py` (`align_keys`): pares de linhas com a mesma chave e as
  contagens do datacompolars (em comum, só esquerda, só direita, excedentes por chave repetida). Chave de qualquer tipo
  da 0.1 e composta; nulo casa com nulo, `-0.0 == 0.0`, NaN casa com NaN. Merge linear quando os dois lados vêm
  ordenados (caminho direto para chave inteira sem nulos), hash join quando não; com chave repetida, só as contagens.
- Medido (Linux, 1 thread, chave int64): merge ~7 ns/linha; hash ~27 ns/linha numa janela de 100 mil linhas e ~59 em
  1 milhão. Particionar o hash e paralelizar ficam para a fase 5.
- `import_column` recusa colunas codificadas por dicionário (Categorical, Enum): os buffers seriam códigos, diferentes
  em cada tabela. A comparação converte para String antes.

### Fase 1b — janelas por faixa de qualquer chave (ABI 3)
- `windows.py` (`plan_windows`): janelas = faixas da tupla inteira da chave (ordem lexicográfica), de qualquer tipo
  ordenável — inteiro, texto, data, chave composta; acaba a exigência do datacompolars de 1ª coluna inteira. Chaves com
  nulo vão para a primeira janela. Fronteiras = a chave a cada N linhas do lado maior, por uma leitura só das colunas
  da chave.
- Tamanho da janela pelo `memory_limit` (`"auto"` = 1/4 da memória livre, até 2 GiB; ou `"2GB"`, `"1.5GiB"`, bytes):
  N = orçamento x 0,8 / (2 janelas x (bytes por linha dos dois lados + trabalho dos kernels)), com os bytes por linha
  medidos pelo Polars numa amostra (memória depois de lida, não o tamanho no disco). `window_rows` fixa N à mão.
- Com o lado maior gravado em ordem, as fronteiras vão para o começo do row group mais próximo: nenhum row group é
  decodificado duas vezes. As linhas de cada row group vêm do rodapé do Parquet, lido por `cpp/parquet_meta.cpp`
  (`dc_parquet_row_groups`, Thrift compacto, sem pyarrow).
- `io.py`: fontes Parquet (arquivo, pasta com subpastas, glob), `pl.DataFrame` e `pl.LazyFrame`; CSV e SAS na fase 6.

### Fase 1c — caminho completo: contagem de linhas por janelas
- `pipeline.py`: `count_rows(left, right, on, memory_limit=..., window_rows=..., columns=...)` abre as fontes, confere
  a chave (Categorical/Enum viram texto; tipos diferentes nos dois lados = erro claro), planeja as janelas, lê cada uma
  pelo Polars e alinha as chaves no C++, somando as contagens. `WindowReader` lê a janela k+1 numa thread enquanto a k
  é processada (no máximo duas em memória), e é a base das fases 2 e 3.
- A primeira janela só leva a cláusula "ou chave nula" quando algum lado tem chave nula: o "ou" impedia o Polars de
  pular row groups pelas estatísticas, e a primeira janela leria a tabela inteira.
- `tools/bench_rows.py`: memória e tempo de 1M a 10M linhas (chave inteira e chave de texto), cada execução num
  processo novo; resultado em `docs/benchmarks/fase1.{md,json}`.
- Medido no primeiro benchmark (`memory_limit = 1GB`): contagens certas em tudo, mas pico de 1,6–2,1 GB em 10M linhas,
  crescendo com as linhas. Corrigido no planejamento: ele não carrega mais todas as chaves (nem cópias ordenadas) —
  só conta as linhas numa passada, lê a chave do começo de cada row group (Parquet em ordem) ou uma amostra de até 200
  mil chaves; a janela passa a contar 4 cópias vivas no pico (medido), não 2; e a base é o lado em ordem no disco
  (antes era sempre o maior, que no make_data tem os arquivos extra-* fora de ordem).
- Medido no segundo (`memory_limit = 1GB`): pico de 1,3–1,6 GB em 10M linhas com a maior janela em ~200 MB. Para achar
  de onde vem o resto: `engine` no `count_rows`/`WindowReader` (o engine do `collect` do Polars em cada janela) e
  `tools/diag_memory.py`, que anota a memória etapa a etapa e janela a janela e roda variações que isolam uma causa
  por vez (sem leitura em paralelo, menos threads, engine streaming, só a chave, janela menor, alocador devolvendo a
  memória na hora, sem os kernels, sem o pool).
- Diagnóstico: o resto do pico era memória que o alocador do Polars (mimalloc, no Windows) guarda depois de liberar e
  vai acumulando janela após janela; janelas menores, sem leitura em paralelo, sem os kernels ou sem o pool quase não
  mudavam o pico. Com `MIMALLOC_PURGE_DELAY=0`: 10M x 10 de 1159 para 605 MB, 10M x 50 de 1094 para 693 MB (e 5,6 →
  4,2 s nesse caso).
- `core/allocator.py`: o import do `nini_datacomp` define `MIMALLOC_PURGE_DELAY=0` quando o Polars ainda não foi
  importado (sem sobrescrever um valor existente); quando já foi, avisa com `AllocatorWarning` (no Windows).
  `NINI_DATACOMP_ALLOCATOR=0` desliga; `nini_datacomp.ALLOCATOR_STATUS` diz o que aconteceu. Os benchmarks medem com o
  alocador configurado.
- `count_rows` solta as tabelas de cada janela antes de pedir a próxima: a variável do laço segurava a janela k
  enquanto o leitor entregava a k+1 e já lia a k+2 (três janelas vivas; ~200 MB a mais no pico em 10M linhas, a
  diferença entre o bench_rows e o diag_memory).

### Fase 2 — hash de linha (ABI 4)
- Kernels `dc_row_hash` e `dc_diff_pairs` (`cpp/rowhash.cpp`) e `core/rowhash.py` (`row_hash`, `diff_pairs`): um hash de
  64 bits por linha das colunas comparadas, por blocos de 65.536 linhas em paralelo, e os pares (da `align_keys`) com
  hash diferente. Hash próprio, igual em toda plataforma e versão do Polars (valores fixados nos testes), com a
  igualdade do `eq_missing`: nulo = nulo, `-0.0 == 0.0`, NaN = NaN, texto inline ou em buffer, com ou sem bitmap de
  validade. Construção: `h = fmix64((h ^ valor) + sal da coluna)` — o fmix64 (do murmur3) é uma bijeção, então duas
  linhas que diferem numa coluna só nunca colidem; o plano falava em wyhash, que precisaria de multiplicação de 128
  bits (não portátil no clang-cl com `-Wpedantic`), sem ganho de qualidade aqui.
- `verify=True` (`verify_hash` no `compare_rows`) reconfere célula a célula os pares de hash igual e conta as colisões.
- Medido (Linux, 1 thread): ~3 ns por célula, inteiro ou real.
- `cpp/columns.hpp`: a leitura das colunas (descritores, valores, igualdade) sai do `keys.cpp` para os dois kernels.
- `pipeline.compare_rows(left, right, on, casting=..., columns=..., verify_hash=...)`: as contagens do `count_rows` e
  as linhas divergentes (`mismatched`, `equal`). Colunas comparadas como no datacompolars: nos dois lados, fora da
  chave, com o mesmo tipo depois do `casting` (`"left"`/`"right"` convertem um lado para os tipos do outro, também na
  chave); as de tipo diferente vão para `type_mismatches`; Categorical/Enum comparados como texto; colunas aninhadas
  (List, Struct) levantam `UnsupportedTypeError` (fase 6). Só as colunas comparadas são lidas.
- Testes: os gabaritos de igualdade exata (linhas divergentes, colunas comparadas, tipos divergentes), com uma janela
  e com janelas de 2 linhas, e os dados do make_data.
- `tools/bench_hash.py`: tempo e memória de 1M a 10M linhas x 10 a 300 colunas contra o caminho hash do
  datacompolars, com a razão e o IC de 95%; resultado em `docs/benchmarks/fase2.{md,json}`.
- Primeiro benchmark (`memory_limit = 1GB`): contagens certas em tudo, mas mais lento que o datacompolars (10M x 300:
  80 s contra 50 s) e acima do limite (1,5 GB). Duas causas corrigidas:
  - o hash juntava cada coluna da janela num pedaço só de memória (`rechunk`, uma cópia da janela inteira por lado)
    quando o Polars a entregava em vários (um por row group): agora `row_hash_parts` lê pedaço a pedaço, sem cópia
    (o `verify_hash` ainda junta, porque lê fora de ordem). O `import_column` nem tenta exportar uma coluna em vários
    pedaços (a exportação falharia e custava tempo).
  - com 300 colunas o orçamento dava janelas de ~52 mil linhas, menores que os row groups de 100 mil: cada row group
    era decodificado duas vezes. A janela agora nunca é menor que o maior row group (com uma nota no plano).
- Segundo benchmark: contagens certas; 10M x 300 em 43 s (0,86x o datacompolars, `ok`), 10M x 10 e 1M x 300 `ok`;
  10M x 50 (1,13x) e 1M x 50 (1,58x) ainda mais lentos. O tempo estava no hash (30 s dos 43 em 10M x 300), e não no
  kernel: ler as colunas uma a uma pelo Python (`import_column`) custa ~0,3 ms por coluna, e uma janela de 300 colunas
  e 100 mil linhas tinha só 2 blocos de 65.536 linhas para 16 threads. Corrigido:
  - `import_frame` (`core/arrow.py`) exporta o pedaço do DataFrame inteiro de uma vez (`__arrow_c_stream__` do
    DataFrame, um struct do Arrow) e `dc_describe_batch` (`cpp/batch.cpp`) preenche os descritores de todas as colunas
    numa chamada só. Sem lote único (Categorical, Polars antigo, vários pedaços), volta à leitura coluna a coluna.
  - o hash divide as linhas em tarefas de 4.096 a 65.536 linhas, ao menos ~4 por thread (o resultado não muda).
- Terceiro benchmark: o hash caiu para 6 s em 10M x 300 (de 30) e o pico para ~1,2 GB (de 1,6), mas o tempo total
  ficou em 44,5 s: agora é a leitura que não fica escondida (`read_wait` 30 s). Ela crescia mais que as linhas
  (10M x 300 leva 10x as linhas e 10x as janelas de 1M x 300, mas ~17x a leitura) porque cada janela lia a fonte
  inteira: o Polars abria o rodapé dos 100 arquivos de cada lado a cada uma das 100 janelas (com 300 colunas, um
  rodapé grande). Agora o planejamento anota a faixa da 1ª coluna da chave de cada arquivo (uma passada pela chave) e
  cada janela lê só os arquivos que encostam nela (`Source.scan`, com as mesmas conversões de tipo da fonte).
- Diagnóstico da leitura (`tools/diag_read.py`): o Polars não tem custo fixo alto por leitura; o tempo é proporcional
  às células (~3,5 ns cada, também com o disco frio), e o corte de arquivos por janela ajuda pouco (fica). O que pesa
  é o tamanho: os row groups de 100 mil linhas dos dados de benchmark eram arbitrários.
- Dados de benchmark com o row group padrão do Polars (`Spec.row_group_size = None`; a pasta ganha `_rg<n>` quando o
  row group é fixo) e arquivos de ~100 milhões de células (antes 30 milhões: um row group por arquivo nas tabelas
  largas). O `read_experiment` (fase 0c) continua com os row groups de 100 mil.
- Grupos de colunas: quando um row group dos dois lados não cabe no `memory_limit`, cada janela (um row group) é lida
  em grupos de colunas seguidas, cada um com a chave, e o hash de linha continua de um grupo para o outro
  (`dc_row_hash(..., first, ...)`, ABI 5: mesmos bits que todas as colunas juntas). A memória passa a depender do
  grupo, não do row group. `Plan.column_groups`/`split_columns`, `WindowReader(groups=...)`/`pieces()`,
  `RowCounts.column_groups`. O `verify_hash` confere célula a célula grupo a grupo e junta os pares diferentes.
- `tools/check.py`: formatação, lint, tipos e testes em sequência, parando no primeiro que falhar; `--fix` e
  `--then <ferramenta>` (benchmarks no fim).
- Benchmark global (dados novos, linha de base refeita): divergentes certos em tudo; todos os picos abaixo de 1 GB
  (10M x 300: 588 MB em 3 grupos de colunas, 47,1 s, 0,94x o datacompolars); mas 1M x 50 (1,41x) e 10M x 50
  (1,56x) mais lentos, com a leitura dominando e só ~2/3 do memory_limit em uso. Ajustes:
  - `IN_FLIGHT` de 4 para 3 (o pico medido, já sem o bug das três janelas e com o alocador configurado, é de 1,9 a
    2,9x a maior janela): janelas ~1/3 maiores, menos leituras.
  - os dois lados de cada janela lidos ao mesmo tempo (`WindowReader(parallel_sides=True)`): uma leitura do Polars
    de poucos row groups não ocupa todas as threads.
- Resultado: 10M x 300 em 36,9 s (0,74x o datacompolars) e 917 MB; 10M x 50 de 6,7 para 5,4 s (1,27x); 1M x 50 quase
  igual (1,37x). Mas a contagem com chave de texto em 10M x 10 passou do limite (1.086 MB): `IN_FLIGHT` = 3,5. Novas
  etapas nos tempos: `setup` (abrir as fontes e alinhar os schemas) e `other` (o que nenhuma etapa mediu; em 10M x 50
  eram ~1,6 s dos 5,4).
- Com `IN_FLIGHT` = 3,5: 10M x 300 em 36,5 s (0,73x), 1M x 300 0,88x, 10M x 10 1,01x, 10M x 50 1,01x (inconclusivo),
  1M x 50 1,31x (mais lento). O `other` era grande (10M x 300: 8,7 s; 10M x 50: 2,1 s): soltar a memória de cada
  janela na thread principal, com o alocador devolvendo as páginas na hora. Agora as janelas são soltas numa thread
  à parte (`_dispose`).
- Os relatórios dão o pico em MiB (`MB` = 1.048.576 bytes) e o `memory_limit` "1GB" é 10^9 bytes = 954 MiB: alguns
  picos passavam do limite por até ~3% (10M x 50: 959 MiB; contagem com chave de texto em 10M x 10: até 985 MiB).
  O `bench_hash` ganhou a coluna `cabe` (o maior pico contra o limite).
- Soltar as janelas noutra thread tirou o `other` (10M x 300: 8,7 → 2,6 s), mas o tempo total não mudou: o custo foi
  para a leitura em paralelo (o gargalo agora é o processador, lendo). Fica, porque não atrapalha. Dois picos ainda
  passavam do limite (10M x 50: 983 MiB; chave de texto em 10M x 10: 992 MiB; o limite de 1GB é 954 MiB):
  `SAFETY` de 0,8 para 0,7 (janelas ~12% menores).
- Com `SAFETY` = 0,7: todos os picos dentro do limite (o maior: 10M x 50, 904 MiB) e os tempos iguais. Fase 2
  fechada: 10M x 300 em 36,5 s (0,73x o datacompolars, 686 MiB); ficam para a fase 5 o 1M x 50 (1,32x) e o
  10M x 50 (1,03x, inconclusivo).

### Fase 3 — coluna a coluna (ABI 6)
- Kernel `dc_compare_column` (`cpp/compare.cpp`) e `core/compare.py` (`compare_column`, `tolerance_for`): uma coluna,
  par a par, exata (o `eq_missing` do Polars, NaN = NaN) ou com tolerância nas numéricas (`|l - r| <= abs_tol +
  rel_tol * |r|`, com a aritmética do Polars: inteiros no próprio tipo, com volta; Float32 em float; Decimal pelo
  inteiro de 128 bits). Nulo dos dois lados = igual. Conferido nos testes contra a própria expressão do datacompolars
  rodando no Polars instalado, em todos os tipos inteiros, nos reais e no Decimal.
- `compare_rows(..., abs_tol=..., rel_tol=..., ignore_case=..., ignore_spaces=...)` e `RowCounts.column_mismatches`
  (divergências por coluna). Em cada grupo de colunas de uma janela: as exatas pelo hash de linha e, só nos pares
  divergentes (poucos: um `gather` do Polars), a contagem coluna a coluna; as com tolerância, uma a uma, em todos os
  pares. Cada grupo tem o seu hash (um par diverge se diverge em algum grupo), então a contagem por coluna sai
  enquanto o grupo está em memória.
- Texto com `ignore_case`/`ignore_spaces`: normalizado na leitura com as mesmas expressões do datacompolars
  (`str.to_lowercase`, `str.replace_all(r"\s+", "")`), e depois comparado exato (pelo hash). O plano previa caixa e
  espaços no C++ (ASCII e Latin-1 rápidos, UTF-8 completo); com as expressões do Polars o resultado bate com o do
  datacompolars em qualquer texto Unicode, e o texto normalizado ainda vai pelo hash. O C++ fica para a fase 5, se o
  perfil mandar.
- Testes: todos os 21 gabaritos agora (tolerâncias e opções de texto inclusive), também por coluna e com
  `verify_hash`; grupos de colunas com tolerância.
- `tools/bench_hash.py --mode columnwise`: `abs_tol=1e-12`, contra o columnwise da linha de base, conferindo também
  as contagens por coluna; resultado em `docs/benchmarks/fase3.{md,json}`.
- Os testes contra o Polars confirmaram a aritmética dos inteiros (com volta, também os sem sinal) e acharam uma
  diferença nos reais: o Polars compara reais em ordem total (NaN maior que tudo, NaN = NaN), não pelo IEEE. Na
  tolerância isso importa: com r infinito, `rel_tol * |r|` pode dar `0 * inf = NaN`, e para o Polars `x <= NaN` é
  verdade (o datacompolars dá esses pares como iguais). O kernel agora compara como o Polars (`polars_le`).
- `column_mismatches` lista todas as colunas comparadas (as sem divergência com 0); o datacompolars lista só as
  com divergência.
- CI (min-deps, Polars 1.30): o Decimal exportado como Series sai com o formato `_pli128` (o mesmo inteiro de 128
  bits); o hash passava (o DataFrame inteiro sai como `d:`), mas o `verify_hash` e a coluna a coluna recusavam a
  coluna. `_pli128` aceito como Decimal no `import_column` e no `dc_describe_batch`.
- Benchmark coluna a coluna (`abs_tol = 1e-12`): contagens certas (também por coluna) em tudo; 10M x 300 em 37,2 s
  (0,57x o columnwise do datacompolars, 805 MiB); 1M x 50, 1M x 300, 10M x 10 e 10M x 50 `ok`; 1M x 10
  inconclusivo (1,17x). Um pico de 10M x 50 passou o limite em 1% (964 MiB): `SAFETY` = 0,65.

### Fase 4a — API pública
- `dc.compare(left, right, on, options=None, **kwargs)`: a função pública. As opções em `CompareOptions` (pydantic,
  `extra="forbid"`) ou como argumentos nomeados; com os dois, os nomeados ganham.
- `CompareOptions`: `names`, `case_sensitive_columns`, `cast`, `abs_tol`/`rel_tol` (número ou por coluna),
  `ignore_whitespace`, `ignore_case`, `details`, `samples`, `memory_limit`, `window_rows`, `threads`, `verify_hash`.
- `ComparisonResult` (dataclasses imutáveis): `rows`, `columns` como mapeamento por nome (`result.columns["v"]`, com
  `mismatches`, `rate`, `dtype`, `tolerance` e as amostras em `pl.DataFrame`), `columns.mismatched()` na ordem do
  datacompolars, `errors`, `warnings`, `execution`, `is_match`, `aborted`, `to_dict()`/`to_json()`.
- Amostras: as primeiras `samples` linhas divergentes de cada coluna, na ordem da chave, com cabeçalhos
  `[*chave, "<coluna> (<esquerda>)", "<coluna> (<direita>)"]`, como no datacompolars. Com `ignore_case`/
  `ignore_whitespace`, o texto normalizado é lido ao lado do original e as amostras mostram o valor de verdade.
- Erros fatais no resultado, não em exceção (como no datacompolars): chave que falta, chave de tipos diferentes sem
  `cast`, chave repetida, nomes que colidem em minúsculas. Avisos: nenhuma coluna além da chave; tolerância por coluna
  com nome que não é comparado.
- Chave repetida sempre aborta: não há `check_unique_keys=False` (o plano previa). A checagem sai de graça do
  alinhamento das chaves, e desligá-la só serviria para contagens infladas.
- Testes: os 21 gabaritos pela API, inclusive as amostras valor a valor, com uma janela e com janelas de 2 linhas.

### Fase 4b — relatório e publicação
- `result.report(format=None, path=None, *, width=100, style="auto", max_columns=20, max_sample_columns=10,
  print_output=False)`: o relatório do datacompolars nos mesmos quatro formatos (texto, Markdown, HTML, JSON),
  portado para o resultado novo. Sem `format`, sai da extensão de `path`; com `path`, também salva. Diferente do
  datacompolars, não imprime sozinho (`print_output=True` imprime).
- A seção "Execution" diz como cada coluna foi comparada (hash de linha ou coluna a coluna com tolerância), as
  janelas, os grupos de colunas, o `memory_limit` e a maior janela, os tempos por etapa e as versões.
- README: uso e a tabela de migração do datacompolars.
- Versão `0.1.0rc1` para o TestPyPI; o teste do TestPyPI nas máquinas limpas agora roda uma comparação de verdade e
  o relatório nos quatro formatos.

### Fase 5a — desempenho (antes da 0.1.0)
- Planejamento das janelas: as linhas de um Parquet saem do rodapé (sem ler dados) e as consultas que restam (a
  amostra de cada lado para os bytes por linha, as chaves nulas, a chave no começo de cada row group) rodam juntas,
  em paralelo, em vez de uma depois da outra (em 1M x 50, o planejamento levava ~0,14 s de 0,59 s).
- `tools/diag_profile.py`: o cProfile de uma comparação pela API, com as etapas de cada execução.
- O planejamento faz uma rodada só de consultas ao Polars: a chave no começo de cada row group e a faixa de chaves
  de cada arquivo não dependem do tamanho das janelas e vão junto com a amostra, as chaves nulas e as linhas.
- As colunas da chave de cada janela numa exportação só do Polars por lado (`align_frames`), em vez de uma por
  coluna.
- Amostras montadas a partir de listas do Python (uma conversão por lado), em vez de indexar uma Series do Polars por
  valor (cada indexação passa por uma consulta).
- `tools/check.py --then "bench_hash --cases 1000000x50" diag_profile`: cada ferramenta entre aspas leva os próprios
  argumentos.
- Resultado (fase 2, caminho exato, `memory_limit="1GB"`): 1M x 50 de 1,39x para 1,05x (ok) e o pico de 801 para
  724 MiB; 1M x 10 0,81x, 1M x 300 0,81x, 10M x 10 0,96x, 10M x 300 0,66x (32,9 s), todos ok; 10M x 50 1,20x,
  inconclusivo. Contagens certas em todos, também no caminho coluna a coluna (fase 3).
- Bytes por linha: os tipos de largura fixa (inteiros, reais, datas, Decimal, booleano) pelo tipo; só as colunas de
  largura variável (texto, Categorical) vão para a amostra. A amostra lia o primeiro row group inteiro de todas as
  colunas: em 1M x 50, ~50 ms dos ~80 ms do planejamento.
- `execution.window_bytes` pelos bytes por linha do plano, sem o `estimated_size` do Polars (alguns ms por janela,
  segurando o GIL enquanto a comparação roda).
- Contagem por coluna nos pares divergentes: `dc_compare_pairs` (ABI 8) compara todas as colunas numa chamada,
  lendo direto dos pedaços de memória da janela que o hash de linha já leu (sem reunir as linhas no Polars, cujas
  threads estão ocupadas lendo a janela seguinte); antes, eram uma `dc_compare_column` e duas exportações do Polars
  por coluna (em 1M x 50, ~0,12 s por comparação). `dc_compare_rows` (`compare_frames`) faz o mesmo com as linhas
  já reunidas.
- Amostras: os pares divergentes vão para a ordem da chave uma vez por janela, e as amostras de cada coluna saem das
  linhas já reunidas para a contagem (as primeiras em que ela diverge); antes, um `select`, um `sort` e um `head` do
  Polars por coluna (em 1M x 50 com `samples=5`, ~0,13 s).

