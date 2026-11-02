# Changelog

Formato: [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/); versões: [SemVer](https://semver.org/lang/pt-BR/).

## [Não lançado]

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
