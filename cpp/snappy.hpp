// snappy.hpp -- descompressor de snappy (formato bruto, sem moldura) usado pelo leitor de Parquet.
#pragma once
#include "common.hpp"

namespace dc {

// Tamanho descomprimido declarado no começo do bloco (varint). false se o cabeçalho for inválido.
bool snappy_uncompressed_length(const u8* src, i64 n, i64* out);

// Descomprime `n` bytes de `src` para exatamente `dst_len` bytes em `dst`. false se o bloco estiver corrompido,
// declarar outro tamanho que `dst_len`, ou mandar ler/escrever fora dos buffers (nada é escrito além de `dst_len`).
bool snappy_decompress(const u8* src, i64 n, u8* dst, i64 dst_len);

}  // namespace dc
