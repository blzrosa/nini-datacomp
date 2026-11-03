/* zlib_shim.c -- uncompress() do zlib (formato zlib: cabeçalho, deflate e Adler-32) com o tinfl do miniz. */
#include "zlib.h"
#include "../miniz/miniz_tinfl.h"

int uncompress(Bytef *dest, uLongf *destLen, const Bytef *source, uLong sourceLen) {
    size_t n = tinfl_decompress_mem_to_mem(dest, (size_t)*destLen, source, (size_t)sourceLen,
                                           TINFL_FLAG_PARSE_ZLIB_HEADER | TINFL_FLAG_COMPUTE_ADLER32);
    if (n == TINFL_DECOMPRESS_MEM_TO_MEM_FAILED) return Z_DATA_ERROR;
    *destLen = (uLongf)n;
    return Z_OK;
}
