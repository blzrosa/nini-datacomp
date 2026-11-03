/* zlib.h -- o pedaço do zlib que o ReadStat usa para ler .zsav (só uncompress), implementado com o tinfl do miniz
 * (third_party/miniz, licença MIT) em zlib_shim.c. Os nomes ganham o prefixo dc_ para não colidir com um zlib de
 * verdade carregado no mesmo processo. */
#pragma once
#include <stddef.h>

typedef unsigned char Bytef;
typedef unsigned long uLong;
typedef unsigned long uLongf;

#define Z_OK 0
#define Z_BUF_ERROR (-5)
#define Z_DATA_ERROR (-3)

#define uncompress dc_zshim_uncompress
int uncompress(Bytef *dest, uLongf *destLen, const Bytef *source, uLong sourceLen);
