/* iconv.h -- o iconv que o ReadStat usa, implementado pelo nini-datacomp (iconv.c) em vez da libiconv/glibc: o
 * mesmo resultado em qualquer sistema e nada a instalar no Windows. Converte para UTF-8 a partir de UTF-8 e das
 * codificações de 1 byte (Latin-1, Windows-125x, CP437/850..., ISO-8859-*, Mac*) em charsets.inc. Os nomes ganham
 * o prefixo dc_ para não colidir com o iconv do sistema carregado no mesmo processo. */
#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void *iconv_t;

#define iconv_open dc_iconv_open
#define iconv dc_iconv
#define iconv_close dc_iconv_close

iconv_t iconv_open(const char *tocode, const char *fromcode);
size_t iconv(iconv_t cd, char **inbuf, size_t *inbytesleft, char **outbuf, size_t *outbytesleft);
int iconv_close(iconv_t cd);

/* Para o próprio nini-datacomp: o nome normalizado reconhecido? (1 = sim; "UTF-8" também conta) */
int dc_iconv_supported(const char *name);

#ifdef __cplusplus
}
#endif
