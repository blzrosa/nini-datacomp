/* iconv.c -- conversão para UTF-8 (veja iconv.h). Mesma semântica do iconv do POSIX para quem chama:
 *   - devolve (size_t)-1 com errno E2BIG (saída cheia), EILSEQ (byte inválido na entrada) ou EINVAL (sequência
 *     UTF-8 incompleta no fim da entrada); os ponteiros param no primeiro byte não convertido;
 *   - inbuf NULL: volta ao estado inicial (aqui não há estado) e devolve 0.
 */
#include "iconv.h"

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "charsets.inc"

typedef struct {
    const uint16_t *table; /* NULL: a origem já é UTF-8 (só confere as sequências) */
} dc_conv;

static void dc_norm(const char *name, char *out, size_t cap) {
    size_t n = 0;
    for (; name && *name && n + 1 < cap; name++) {
        char c = *name;
        if (c >= 'a' && c <= 'z') c = (char)(c - 'a' + 'A');
        if ((c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) out[n++] = c;
    }
    out[n] = '\0';
}

static int dc_is_utf8(const char *norm) { return strcmp(norm, "UTF8") == 0; }

static const uint16_t *dc_table(const char *norm) {
    size_t i;
    for (i = 0; i < sizeof(dc_charsets) / sizeof(dc_charsets[0]); i++) {
        if (strcmp(dc_charsets[i].name, norm) == 0) return dc_charsets[i].table;
    }
    return NULL;
}

int dc_iconv_supported(const char *name) {
    char norm[64];
    dc_norm(name, norm, sizeof norm);
    return dc_is_utf8(norm) || dc_table(norm) != NULL;
}

iconv_t iconv_open(const char *tocode, const char *fromcode) {
    char to[64], from[64];
    const uint16_t *table = NULL;
    dc_conv *cd;
    dc_norm(tocode, to, sizeof to);
    dc_norm(fromcode, from, sizeof from);
    if (!dc_is_utf8(to)) {
        errno = EINVAL;
        return (iconv_t)-1;
    }
    if (!dc_is_utf8(from)) {
        table = dc_table(from);
        if (table == NULL) {
            errno = EINVAL;
            return (iconv_t)-1;
        }
    }
    cd = (dc_conv *)malloc(sizeof *cd);
    if (cd == NULL) {
        errno = ENOMEM;
        return (iconv_t)-1;
    }
    cd->table = table;
    return (iconv_t)cd;
}

int iconv_close(iconv_t cd) {
    if (cd == (iconv_t)-1 || cd == NULL) {
        errno = EBADF;
        return -1;
    }
    free(cd);
    return 0;
}

/* Comprimento da sequência UTF-8 que começa em s (com no máximo n bytes): 1..4; 0 = inválida; -1 = incompleta. */
static int dc_utf8_len(const unsigned char *s, size_t n) {
    unsigned char c = s[0];
    int len, i;
    uint32_t cp;
    if (c < 0x80) return 1;
    if (c >= 0xC2 && c <= 0xDF) len = 2, cp = c & 0x1F;
    else if (c >= 0xE0 && c <= 0xEF) len = 3, cp = c & 0x0F;
    else if (c >= 0xF0 && c <= 0xF4) len = 4, cp = c & 0x07;
    else return 0;
    for (i = 1; i < len; i++) {
        if ((size_t)i >= n) return -1;
        if ((s[i] & 0xC0) != 0x80) return 0;
        cp = (cp << 6) | (s[i] & 0x3F);
    }
    if ((len == 3 && cp < 0x800) || (len == 4 && (cp < 0x10000 || cp > 0x10FFFF))) return 0; /* forma longa */
    if (cp >= 0xD800 && cp <= 0xDFFF) return 0;                                              /* surrogate */
    return len;
}

size_t iconv(iconv_t handle, char **inbuf, size_t *inbytesleft, char **outbuf, size_t *outbytesleft) {
    dc_conv *cd = (dc_conv *)handle;
    const unsigned char *in;
    unsigned char *out;
    size_t inleft, outleft;
    if (cd == NULL || cd == (dc_conv *)-1) {
        errno = EBADF;
        return (size_t)-1;
    }
    if (inbuf == NULL || *inbuf == NULL) return 0;
    in = (const unsigned char *)*inbuf;
    out = (unsigned char *)*outbuf;
    inleft = *inbytesleft;
    outleft = *outbytesleft;
    while (inleft > 0) {
        unsigned char buf[3];
        size_t need, consumed;
        if (cd->table == NULL) {
            int len = dc_utf8_len(in, inleft);
            if (len <= 0) {
                errno = len == 0 ? EILSEQ : EINVAL;
                goto fail;
            }
            need = consumed = (size_t)len;
            if (outleft < need) {
                errno = E2BIG;
                goto fail;
            }
            memcpy(out, in, need);
        } else {
            uint16_t cp = cd->table[in[0]];
            if (cp == 0xFFFF) {
                errno = EILSEQ;
                goto fail;
            }
            consumed = 1;
            if (cp < 0x80) buf[0] = (unsigned char)cp, need = 1;
            else if (cp < 0x800) buf[0] = (unsigned char)(0xC0 | (cp >> 6)), buf[1] = (unsigned char)(0x80 | (cp & 0x3F)), need = 2;
            else
                buf[0] = (unsigned char)(0xE0 | (cp >> 12)), buf[1] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F)),
                buf[2] = (unsigned char)(0x80 | (cp & 0x3F)), need = 3;
            if (outleft < need) {
                errno = E2BIG;
                goto fail;
            }
            memcpy(out, buf, need);
        }
        in += consumed;
        inleft -= consumed;
        out += need;
        outleft -= need;
    }
    *inbuf = (char *)in;
    *inbytesleft = inleft;
    *outbuf = (char *)out;
    *outbytesleft = outleft;
    return 0;
fail:
    *inbuf = (char *)in;
    *inbytesleft = inleft;
    *outbuf = (char *)out;
    *outbytesleft = outleft;
    return (size_t)-1;
}
