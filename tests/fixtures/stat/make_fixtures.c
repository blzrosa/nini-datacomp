/* make_fixtures.c -- gera os arquivos de teste de SAS, Stata e SPSS (tests/fixtures/stat/) com os escritores do
 * ReadStat. Os arquivos gerados vão para o repositório; este programa só precisa rodar de novo se a tabela mudar.
 *
 * Compilar (Linux, com o ReadStat completo de https://github.com/WizardMac/ReadStat em $RS, mais iconv e zlib do
 * sistema):
 *
 *   gcc -O1 -DHAVE_ZLIB=1 -I$RS/src make_fixtures.c $(find $RS/src -name '*.c' -not -path '*bin*' \
 *       -not -path '*test*' -not -path '*fuzz*' -not -name '*_txt_*') -lz -lm -o make_fixtures
 *   ./make_fixtures tests/fixtures/stat
 *
 * A mesma tabela em todos os formatos (os testes em tests/test_stat.py conferem os valores):
 *
 *   id  valor   nome      data        momento                  hora
 *   1   1.5     Ana       2024-01-31  2024-01-31 12:34:56      01:00:00
 *   2   (nulo)  (vazio)   (nulo)      1960-01-01 00:00:00      00:00:00
 *   3   -3.25   José      1960-01-01  (nulo)                   23:59:59
 *   4   1e10    "  lead"  1959-12-31  1999-12-31 23:59:59.5    (nulo)
 *   5   0.1     Zoë       2000-02-29  2024-02-29 00:00:01      12:34:56
 *
 * No Stata não há coluna de hora (o formato não tem esse tipo); no lugar entram `flag` (int8: 0, 1, nulo, -5, 100) e
 * `taxa` (float: 0.5, nulo, 2.25, -1, 3). O .por tem os nomes das colunas em maiúsculas e o texto sem acentos (o
 * formato só aceita isso). latin1.xpt tem só id e nome, com o texto gravado em Latin-1.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "readstat.h"

static const char *DIR;

static ssize_t write_bytes(const void *data, size_t len, void *ctx) {
    return (ssize_t)fwrite(data, 1, len, (FILE *)ctx);
}

#define CHECK(x)                                                                     \
    do {                                                                             \
        readstat_error_t _e = (x);                                                   \
        if (_e != READSTAT_OK) {                                                     \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, readstat_error_message(_e)); \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

static const double ID[5] = {1, 2, 3, 4, 5};
static const double VALOR[5] = {1.5, NAN, -3.25, 1e10, 0.1};
static const char *NOME[5] = {"Ana", "", "Jos\xc3\xa9", "  lead", "Zo\xc3\xab"};
static const char *NOME_ASCII[5] = {"Ana", "", "Jose", "  lead", "Zoe"}; /* o .por só grava ASCII */
static const double SAS_DAYS[5] = {23406, NAN, 0, -1, 14669};
static const double SAS_SECS[5] = {2022323696.0, 0.0, NAN, 1262303999.5, 2024784001.0};
static const double SPSS_DATE[5] = {13926038400.0, NAN, 11903760000.0, 11903673600.0, 13171161600.0};
static const double SPSS_DT[5] = {13926083696.0, 11903760000.0, NAN, 13166063999.5, 13928544001.0};
static const double HORA[5] = {3600, 0, 86399, NAN, 45296};

static FILE *open_out(const char *name) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", DIR, name);
    FILE *f = fopen(path, "wb");
    if (!f) {
        perror(path);
        exit(1);
    }
    return f;
}

static void put_double(readstat_writer_t *w, readstat_variable_t *v, double x) {
    if (isnan(x)) CHECK(readstat_insert_missing_value(w, v));
    else CHECK(readstat_insert_double_value(w, v, x));
}

/* SAS (.sas7bdat e .xpt) e SPSS (.sav, .zsav, .por): id, valor, nome, data, momento, hora */
static void write_common(const char *name, int spss, int sav_compress, int upper,
                         readstat_error_t (*begin)(readstat_writer_t *, void *, long)) {
    FILE *f = open_out(name);
    readstat_writer_t *w = readstat_writer_init();
    CHECK(readstat_set_data_writer(w, write_bytes));
    if (sav_compress) CHECK(readstat_writer_set_compression(w, READSTAT_COMPRESS_BINARY));
    readstat_variable_t *id = readstat_add_variable(w, upper ? "ID" : "id", READSTAT_TYPE_DOUBLE, 0);
    readstat_variable_t *valor = readstat_add_variable(w, upper ? "VALOR" : "valor", READSTAT_TYPE_DOUBLE, 0);
    readstat_variable_t *nome = readstat_add_variable(w, upper ? "NOME" : "nome", READSTAT_TYPE_STRING, 20);
    readstat_variable_t *data = readstat_add_variable(w, upper ? "DATA" : "data", READSTAT_TYPE_DOUBLE, 0);
    readstat_variable_t *momento = readstat_add_variable(w, upper ? "MOMENTO" : "momento", READSTAT_TYPE_DOUBLE, 0);
    readstat_variable_t *hora = readstat_add_variable(w, upper ? "HORA" : "hora", READSTAT_TYPE_DOUBLE, 0);
    if (spss) {
        readstat_variable_set_format(id, "F8.0");
        readstat_variable_set_format(valor, "F12.4");
        readstat_variable_set_format(nome, "A20");
        readstat_variable_set_format(data, "DATE11");
        readstat_variable_set_format(momento, "DATETIME22.1");
        readstat_variable_set_format(hora, "TIME8");
    } else {
        readstat_variable_set_format(id, "BEST12.");
        readstat_variable_set_format(nome, "$20.");
        readstat_variable_set_format(data, "DATE9.");
        readstat_variable_set_format(momento, "DATETIME22.1");
        readstat_variable_set_format(hora, "TIME8.");
    }
    CHECK(begin(w, f, 5));
    for (int i = 0; i < 5; i++) {
        CHECK(readstat_begin_row(w));
        put_double(w, id, ID[i]);
        put_double(w, valor, VALOR[i]);
        CHECK(readstat_insert_string_value(w, nome, upper ? NOME_ASCII[i] : NOME[i]));
        put_double(w, data, spss ? SPSS_DATE[i] : SAS_DAYS[i]);
        put_double(w, momento, spss ? SPSS_DT[i] : SAS_SECS[i]);
        put_double(w, hora, HORA[i]);
        CHECK(readstat_end_row(w));
    }
    CHECK(readstat_end_writing(w));
    readstat_writer_free(w);
    fclose(f);
}

static readstat_error_t begin_xpt8(readstat_writer_t *w, void *ctx, long rows) {
    CHECK(readstat_writer_set_file_format_version(w, 8));
    return readstat_begin_writing_xport(w, ctx, rows);
}

static void write_dta(void) {
    static const double FLAG[5] = {0, 1, NAN, -5, 100};
    static const double TAXA[5] = {0.5, NAN, 2.25, -1, 3};
    FILE *f = open_out("sample.dta");
    readstat_writer_t *w = readstat_writer_init();
    CHECK(readstat_set_data_writer(w, write_bytes));
    CHECK(readstat_writer_set_file_format_version(w, 118));
    readstat_variable_t *id = readstat_add_variable(w, "id", READSTAT_TYPE_INT32, 0);
    readstat_variable_t *valor = readstat_add_variable(w, "valor", READSTAT_TYPE_DOUBLE, 0);
    readstat_variable_t *nome = readstat_add_variable(w, "nome", READSTAT_TYPE_STRING, 20);
    readstat_variable_t *data = readstat_add_variable(w, "data", READSTAT_TYPE_INT32, 0);
    readstat_variable_t *momento = readstat_add_variable(w, "momento", READSTAT_TYPE_DOUBLE, 0);
    readstat_variable_t *flag = readstat_add_variable(w, "flag", READSTAT_TYPE_INT8, 0);
    readstat_variable_t *taxa = readstat_add_variable(w, "taxa", READSTAT_TYPE_FLOAT, 0);
    readstat_variable_set_format(id, "%8.0g");
    readstat_variable_set_format(valor, "%10.0g");
    readstat_variable_set_format(nome, "%20s");
    readstat_variable_set_format(data, "%td");
    readstat_variable_set_format(momento, "%tc");
    readstat_variable_set_format(flag, "%8.0g");
    readstat_variable_set_format(taxa, "%9.0g");
    CHECK(readstat_begin_writing_dta(w, f, 5));
    for (int i = 0; i < 5; i++) {
        CHECK(readstat_begin_row(w));
        CHECK(readstat_insert_int32_value(w, id, (int32_t)ID[i]));
        put_double(w, valor, VALOR[i]);
        CHECK(readstat_insert_string_value(w, nome, NOME[i]));
        if (isnan(SAS_DAYS[i])) CHECK(readstat_insert_missing_value(w, data));
        else CHECK(readstat_insert_int32_value(w, data, (int32_t)SAS_DAYS[i]));
        put_double(w, momento, isnan(SAS_SECS[i]) ? NAN : SAS_SECS[i] * 1000.0);
        if (isnan(FLAG[i])) CHECK(readstat_insert_missing_value(w, flag));
        else CHECK(readstat_insert_int8_value(w, flag, (int8_t)FLAG[i]));
        if (isnan(TAXA[i])) CHECK(readstat_insert_missing_value(w, taxa));
        else CHECK(readstat_insert_float_value(w, taxa, (float)TAXA[i]));
        CHECK(readstat_end_row(w));
    }
    CHECK(readstat_end_writing(w));
    readstat_writer_free(w);
    fclose(f);
}

/* Texto em Latin-1 num .xpt (o XPORT não diz a codificação: o leitor precisa receber encoding="latin1") */
static void write_latin1(void) {
    static const char *L1[3] = {"Jos\xe9", "S\xe3o Paulo", "a\xe7\xfa"};
    FILE *f = open_out("latin1.xpt");
    readstat_writer_t *w = readstat_writer_init();
    CHECK(readstat_set_data_writer(w, write_bytes));
    readstat_variable_t *id = readstat_add_variable(w, "id", READSTAT_TYPE_DOUBLE, 0);
    readstat_variable_t *nome = readstat_add_variable(w, "nome", READSTAT_TYPE_STRING, 20);
    CHECK(readstat_begin_writing_xport(w, f, 3));
    for (int i = 0; i < 3; i++) {
        CHECK(readstat_begin_row(w));
        CHECK(readstat_insert_double_value(w, id, i + 1));
        CHECK(readstat_insert_string_value(w, nome, L1[i]));
        CHECK(readstat_end_row(w));
    }
    CHECK(readstat_end_writing(w));
    readstat_writer_free(w);
    fclose(f);
}

int main(int argc, char **argv) {
    DIR = argc > 1 ? argv[1] : ".";
    write_common("sample.sas7bdat", 0, 0, 0, readstat_begin_writing_sas7bdat);
    write_common("sample.xpt", 0, 0, 0, begin_xpt8);
    write_common("sample.sav", 1, 0, 0, readstat_begin_writing_sav);
    write_common("sample.zsav", 1, 1, 0, readstat_begin_writing_sav);
    write_common("sample.por", 1, 0, 1, readstat_begin_writing_por); /* o .por só aceita nomes em maiúsculas */
    write_dta();
    write_latin1();
    return 0;
}
