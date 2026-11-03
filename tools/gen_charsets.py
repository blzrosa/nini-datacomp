"""gen_charsets.py -- gera cpp/third_party/shim/charsets.inc: as tabelas das codificações de 1 byte (byte -> Unicode)
que o iconv próprio do nini-datacomp converte para UTF-8 (cpp/third_party/shim/iconv.c).

    python tools/gen_charsets.py          # regenera o arquivo
    python tools/gen_charsets.py --check  # só confere se está atualizado

As tabelas vêm dos codecs do próprio Python (os mesmos em qualquer sistema), então a conversão dá o mesmo resultado
no Windows, no Linux e no macOS, sem depender da libiconv. Os nomes são os que o ReadStat pede (tabelas de
codificação do SAS e do SPSS) mais apelidos comuns (latin1, cp1252, wlatin1...). Codificações de vários bytes (CP932,
GB18030, Big5, EUC-*) ficam de fora: o iconv_open falha e o ReadStat avisa que a codificação não é suportada.
"""

from __future__ import annotations

import codecs
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cpp" / "third_party" / "shim" / "charsets.inc"

# nome que o ReadStat usa -> codec do Python
CHARSETS = {
    "US-ASCII": "ascii",
    **{f"ISO-8859-{n}": f"iso8859_{n}" for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16)},
    **{f"WINDOWS-{n}": f"cp{n}" for n in range(1250, 1259)},
    **{
        f"CP{n}": f"cp{n}"
        for n in (437, 720, 737, 775, 850, 852, 855, 857, 858, 860, 861, 862, 863, 864, 865, 866, 869, 874)
    },
    "IBM037": "cp037",
    "KOI8-R": "koi8_r",
    "KOI8-U": "koi8_u",
    "MACROMAN": "mac_roman",
    "MACCENTRALEUROPE": "mac_latin2",
    "MACCROATIAN": "mac_croatian",
    "MACCYRILLIC": "mac_cyrillic",
    "MACUKRAINE": "mac_cyrillic",  # o MacUkrainian difere do MacCyrillic em 2 posições; o Python não tem o outro
    "MACGREEK": "mac_greek",
    "MACICELAND": "mac_iceland",
    "MACROMANIA": "mac_romanian",
    "MACTURKISH": "mac_turkish",
    "MACARABIC": "mac_arabic",
}
# apelidos -> nome do ReadStat (comparados sem maiúsculas/minúsculas, hífens, sublinhados nem espaços)
ALIASES = {
    "ASCII": "US-ASCII",
    "ANSI_X3.4-1968": "US-ASCII",
    "LATIN1": "ISO-8859-1",
    "L1": "ISO-8859-1",
    "LATIN2": "ISO-8859-2",
    "LATIN3": "ISO-8859-3",
    "LATIN4": "ISO-8859-4",
    "CYRILLIC": "ISO-8859-5",
    "ARABIC": "ISO-8859-6",
    "ASMO-708": "ISO-8859-6",
    "GREEK": "ISO-8859-7",
    "HEBREW": "ISO-8859-8",
    "LATIN5": "ISO-8859-9",
    "LATIN6": "ISO-8859-10",
    "THAI": "ISO-8859-11",
    "LATIN7": "ISO-8859-13",
    "LATIN8": "ISO-8859-14",
    "LATIN9": "ISO-8859-15",
    "LATIN10": "ISO-8859-16",
    **{f"CP{n}": f"WINDOWS-{n}" for n in range(1250, 1259)},
    **{f"MS{n}": f"WINDOWS-{n}" for n in range(1250, 1259)},
    # os nomes do SAS (ENCODING=)
    "WLATIN1": "WINDOWS-1252",
    "WLATIN2": "WINDOWS-1250",
    "WCYRILLIC": "WINDOWS-1251",
    "WGREEK": "WINDOWS-1253",
    "WTURKISH": "WINDOWS-1254",
    "WHEBREW": "WINDOWS-1255",
    "WARABIC": "WINDOWS-1256",
    "WBALTIC": "WINDOWS-1257",
    "WVIETNAMESE": "WINDOWS-1258",
    "PCOEM437": "CP437",
    "PCOEM850": "CP850",
    "PCOEM852": "CP852",
    "PCOEM860": "CP860",
    "PCOEM865": "CP865",
    **{f"IBM{n}": f"CP{n}" for n in (437, 737, 775, 850, 852, 855, 857, 858, 860, 861, 862, 863, 864, 865, 866, 869)},
    "MAC": "MACROMAN",
    "MACINTOSH": "MACROMAN",
}
UNDEFINED = 0xFFFF


def norm(name: str) -> str:
    return "".join(ch for ch in name.upper() if ch.isalnum())


def table(codec: str) -> list[int]:
    out = []
    for b in range(256):
        try:
            s = bytes([b]).decode(codec)
        except UnicodeDecodeError:
            out.append(UNDEFINED)
            continue
        if len(s) != 1 or ord(s) > 0xFFFE:
            raise ValueError(f"{codec}: byte {b:#x} não vira um só caractere do plano básico")
        out.append(ord(s))
    return out


def render() -> str:
    tables: dict[str, list[int]] = {}
    names: list[tuple[str, str]] = []
    for name, codec in CHARSETS.items():
        try:
            codecs.lookup(codec)
        except LookupError:  # pragma: no cover - depende da versão do Python
            continue
        tables.setdefault(codec, table(codec))
        names.append((norm(name), codec))
    known = {n for n, _ in names}
    for alias, name in ALIASES.items():
        target = dict(names).get(norm(name))
        if target is None or norm(alias) in known:
            continue
        names.append((norm(alias), target))
        known.add(norm(alias))
    lines = [
        "/* charsets.inc -- GERADO por tools/gen_charsets.py (tabelas dos codecs do Python): não edite à mão.",
        f" * Byte -> ponto de código Unicode; 0x{UNDEFINED:04X} = byte sem caractere na codificação. */",
        "",
    ]
    ident = {codec: "dc_cs_" + codec.replace("-", "_") for codec in tables}
    for codec, values in tables.items():
        lines.append(f"static const uint16_t {ident[codec]}[256] = {{")
        lines.extend("    " + ", ".join(f"0x{v:04X}" for v in values[i : i + 16]) + "," for i in range(0, 256, 16))
        lines.append("};")
    lines += ["", "/* nome normalizado (só letras e dígitos, em maiúsculas) -> tabela */", "static const struct {"]
    lines += ["    const char *name;", "    const uint16_t *table;", "} dc_charsets[] = {"]
    lines += [f'    {{"{n}", {ident[c]}}},' for n, c in sorted(names)]
    lines += ["};", ""]
    return "\n".join(lines)


def main() -> int:
    text = render()
    if "--check" in sys.argv:
        if not OUT.is_file() or OUT.read_text(encoding="utf-8") != text:
            print(f"{OUT.relative_to(ROOT)} desatualizado: rode python tools/gen_charsets.py")
            return 1
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"gerado {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
