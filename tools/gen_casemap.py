"""gen_casemap.py -- gera cpp/casemap.inc: as tabelas de minúsculas e maiúsculas do Unicode para o ``lower()`` e o
``upper()`` do dc.expr (cpp/expr.cpp), tiradas do ``str.lower``/``str.upper`` do Python (mapeamento completo, como o
``to_lowercase``/``to_uppercase`` do Rust que o Polars usa: "ß".upper() == "SS").

    uv run python tools/gen_casemap.py           # regenera
    uv run python tools/gen_casemap.py --check   # só confere

Cada tabela: os códigos que mudam (em ordem, para busca binária) e, para cada um, de 1 a 3 códigos de saída. O
sigma final (Σ no fim de palavra vira ς) é tratado no C++, à parte.
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cpp" / "casemap.inc"


def _table(name: str, convert: str) -> list[str]:
    keys, outs = [], []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        ch = chr(cp)
        to = getattr(ch, convert)()
        if to != ch:
            if cp == 0x3A3 and convert == "lower":
                to = "σ"  # o sigma final depende do contexto: o C++ decide
            codes = [ord(c) for c in to]
            assert len(codes) <= 3, (hex(cp), codes)
            keys.append(cp)
            outs.append(codes + [0] * (3 - len(codes)))
    lines = [f"static const uint32_t {name}_keys[{len(keys)}] = {{"]
    lines.extend("    " + ", ".join(f"0x{k:X}" for k in keys[i : i + 12]) + "," for i in range(0, len(keys), 12))
    lines.append("};")
    lines.append(f"static const uint32_t {name}_vals[{len(keys)}][3] = {{")
    flat = [f"{{0x{a:X}, 0x{b:X}, 0x{c:X}}}" for a, b, c in outs]
    lines.extend("    " + ", ".join(flat[i : i + 6]) + "," for i in range(0, len(flat), 6))
    lines.append("};")
    lines.append(f"static const size_t {name}_count = {len(keys)};")
    return lines


def render() -> str:
    head = [
        "// casemap.inc -- gerado por tools/gen_casemap.py (não edite à mão).",
        f"// Unicode {unicodedata.unidata_version}, de str.lower/str.upper do Python {sys.version.split()[0]}.",
        "#pragma once",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
    ]
    return "\n".join([*head, *_table("dc_lower", "lower"), "", *_table("dc_upper", "upper"), ""])


def main() -> int:
    text = render()
    if "--check" in sys.argv:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        version_line = current.split("\n")[1] if current.count("\n") >= 1 else ""
        if current and f"Unicode {unicodedata.unidata_version}," not in version_line:
            print(f"pulado: o Python que roda tem outro Unicode ({unicodedata.unidata_version}) que o da tabela")
            return 0
        # a versão do Python na linha 2 pode mudar; o resto (as tabelas) não pode
        same = current.split("\n", 2)[2:] == text.split("\n", 2)[2:]
        print("ok" if same else f"{OUT.name} desatualizado: rode tools/gen_casemap.py")
        return 0 if same else 1
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"gerado {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
