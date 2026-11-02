"""gen_bindings.py -- gera cpp/datacomp.h e src/nini_datacomp/_signatures.py a partir das funções `DC_EXPORT` dos .cpp.

    uv run python tools/gen_bindings.py          # regenera os dois arquivos
    uv run python tools/gen_bindings.py --check  # só confere se estão atualizados (usado no CI e nos testes)

Uma fonte só para a ABI: a assinatura escrita no .cpp. O header documenta a ABI em C; o _signatures.py diz ao ctypes
os tipos de cada argumento (um tipo errado no ctypes vira corrupção de memória silenciosa, não erro).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPP = ROOT / "cpp"
HEADER = CPP / "datacomp.h"
SIGS = ROOT / "src" / "nini_datacomp" / "_signatures.py"
ABI_VERSION = 5

C_TYPES = {"u64": "uint64_t", "u32": "uint32_t", "u16": "uint16_t", "u8": "uint8_t", "i64": "int64_t", "i32": "int32_t"}
CTYPES = {
    "int": "c_int",
    "uint64_t": "c_uint64",
    "uint32_t": "c_uint32",
    "uint16_t": "c_uint16",
    "uint8_t": "c_uint8",
    "int64_t": "c_int64",
    "int32_t": "c_int32",
    "double": "c_double",
    "float": "c_float",
}


def exported() -> list[tuple[str, str, str, list[str]]]:
    """[(arquivo, retorno, nome, [argumentos em C, ex. 'const uint32_t* V'])] na ordem dos arquivos."""
    out = []
    for f in sorted(CPP.glob("*.cpp")):
        text = f.read_text(encoding="utf-8")
        for m in re.finditer(r"DC_EXPORT (int|void|i64|double) (dc_\w+)\(((?:[^{])*?)\)\s*\{", text, flags=re.DOTALL):
            args = [_c(a.strip()) for a in re.sub(r"\s+", " ", m.group(3)).split(",") if a.strip()]
            out.append((f.name, _c(m.group(1)), m.group(2), args))
    return out


def _c(arg: str) -> str:
    """Troca os apelidos do C++ (u32...) pelos tipos de <stdint.h>."""
    for k, v in C_TYPES.items():
        arg = re.sub(rf"\b{k}\b", v, arg)
    return arg


def header(funcs) -> str:
    h = [
        "// datacomp.h -- ABI em C do nini-datacomp, chamada pelo Python via ctypes (src/nini_datacomp/_native.py).",
        "// GERADO por tools/gen_bindings.py a partir dos .cpp: não edite à mão. Os kernels leem os buffers do Arrow",
        "// entregues pelo Polars (sem cópia) e escrevem em buffers alocados pelo chamador.",
        "// Mudou a assinatura de alguma função? Aumente ABI_VERSION em tools/gen_bindings.py e regenere.",
        "#pragma once",
        "#include <stdint.h>",
        "",
        f"#define DC_ABI_VERSION {ABI_VERSION}",
        "",
        "// DC_API: exportada ao compilar a biblioteca (DC_BUILDING), importada ao usá-la de outro código C/C++",
        "#if defined(_WIN32)",
        "#  if defined(DC_BUILDING)",
        "#    define DC_API __declspec(dllexport)",
        "#  else",
        "#    define DC_API __declspec(dllimport)",
        "#  endif",
        "#else",
        '#  define DC_API __attribute__((visibility("default")))',
        "#endif",
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
    ]
    cur = None
    for f, ret, name, args in funcs:
        if f != cur:
            h.append(f"\n/* {f} */")
            cur = f
        h.append(f"DC_API {ret} {name}({', '.join(args)});")
    return "\n".join([*h, "", "#ifdef __cplusplus", "}", "#endif", ""])


def ctype_of(arg: str) -> str:
    t = re.sub(r"\s*\b\w+$", "", arg.strip())  # 'const uint32_t* V' -> 'const uint32_t*'
    if "*" in t:
        return "c_void_p"
    t = t.replace("const ", "").strip()
    return CTYPES[t]


def signatures(funcs) -> str:
    doc = '"""Tipos dos argumentos de cada função da biblioteca C++ (GERADO por tools/gen_bindings.py; não edite)."""'
    imp = "from ctypes import c_double, c_float, c_int, c_int64, c_uint8, c_uint16, c_uint32, c_uint64, c_void_p"
    lines = [
        doc,
        "from __future__ import annotations",
        "",
        imp,
        "",
        f"ABI_VERSION = {ABI_VERSION}",
        "",
        "# nome: (tipo de retorno, [tipos dos argumentos])",
        "SIGNATURES = {",
    ]
    for _, ret, name, args in funcs:
        r = {"int": "c_int", "int64_t": "c_int64", "double": "c_double", "void": "None"}[ret]
        lines.append(f'    "{name}": ({r}, [{", ".join(ctype_of(a) for a in args)}]),')
    lines += [
        "}",
        "",
        (
            "__all__ = ['ABI_VERSION', 'SIGNATURES', 'c_double', 'c_float', 'c_int', 'c_int64', 'c_uint8', "
            "'c_uint16', 'c_uint32', 'c_uint64', 'c_void_p']"
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    funcs = exported()
    want = {HEADER: header(funcs), SIGS: signatures(funcs)}
    if "--check" in sys.argv:
        stale = [p.name for p, t in want.items() if not p.exists() or p.read_text(encoding="utf-8") != t]
        if stale:
            print(f"desatualizados: {stale} (rode: uv run python tools/gen_bindings.py)")
            return 1
        print(f"ok: {len(funcs)} funções")
        return 0
    for p, t in want.items():
        p.write_text(t, encoding="utf-8", newline="\n")
    print(f"{len(funcs)} funções -> {HEADER.relative_to(ROOT)}, {SIGS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
