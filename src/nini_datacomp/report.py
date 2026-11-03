"""Relatório de uma comparação, em texto, Markdown, HTML ou JSON (portado do ``report.py`` do datacompolars 0.4.4).

O conteúdo é montado uma vez só, num modelo neutro de blocos (`build_blocks`); cada formato só decide a apresentação.
Texto e Markdown respeitam ``max_columns`` e ``max_sample_columns``; HTML e JSON trazem tudo.
"""

from __future__ import annotations

import html as _html
import math
import sys
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from .results import ComparisonResult

__all__ = ["Format", "ReportOptions", "build_blocks", "render"]

Format = Literal["text", "markdown", "html", "json"]
Style = Literal["auto", "unicode", "ascii"]
_SUFFIXES: dict[str, Format] = {".md": "markdown", ".markdown": "markdown", ".html": "html", ".htm": "html"}
_SUFFIXES |= {".json": "json", ".txt": "text"}


@dataclass(frozen=True)
class ReportOptions:
    width: int = 100  # largura do texto (60 a 200)
    style: Style = "auto"  # bordas e barras do texto: "auto" usa ASCII se o console não tem Unicode
    max_columns: int = 20  # colunas com divergência listadas no texto e no Markdown
    max_sample_columns: int = 10  # colunas com tabela de amostras no texto e no Markdown

    def __post_init__(self) -> None:
        if not 60 <= self.width <= 200:
            raise ValueError("width deve ficar entre 60 e 200")
        if self.style not in ("auto", "unicode", "ascii"):
            raise ValueError(f"style desconhecido: {self.style!r} (use 'auto', 'unicode' ou 'ascii')")
        if self.max_columns < 1 or self.max_sample_columns < 0:
            raise ValueError("max_columns deve ser >= 1 e max_sample_columns >= 0")


def resolve_format(fmt: str | None, path: str | Path | None) -> Format:
    """O formato pedido, ou o da extensão de ``path`` (.md, .html, .json, .txt), ou texto."""
    if fmt is not None:
        if fmt not in ("text", "markdown", "html", "json"):
            raise ValueError(f"formato desconhecido: {fmt!r} (use 'text', 'markdown', 'html' ou 'json')")
        return cast("Format", fmt)
    if path is not None:
        return _SUFFIXES.get(Path(path).suffix.lower(), "text")
    return "text"


def render(result: ComparisonResult, fmt: Format, options: ReportOptions) -> str:
    return _RENDERERS[fmt](result, options)


# ----------------------------------------------------------------------------- modelo neutro
@dataclass(frozen=True)
class Bar:
    frac: float


Cell = str | int | float | bool | Bar | None


@dataclass(frozen=True)
class Heading:
    text: str
    level: int = 2


@dataclass(frozen=True)
class Verdict:
    level: Literal["ok", "diff", "error"]
    text: str
    detail: str = ""


@dataclass(frozen=True)
class Facts:
    items: list[tuple[str, str]]


@dataclass(frozen=True)
class Lines:
    lines: list[str]
    kind: Literal["plain", "note", "warn", "error"] = "plain"


@dataclass(frozen=True)
class Table:
    headers: list[str]
    rows: list[list[Cell]]
    align: list[str]  # "l" | "r" | "c" por coluna
    caption: str = ""


Block = Heading | Verdict | Facts | Lines | Table


# ----------------------------------------------------------------------------- formatação de valores
def _n(x: int) -> str:
    return f"{x:,}"


def _pct(frac: float) -> str:
    if frac <= 0:
        return "0%"
    if frac >= 1:
        return "100%"
    p = frac * 100
    return "<0.01%" if p < 0.01 else f"{p:.2f}%"


def _dur(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.0f}s"


def _bytes(n: int) -> str:
    for unit, size in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if n >= size:
            return f"{n / size:.1f} {unit}"
    return f"{n} B"


def _value(v: Any, limit: int) -> str:
    if v is None:
        return "null"
    if isinstance(v, float):
        return "NaN" if math.isnan(v) else f"{v:.6g}"
    s = str(v).replace("\n", " ").replace("\r", " ")
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _names(names: Sequence[str], limit: int = 20) -> str:
    shown = ", ".join(names[:limit])
    return shown + (f" (+{len(names) - limit} more)" if len(names) > limit else "")


# ----------------------------------------------------------------------------- conteúdo
def build_blocks(r: ComparisonResult, options: ReportOptions, *, full: bool = False) -> list[Block]:
    """O relatório em blocos. ``full``: todas as colunas e amostras (HTML), sem os limites do texto."""
    left, right = r.names
    b: list[Block] = [
        Heading("nini-datacomp Comparison Report", 1),
        Facts(
            [
                ("Compared", f"{left}  vs  {right}"),
                ("Join key(s)", ", ".join(r.join_columns)),
                ("Generated", r.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")),
            ]
        ),
        _verdict(r),
    ]
    if r.errors:
        b += [Heading("Errors"), Lines(list(r.errors), "error")]
    if r.warnings:
        b += [Heading("Warnings"), Lines(list(r.warnings), "warn")]
    if not r.aborted:
        b += _rows(r)
    b += _columns(r)
    b += _differences(r, options, full=full)
    b += _samples(r, options, full=full)
    b += _execution(r)
    return b


def _verdict(r: ComparisonResult) -> Verdict:
    left, right = r.names
    rows, cols = r.rows, r.columns
    if r.aborted:
        return Verdict("error", "COMPARISON ABORTED", "The row comparison could not be performed.")
    if r.is_match:
        return Verdict("ok", "IDENTICAL", "Same rows, same values and same columns.")
    parts = []
    if rows.mismatched:
        parts.append(f"{_n(rows.mismatched)} of {_n(rows.common)} common rows differ ({_pct(rows.mismatch_rate)})")
    if rows.left_only:
        parts.append(f"{_n(rows.left_only)} rows only in {left}")
    if rows.right_only:
        parts.append(f"{_n(rows.right_only)} rows only in {right}")
    if cols.left_only or cols.right_only:
        parts.append("the column sets differ")
    if cols.type_mismatches:
        parts.append(f"{len(cols.type_mismatches)} columns have different types")
    return Verdict("diff", "DIFFERENCES FOUND", "; ".join(parts) + ".")


def _rows(r: ComparisonResult) -> list[Block]:
    left, right = r.names
    rows = r.rows
    common = rows.common

    def share(x: int) -> str:
        return _pct(x / common) if common else ""

    def bar(x: int) -> Cell:
        return Bar(x / common) if common else ""

    table = Table(
        ["Category", "Rows", "% of common", ""],
        [
            ["In common", _n(common), "", ""],
            ["  equal", _n(rows.matched), share(rows.matched), bar(rows.matched)],
            ["  with differences", _n(rows.mismatched), share(rows.mismatched), bar(rows.mismatched)],
            [f"Only in {left}", _n(rows.left_only), "", ""],
            [f"Only in {right}", _n(rows.right_only), "", ""],
            [f"Total in {left}", _n(rows.left_total), "", ""],
            [f"Total in {right}", _n(rows.right_total), "", ""],
        ],
        ["l", "r", "r", "l"],
    )
    return [Heading("Rows"), table]


def _columns(r: ComparisonResult) -> list[Block]:
    left, right = r.names
    cols = r.columns
    b: list[Block] = [
        Heading("Columns"),
        Facts(
            [
                ("Compared", _n(len(cols.compared))),
                (f"Only in {left}", _n(len(cols.left_only))),
                (f"Only in {right}", _n(len(cols.right_only))),
                ("Different types (skipped)", _n(len(cols.type_mismatches))),
            ]
        ),
    ]
    notes = []
    if cols.left_only:
        notes.append(f"Only in {left}: {_names(cols.left_only)}")
    if cols.right_only:
        notes.append(f"Only in {right}: {_names(cols.right_only)}")
    if notes:
        b.append(Lines(notes, "note"))
    if cols.type_mismatches:
        b.append(
            Table(
                ["Column", f"Type in {left}", f"Type in {right}"],
                [[t.column, t.left, t.right] for t in cols.type_mismatches],
                ["l", "l", "l"],
                "Columns with different types (not compared)",
            )
        )
    return b


def _differences(r: ComparisonResult, options: ReportOptions, *, full: bool) -> list[Block]:
    stats = r.columns.mismatched()
    if not stats:
        if r.rows.mismatched and not r.execution.details_computed:
            note = Lines(["Per-column details were not computed (details=False)."], "note")
            return [Heading("Differences by column"), note]
        return []
    shown = stats if full else stats[: options.max_columns]
    b: list[Block] = [
        Heading("Differences by column"),
        Table(
            ["Column", "Type", "Differences", "% of common", ""],
            [[c.name, c.dtype, _n(c.mismatches or 0), _pct(c.rate or 0.0), Bar(c.rate or 0.0)] for c in shown],
            ["l", "l", "r", "r", "l"],
        ),
    ]
    if len(shown) < len(stats):
        more = f"{len(stats) - len(shown)} more columns with differences"
        more += " (use the JSON or HTML report for the full list)."
        b.append(Lines([more], "note"))
    return b


def _samples(r: ComparisonResult, options: ReportOptions, *, full: bool) -> list[Block]:
    with_samples = [c for c in r.columns.mismatched() if c.samples.height]
    if not with_samples or not (full or options.max_sample_columns > 0):
        return []
    shown = with_samples if full else with_samples[: options.max_sample_columns]
    limit = 80 if full else 30
    b: list[Block] = [Heading("Sample differing rows")]
    for c in shown:
        headers = list(c.samples.columns)
        b.append(
            Table(
                headers,
                [[_value(v, limit) for v in row] for row in c.samples.rows()],
                ["l"] * len(headers),
                f"{c.name}: {_n(c.mismatches or 0)} differing rows, showing {c.samples.height}",
            )
        )
    if len(shown) < len(with_samples):
        b.append(Lines([f"Samples for {len(with_samples) - len(shown)} more columns omitted."], "note"))
    return b


def _execution(r: ComparisonResult) -> list[Block]:
    ex = r.execution
    exact = [c for c in r.columns.compared if c not in ex.tolerance_columns]
    methods = []
    if exact:
        methods.append(f"row hash, exact ({_n(len(exact))} columns)")
    if ex.tolerance_columns:
        methods.append(f"column by column, with tolerance ({_n(len(ex.tolerance_columns))} columns)")
    if ex.verify_hash:
        methods.append(f"hash verified cell by cell ({_n(ex.hash_collisions)} collisions)")
    facts: list[tuple[str, str]] = []
    if methods:
        facts.append(("Method", "; ".join(methods)))
    if ex.windows:
        each = f"  (~{_n(ex.rows_per_window)} rows each)" if ex.rows_per_window else ""
        facts.append(("Execution windows", f"{_n(ex.windows)}{each}"))
    if ex.column_groups > 1:
        facts.append(("Column groups per window", _n(ex.column_groups)))
    if ex.memory_limit:
        facts.append(("Memory limit", f"{_bytes(ex.memory_limit)} (largest window: {_bytes(ex.window_bytes)})"))
    facts.append(("Column details", "computed" if ex.details_computed else "skipped"))
    facts.append(("Total time", _dur(ex.total_seconds)))
    phases = {k: v for k, v in ex.timings.items() if k != "total" and v >= 0.0005}
    if phases:
        facts.append(("Time by phase", ", ".join(f"{k} {_dur(v)}" for k, v in phases.items())))
    versions = [f"nini-datacomp {ex.version}"] if ex.version else []
    versions += [f"Polars {ex.polars_version}"] if ex.polars_version else []
    if versions:
        facts.append(("Versions", ", ".join(versions)))
    return [Heading("Execution"), Facts(facts)]


# ----------------------------------------------------------------------------- texto
@dataclass(frozen=True)
class _Chars:
    h: str
    v: str
    tl: str
    tr: str
    bl: str
    br: str
    tj: str
    bj: str
    lj: str
    rj: str
    x: str
    dh: str
    full: str
    empty: str
    ell: str
    marks: dict[str, str] = field(default_factory=dict)  # ok, diff, error, warn, note


_UNICODE = _Chars(
    "─", "│", "┌", "┐", "└", "┘", "┬", "┴", "├", "┤", "┼", "═", "█", "░", "…",
    {"ok": "✔", "diff": "✘", "error": "✖", "warn": "!", "note": "·", "plain": "·"},
)  # fmt: skip
_ASCII = _Chars(
    "-", "|", "+", "+", "+", "+", "+", "+", "+", "+", "+", "=", "#", ".", "...",
    {"ok": "[OK]", "diff": "[DIFF]", "error": "[ERROR]", "warn": "!", "note": "-", "plain": "-"},
)  # fmt: skip
_BAR_WIDTH = 12


def _use_unicode(style: Style) -> bool:
    if style != "auto":
        return style == "unicode"
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "─█░✔✘✖…·".encode(enc)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def _trunc(s: str, w: int, ell: str) -> str:
    return s if len(s) <= w else s[: max(0, w - len(ell))] + ell


def _pad(s: str, w: int, align: str) -> str:
    if align == "r":
        return s.rjust(w)
    return s.center(w) if align == "c" else s.ljust(w)


def _bar_text(frac: float, full: str, empty: str) -> str:
    k = round(max(0.0, min(1.0, frac)) * _BAR_WIDTH)
    k = max(k, 1) if frac > 0 else 0
    return full * k + empty * (_BAR_WIDTH - k)


def _text_table(t: Table, width: int, cs: _Chars) -> list[str]:
    def cell(c: Cell) -> str:
        if isinstance(c, Bar):
            return _bar_text(c.frac, cs.full, cs.empty)
        return "" if c is None else str(c)

    body = [[cell(c) for c in row] for row in t.rows]
    n = len(t.headers)
    widths = [max([len(t.headers[i])] + [len(row[i]) for row in body]) for i in range(n)]
    budget = width - 2  # recuo
    while sum(widths) + 3 * n + 1 > budget and max(widths) > 8:
        widths[widths.index(max(widths))] -= 1

    def line(left: str, mid: str, right: str) -> str:
        return left + mid.join(cs.h * (w + 2) for w in widths) + right

    def row(cells: Sequence[str], aligns: Sequence[str]) -> str:
        parts = [" " + _pad(_trunc(c, w, cs.ell), w, a) + " " for c, w, a in zip(cells, widths, aligns, strict=True)]
        return cs.v + cs.v.join(parts) + cs.v

    out = [f"  {t.caption}"] if t.caption else []
    out.append("  " + line(cs.tl, cs.tj, cs.tr))
    out.append("  " + row(t.headers, ["l"] * n))
    out.append("  " + line(cs.lj, cs.x, cs.rj))
    out += ["  " + row(r_, t.align) for r_ in body]
    out.append("  " + line(cs.bl, cs.bj, cs.br))
    return out


def _text_block(blk: Block, width: int, cs: _Chars, first: bool) -> list[str]:
    if isinstance(blk, Heading):
        if blk.level == 1:
            return [cs.dh * width, f"  {blk.text}", cs.dh * width]
        return ["", blk.text.upper(), cs.h * len(blk.text)]
    if isinstance(blk, Verdict):
        title = f"{cs.marks[blk.level]} {blk.text}"
        inner = [title, *textwrap.wrap(blk.detail, width - 6)] if blk.detail else [title]
        box_w = max(len(x) for x in inner) + 2
        return [
            "",
            cs.tl + cs.h * box_w + cs.tr,
            *[cs.v + " " + x.ljust(box_w - 1) + cs.v for x in inner],
            cs.bl + cs.h * box_w + cs.br,
        ]
    if isinstance(blk, Facts):
        lw = max(len(k) for k, _ in blk.items)
        indent = "  " + " " * (lw + 2)
        out = [] if first else [""]
        out += [textwrap.fill(f"{k.ljust(lw)}  {v}", width, initial_indent="  ", subsequent_indent=indent)
                for k, v in blk.items]  # fmt: skip
        return out
    if isinstance(blk, Lines):
        mark = cs.marks[blk.kind]
        return [textwrap.fill(ln, width, initial_indent=f"  {mark} ", subsequent_indent="    ") for ln in blk.lines]
    return ["", *_text_table(blk, width, cs)]


def render_text(r: ComparisonResult, options: ReportOptions) -> str:
    unicode_ok = _use_unicode(options.style)
    cs = _UNICODE if unicode_ok else _ASCII
    out: list[str] = []
    for blk in build_blocks(r, options):
        out += _text_block(blk, options.width, cs, first=not out or out[-1] == "")
    text = "\n".join(out).strip("\n") + "\n"
    return text if unicode_ok else text.encode("ascii", "replace").decode("ascii")


# ----------------------------------------------------------------------------- markdown
def _md(s: Any) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def _md_cell(c: Cell) -> str:
    if isinstance(c, Bar):
        k = round(max(0.0, min(1.0, c.frac)) * _BAR_WIDTH)
        return "`" + "█" * k + "░" * (_BAR_WIDTH - k) + "`"
    return _md("" if c is None else c)


def _md_block(blk: Block) -> list[str]:
    if isinstance(blk, Heading):
        return ["", "#" * blk.level + " " + blk.text, ""]
    if isinstance(blk, Verdict):
        icon = {"ok": "✅", "diff": "⚠️", "error": "❌"}[blk.level]
        return ["", f"> **{icon} {blk.text}**", f"> {blk.detail}" if blk.detail else "", ""]
    if isinstance(blk, Facts):
        return [f"- **{k}:** {_md(v)}" for k, v in blk.items]
    if isinstance(blk, Lines):
        pre = {"error": "❌ ", "warn": "⚠️ ", "note": "", "plain": ""}[blk.kind]
        return ["", *[f"- {pre}{_md(x)}" for x in blk.lines]]
    sep = {"l": ":---", "r": "---:", "c": ":---:"}
    out = [""]
    if blk.caption:
        out += [f"**{_md(blk.caption)}**", ""]
    out.append("| " + " | ".join(_md(h) for h in blk.headers) + " |")
    out.append("| " + " | ".join(sep[a] for a in blk.align) + " |")
    out += ["| " + " | ".join(_md_cell(c) for c in row) + " |" for row in blk.rows]
    return out


def render_markdown(r: ComparisonResult, options: ReportOptions) -> str:
    text = "\n".join(line for blk in build_blocks(r, options) for line in _md_block(blk))
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip() + "\n"


# ----------------------------------------------------------------------------- html
_CSS = """
:root{--bg:#f6f7f9;--card:#fff;--fg:#1c2330;--muted:#667085;--line:#e4e7ec;--accent:#3b6fd4;
--ok:#12805c;--ok-bg:#e3f6ee;--diff:#a15c07;--diff-bg:#fdf1dc;--error:#b42318;--error-bg:#fde7e5;--bar:#e4e7ec}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f141b;--card:#171e29;--fg:#e6e9ef;
--muted:#98a2b3;--line:#2a3342;--accent:#7aa2f7;--ok:#4fd1a1;--ok-bg:#12302a;--diff:#f2b45a;--diff-bg:#33280f;
--error:#ff8a80;--error-bg:#3a1b19;--bar:#2a3342}}
:root[data-theme="dark"]{--bg:#0f141b;--card:#171e29;--fg:#e6e9ef;--muted:#98a2b3;--line:#2a3342;--accent:#7aa2f7;
--ok:#4fd1a1;--ok-bg:#12302a;--diff:#f2b45a;--diff-bg:#33280f;--error:#ff8a80;--error-bg:#3a1b19;--bar:#2a3342}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1000px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:24px;margin:0 0 12px}
h2{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:30px 0 10px;
border-bottom:1px solid var(--line);padding-bottom:6px}
h3{font-size:14px;margin:18px 0 6px;font-weight:600}
dl{display:grid;grid-template-columns:max-content 1fr;gap:4px 18px;margin:8px 0}
dt{color:var(--muted)} dd{margin:0;word-break:break-word}
.verdict{border-radius:10px;padding:14px 18px;margin:16px 0;border:1px solid var(--line)}
.verdict b{font-size:17px;display:block}
.verdict.ok{background:var(--ok-bg);color:var(--ok)} .verdict.diff{background:var(--diff-bg);color:var(--diff)}
.verdict.error{background:var(--error-bg);color:var(--error)}
.verdict span{color:var(--fg);opacity:.85}
ul.lines{margin:6px 0;padding-left:20px} ul.error li{color:var(--error)} ul.warn li{color:var(--diff)}
ul.note li{color:var(--muted)}
.tw{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px;margin:6px 0 14px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:7px 12px;text-align:left;white-space:nowrap;border-bottom:1px solid var(--line)}
th{font-size:12px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
tr:last-child td{border-bottom:0} td.r,th.r{text-align:right} td.c,th.c{text-align:center}
.bar{width:120px;height:8px;border-radius:4px;background:var(--bar);overflow:hidden}
.bar i{display:block;height:100%;background:var(--accent)}
"""


def _html_cell(c: Cell) -> str:
    if isinstance(c, Bar):
        w = 0.0 if c.frac <= 0 else max(2.0, min(100.0, c.frac * 100))
        return f'<div class="bar"><i style="width:{w:.1f}%"></i></div>'
    return _html.escape("" if c is None else str(c)).replace("  ", "&nbsp;&nbsp;")


def _html_block(blk: Block) -> str:
    e = _html.escape
    if isinstance(blk, Heading):
        return f"<h{blk.level}>{e(blk.text)}</h{blk.level}>"
    if isinstance(blk, Verdict):
        icon = {"ok": "&#10004;", "diff": "&#9888;", "error": "&#10006;"}[blk.level]
        detail = f"<span>{e(blk.detail)}</span>" if blk.detail else ""
        return f'<div class="verdict {blk.level}"><b>{icon} {e(blk.text)}</b>{detail}</div>'
    if isinstance(blk, Facts):
        return "<dl>" + "".join(f"<dt>{e(k)}</dt><dd>{e(v)}</dd>" for k, v in blk.items) + "</dl>"
    if isinstance(blk, Lines):
        return f'<ul class="lines {blk.kind}">' + "".join(f"<li>{e(x)}</li>" for x in blk.lines) + "</ul>"
    head = "".join(f'<th class="{a}">{e(h)}</th>' for h, a in zip(blk.headers, blk.align, strict=True))
    align = blk.align

    def row(cells: Sequence[Cell]) -> str:
        return "".join(f'<td class="{a}">{_html_cell(c)}</td>' for c, a in zip(cells, align, strict=True))

    rows = "".join(f"<tr>{row(r)}</tr>" for r in blk.rows)
    cap = f"<h3>{e(blk.caption)}</h3>" if blk.caption else ""
    return f'{cap}<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


def render_html(r: ComparisonResult, options: ReportOptions) -> str:
    body = "\n".join(_html_block(blk) for blk in build_blocks(r, options, full=True))
    title = _html.escape(f"nini-datacomp: {r.names[0]} vs {r.names[1]}")
    return (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{title}</title><style>{_CSS}</style></head><body><main>\n{body}\n</main></body></html>\n"
    )


# ----------------------------------------------------------------------------- saída
def render_json(r: ComparisonResult, options: ReportOptions) -> str:
    del options  # mesma assinatura dos outros formatos; o JSON traz tudo
    return r.to_json()


_RENDERERS: dict[Format, Callable[[ComparisonResult, ReportOptions], str]] = {
    "text": render_text,
    "markdown": render_markdown,
    "html": render_html,
    "json": render_json,
}


def safe_print(text: str) -> None:
    """``print`` que não quebra num console sem Unicode (o cmd antigo do Windows): troca o que não cabe por ``?``."""
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(enc, "replace").decode(enc))
